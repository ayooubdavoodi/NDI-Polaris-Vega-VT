#!/usr/bin/env python3
from __future__ import annotations
import os
import time
import math
import json
import threading
import queue
from typing import Dict, Tuple, Optional, Iterable

import numpy as np
import cv2
import trimesh

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from sksurgerynditracker.nditracker import NDITracker


# ============================ math helpers (your functions) ============================

def isRotationMatrix(R: np.ndarray) -> bool:
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R: np.ndarray) -> np.ndarray:
    assert isRotationMatrix(R)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0.0
    return np.array([x, y, z], dtype=np.float64)

def tranMat_to_pose(T: np.ndarray) -> np.ndarray:
    x, y, z = T[0,3], T[1,3], T[2,3]
    R = T[0:3, 0:3]
    euler = rotationMatrixToEulerAngles(R)
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    return np.array([x, y, z, roll, pitch, yaw], dtype=np.float64)

def quat_to_R(qw: float, qx: float, qy: float, qz: float) -> Optional[np.ndarray]:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        return None
    qw, qx, qy, qz = (q / n).tolist()
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),   1 - 2*(qx*qx+qy*qy)]
    ], dtype=np.float64)
    return R if np.all(np.isfinite(R)) else None


# ============================ projection helpers ============================

def load_calibration(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def build_camera_model(cfg: Dict, invert_extrinsic: bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (K, dist, rvec_CT, t_CT) for OpenCV projectPoints.
    - K: 3x3 intrinsics adjusted for ROI/binning.
    - dist: [k1,k2,p1,p2,k3]
    - rvec_CT: Rodrigues rotation for camera<-tracker
    - t_CT: 3x1 translation (mm) for camera<-tracker
    """
    # Intrinsics (full sensor)
    fu, fv, u0, v0 = (cfg["intrinsic"][k] for k in ("fu","fv","u0","v0"))
    # Distortion
    k1, k2, k3 = (cfg["distortion"][k] for k in ("k1","k2","k3"))
    p1, p2     = (cfg["distortion"][k] for k in ("p1","p2"))
    # ROI/binning
    bin_x, bin_y = cfg["roi"]["binning_x"], cfg["roi"]["binning_y"]
    left, top    = cfg["roi"]["left"], cfg["roi"]["top"]

    # Adjust intrinsics for crop & binning
    fx = fu / bin_x
    fy = fv / bin_y
    cx = (u0 - left) / bin_x
    cy = (v0 - top)  / bin_y

    K    = np.array([[fx, 0,  cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    # Extrinsic (camera <- tracker)
    q0, qx, qy, qz = (cfg["extrinsic"][k] for k in ("q0","qx","qy","qz"))
    tx, ty, tz     = (cfg["extrinsic"][k] for k in ("tx","ty","tz"))

    R_CT = quat_to_R(q0,qx,qy,qz)
    if R_CT is None:
        raise RuntimeError("Invalid VCU 6D quaternion in JSON.")
    t_CT = np.array([[tx],[ty],[tz]], dtype=np.float64)

    if invert_extrinsic:
        R_CT = R_CT.T
        t_CT = -R_CT @ t_CT

    rvec_CT, _ = cv2.Rodrigues(R_CT)
    return K, dist, rvec_CT, t_CT

def project_points_tracker(points_xyz_tracker: np.ndarray,
                           rvec_CT: np.ndarray,
                           t_CT: np.ndarray,
                           K: np.ndarray,
                           dist: np.ndarray) -> np.ndarray:
    obj = np.asarray(points_xyz_tracker, dtype=np.float64).reshape(-1,1,3)
    img_pts, _ = cv2.projectPoints(obj, rvec_CT, t_CT, K, dist)
    return img_pts.reshape(-1,2)


# ============================ stream utils ============================

def gst_pipeline(url: str, latency_ms: int = 50) -> str:
    return (f"rtspsrc location={url} latency={latency_ms} ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! "
            f"videoconvert ! appsink drop=true sync=false max-buffers=1")

class PoseBuffer:
    """Lock-free single-sample buffer (latest only)."""
    def __init__(self):
        self._q = queue.Queue(maxsize=1)
        self._last = None
    def update(self, item: Dict) -> None:
        try:
            if self._q.full():
                _ = self._q.get_nowait()
        except queue.Empty:
            pass
        self._q.put(item)
        self._last = item
    def latest(self) -> Optional[Dict]:
        try:
            while True:
                self._last = self._q.get_nowait()
        except queue.Empty:
            pass
        return self._last

def to_qtx(entry: Iterable[float]) -> Optional[Tuple[float,float,float,float,float,float,float]]:
    """
    Accepts tracker sample for one tool: expected [qw,qx,qy,qz,x,y,z].
    Returns tuple or None if invalid.
    """
    try:
        v = np.asarray(entry, dtype=np.float64).reshape(-1)
        if v.size < 7:
            return None
        vals = tuple(float(v[i]) for i in range(7))
        return vals if np.all(np.isfinite(vals)) else None
    except Exception:
        return None


# ============================ ROS2 Node ============================

class VegaOverlayNode(Node):
    def __init__(self,
                 calib_json: str,
                 vega_ip: str,
                 vega_port: int,
                 roms: list[str],
                 rtsp_url: str,
                 stl_path: str,
                 use_gst: bool = True,
                 latency_ms: int = 50,
                 invert_extrinsic: bool = False,
                 axis_len_mm: float = 40.0,
                 show_window: bool = False,
                 show_overlay: bool=False):
        super().__init__("vega_overlay_node")

        # --- Publishers ---
        self.pub_tracking = self.create_publisher(
            Float64MultiArray, "/marker/marker_tracking", QoSProfile(depth=10)
        )
        self.img_pub = self.create_publisher(
            Image, "/image_polaris_vega", QoSProfile(depth=1)
        )
        self.bridge = CvBridge()

        # --- Camera model ---
        cfg = load_calibration(calib_json)
        self.K, self.dist, self.rvec_CT, self.t_CT = build_camera_model(cfg, invert_extrinsic)

        # --- Load mesh (in tool frame, mm) ---
        if not os.path.isfile(stl_path):
            raise FileNotFoundError(f"STL not found: {stl_path}")
        mesh = trimesh.load(stl_path, force='mesh')
        self.V_tool = np.asarray(mesh.vertices, dtype=np.float64)  # (N,3)
        self.F = np.asarray(mesh.faces, dtype=np.int32)            # (M,3)

        # NOTE: If STL is not in tool frame or not in mm, fix here:
        # self.V_tool = (R_mt @ self.V_tool.T).T + t_mt
        # self.V_tool *= scale  # e.g. 1000 if meters->mm

        self.axis_len_mm = float(axis_len_mm)
        self.show_window = bool(show_window)
        self.show_overlay= bool(show_overlay)


        # --- Tracker thread ---
        self.pose_buffer = PoseBuffer()
        self.tracker = NDITracker({
            "tracker type": "vega",
            "ip address": vega_ip,
            "port": vega_port,
            "romfiles": roms,
            "use quaternions": True,
        })
        self.tracker.start_tracking()
        self.get_logger().info(f"Tools: {self.tracker.get_tool_descriptions()}")

        self.tracker_thread = threading.Thread(target=self._tracker_loop, daemon=True)
        self.tracker_thread.start()

        # --- Video capture thread ---
        self.cap = (cv2.VideoCapture(gst_pipeline(rtsp_url, latency_ms), cv2.CAP_GSTREAMER)
                    if use_gst else cv2.VideoCapture(rtsp_url))
        if not self.cap.isOpened():
            raise RuntimeError("Could not open RTSP stream.")

        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

        self.get_logger().info("Vega overlay node started. Publishing pose and image.")

    # --- Threads ---

    def _tracker_loop(self):
        try:
            while rclpy.ok():
                ports, ts, frames, tracking, quality = self.tracker.get_frame()
                if tracking:
                    self.pose_buffer.update({
                        "time": time.time(),
                        "ports": ports,
                        "timestamps": ts,
                        "frames": frames,
                        "tracking": tracking,   # list of [qw,qx,qy,qz,x,y,z]
                        "quality": quality,
                    })
                time.sleep(0.002)
        finally:
            # node shutdown will close in destroy_node
            pass

    def _video_loop(self):
        last_status = "MISSING"
        while rclpy.ok():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            sample = self.pose_buffer.latest()
            if sample and sample.get("tracking"):
                qtx = to_qtx(sample["tracking"][0])
                if qtx is not None:
                    # --- publish pose (tracker frame) ---
                    qw,qx,qy,qz,x,y,z = qtx
                    R_tool = quat_to_R(qw,qx,qy,qz)
                    if R_tool is not None:
                        T_tool = np.eye(4, dtype=np.float64)
                        T_tool[:3,:3] = R_tool
                        T_tool[:3, 3] = np.array([x,y,z], dtype=np.float64)
                        pose_vec = tranMat_to_pose(T_tool)  # [x,y,z,roll,pitch,yaw]
                        msg = Float64MultiArray()
                        msg.data = pose_vec.tolist()
                        self.pub_tracking.publish(msg)

                        # Also draw tool axes (nice visual)
                        frame = self._draw_axes(frame, R_tool, np.array([x,y,z], dtype=np.float64))

                        if self.show_overlay:
                            # --- overlay STL mesh (filled triangles, painter’s) ---
                            frame = self._overlay_mesh(frame, R_tool, np.array([x,y,z], dtype=np.float64))

                        last_status = "LIVE"
                    else:
                        last_status = "BAD_QUAT"
                else:
                    last_status = "EMPTY_QTX"
            else:
                last_status = "MISSING"

            if self.show_window:
                cv2.imshow("Vega RTSP + STL overlay", frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                    rclpy.shutdown()

            # --- publish image ---
            cv2.putText(frame, last_status, (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if last_status!="LIVE" else (0,255,0), 2,
                            cv2.LINE_AA)
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "vega_camera"
            self.img_pub.publish(img_msg)

        # loop ended
        self.cap.release()
        cv2.destroyAllWindows()

    # --- rendering ---

    def _overlay_mesh(self, frame: np.ndarray, R_tool: np.ndarray, t_tool: np.ndarray,
                      color=(255, 0, 255), alpha=0.35) -> np.ndarray:
        """
        Simple painter’s algorithm: sort triangles by average Z in camera frame
        and fill back-to-front. Uses Vega distortion & intrinsics for projection.
        """
        # tracker->camera
        R_CT, _ = cv2.Rodrigues(self.rvec_CT)
        t_CT = self.t_CT.reshape(3)

        # tool->tracker
        V_trk = (R_tool @ self.V_tool.T).T + t_tool.reshape(1,3)
        # tracker->camera
        V_cam = (R_CT @ V_trk.T).T + t_CT.reshape(1,3)

        z = V_cam[:,2]
        valid = np.all(z[self.F] > 1e-6, axis=1)
        Fv = self.F[valid]
        if Fv.size == 0:
            return frame

        depth = z[Fv].mean(axis=1)
        order = np.argsort(-depth)  # back-to-front
        Fv = Fv[order]

        # Project all verts once
        pts2d, _ = cv2.projectPoints(
            V_cam.reshape(-1,1,3),
            np.zeros((3,1)), np.zeros((3,1)),
            self.K, self.dist
        )
        pts2d = pts2d.reshape(-1,2)

        h, w = frame.shape[:2]
        overlay = frame.copy()
        for tri in Fv:
            poly = pts2d[tri].astype(np.int32)
            # Optional: coarse clip, or just draw — OpenCV will clip
            cv2.fillConvexPoly(overlay, poly, color)
        blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return blended

    def _draw_axes(self, frame: np.ndarray, R_tool: np.ndarray, t_tool: np.ndarray) -> np.ndarray:
        L = self.axis_len_mm
        P0 = t_tool
        Px = P0 + L * R_tool[:,0]
        Py = P0 + L * R_tool[:,1]
        Pz = P0 + L * R_tool[:,2]
        uv = project_points_tracker(np.vstack([P0,Px,Py,Pz]), self.rvec_CT, self.t_CT, self.K, self.dist)
        if not np.all(np.isfinite(uv)): return frame
        (u0,v0), (ux,vx), (uy,vy), (uz,vz) = uv
        O = (int(round(u0)), int(round(v0)))
        X = (int(round(ux)), int(round(vx)))
        Y = (int(round(uy)), int(round(vy)))
        Z = (int(round(uz)), int(round(vz)))
        cv2.circle(frame, O, 4, (255,255,255), -1)
        cv2.line(frame, O, X, (  0,  0,255), 2)
        cv2.line(frame, O, Y, (  0,255,  0), 2)
        cv2.line(frame, O, Z, (255,  0,  0), 2)
        return frame

    # --- cleanup ---

    def destroy_node(self):
        try:
            self.tracker.stop_tracking()
        except Exception:
            pass
        try:
            self.tracker.close()
        except Exception:
            pass
        super().destroy_node()

        try:
            if self.tracker_thread.is_alive():
                self.tracker_thread.join(timeout=1.0)
            if self.video_thread.is_alive():
                self.video_thread.join(timeout=1.0)
        except Exception:
            pass


# ============================ main ============================

def main():
    # ---- edit your paths/addresses here ----
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CALIB_JSON = os.path.join(REPO_ROOT, "setting", "vega_calibration.json")
    VEGA_IP, VEGA_PORT = "169.254.7.143", 8765
    ROMS = ["/home/ayoob/Polaris_Vega_VT/marker_definition/Kia_phantom_marker.rom"]
    RTSP_URL = "rtsp://169.254.7.143:554/video"
    STL_PATH = "/home/ayoob/Polaris_Vega_VT/phantom/Phantom_v1_V2_new.STL"

    rclpy.init()
    node = None
    try:
        node = VegaOverlayNode(
            calib_json=CALIB_JSON,
            vega_ip=VEGA_IP,
            vega_port=VEGA_PORT,
            roms=ROMS,
            rtsp_url=RTSP_URL,
            stl_path=STL_PATH,
            use_gst=True,
            latency_ms=50,
            invert_extrinsic=False,   # flip if overlay looks mirrored
            axis_len_mm=40.0,
            show_window=False,         # set True if you want a local preview window
            show_overlay=False         # set True if you want to show ovelay of stl file
        )
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
