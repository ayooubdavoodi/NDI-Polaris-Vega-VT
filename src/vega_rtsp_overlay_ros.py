#!/usr/bin/env python3
from __future__ import annotations
import os
import math
import time
import threading
import queue
from typing import Optional, Tuple, Iterable, Dict, List
from datetime import datetime
import json

import numpy as np
import cv2

# ROS 2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# NDI
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
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0.0
    return np.array([x, y, z], dtype=np.float64)

def tranMat_to_pose(T: np.ndarray) -> np.ndarray:
    x = float(T[0,3]); y = float(T[1,3]); z = float(T[2,3])
    R = T[0:3, 0:3]
    euler = rotationMatrixToEulerAngles(R)
    roll, pitch, yaw = float(euler[0]), float(euler[1]), float(euler[2])
    return np.array([x, y, z, roll, pitch, yaw], dtype=np.float64)

def quat_to_R(qw: float, qx: float, qy: float, qz: float) -> Optional[np.ndarray]:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        return None
    qw, qx, qy, qz = (q / n).tolist()
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx*qx+qz*qz),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),     1 - 2*(qx*qx+qy*qy)]
    ], dtype=np.float64)
    return R if np.all(np.isfinite(R)) else None

def quat_to_T(qw: float, qx: float, qy: float, qz: float, x: float, y: float, z: float) -> Optional[np.ndarray]:
    R = quat_to_R(qw,qx,qy,qz)
    if R is None:
        return None
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return T


# ============================ camera model / projection ============================

def load_calibration(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def build_camera_model(cfg: Dict, invert_extrinsic: bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (K, dist, rvec, tvec) for cv2.projectPoints.
    - Extrinsic is camera <- tracker from VCU-0.Param.Lens.6D.* (q0,qx,qy,qz, tx,ty,tz) unless invert_extrinsic=True.
    - Intrinsics are adjusted for ROI binning and crop.
    """
    # Intrinsic
    fu, fv, u0, v0 = (cfg["intrinsic"][k] for k in ("fu","fv","u0","v0"))
    # Distortion
    k1, k2, k3 = (cfg["distortion"][k] for k in ("k1","k2","k3"))
    p1, p2     = (cfg["distortion"][k] for k in ("p1","p2"))
    # ROI/binning
    bin_x, bin_y = cfg["roi"]["binning_x"], cfg["roi"]["binning_y"]
    left, top    = cfg["roi"]["left"], cfg["roi"]["top"]

    fx = fu / bin_x
    fy = fv / bin_y
    cx = (u0 - left) / bin_x
    cy = (v0 - top)  / bin_y
    K    = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    # camera <- tracker
    q0, qx, qy, qz = (cfg["extrinsic"][k] for k in ("q0","qx","qy","qz"))
    tx, ty, tz     = (cfg["extrinsic"][k] for k in ("tx","ty","tz"))
    R_CT = quat_to_R(q0,qx,qy,qz)
    if R_CT is None:
        raise RuntimeError("Invalid 6D quaternion in calibration JSON.")
    t_CT = np.array([[tx],[ty],[tz]], dtype=np.float64)

    if invert_extrinsic:
        R_use = R_CT.T
        t_use = -R_CT.T @ t_CT
    else:
        R_use = R_CT
        t_use = t_CT

    rvec, _ = cv2.Rodrigues(R_use)
    return K, dist, rvec, t_use

def project_points_tracker(points_xyz_tracker: np.ndarray,
                           rvec: np.ndarray, tvec: np.ndarray,
                           K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    obj = np.asarray(points_xyz_tracker, dtype=np.float64).reshape(-1,1,3)
    img_pts, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return img_pts.reshape(-1,2)

def build_axes_points_in_tracker(qw: float, qx: float, qy: float, qz: float,
                                 x: float, y: float, z: float,
                                 axis_len_mm: float) -> Optional[np.ndarray]:
    R_tool = quat_to_R(qw,qx,qy,qz)
    if R_tool is None or not np.all(np.isfinite([x,y,z])):
        return None
    P0 = np.array([x,y,z], dtype=np.float64)
    Px = P0 + axis_len_mm * R_tool[:,0]
    Py = P0 + axis_len_mm * R_tool[:,1]
    Pz = P0 + axis_len_mm * R_tool[:,2]
    P  = np.vstack([P0, Px, Py, Pz])
    return P if np.all(np.isfinite(P)) else None

def draw_axes_at(frame: np.ndarray, uv: np.ndarray) -> None:
    if uv.shape != (4,2) or not np.all(np.isfinite(uv)):
        return
    (u0,v0), (ux,vx), (uy,vy), (uz,vz) = uv
    O = (int(round(u0)), int(round(v0)))
    X = (int(round(ux)), int(round(vx)))
    Y = (int(round(uy)), int(round(vy)))
    Z = (int(round(uz)), int(round(vz)))
    cv2.circle(frame, O, 4, (255,255,255), -1)
    cv2.line(frame, O, X, (  0,  0,255), 2)  # X red
    cv2.line(frame, O, Y, (  0,255,  0), 2)  # Y green
    cv2.line(frame, O, Z, (255,  0,  0), 2)  # Z blue


# ============================ simple “latest only” buffer ============================

class PoseBuffer:
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



# ============================ ROS2 Node ============================

class VegaTrackerNode(Node):
    """
    - Publishes 6-DoF pose (x,y,z, roll,pitch,yaw) to /KUL/marker/marker_tracking (Float64MultiArray)
    - Publishes BGR image to /image_polaris_vega (sensor_msgs/Image)
    - Overlays tool axes at tool center (projected by camera model)
    """
    def __init__(self):
        super().__init__("vega_polaris_node")

        # ---------- Parameters (declare + get) ----------
        self.declare_parameter("vega_ip", "169.254.7.143")
        self.declare_parameter("vega_port", 8765)
        self.declare_parameter("roms", ["/home/ayoob/Polaris_Vega_VT/marker_definition/Kia_phantom_marker.rom"])
        self.declare_parameter("rtsp_url", "rtsp://169.254.7.143:554/video")
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.declare_parameter("calib_json", os.path.join(repo_root, "setting", "vega_calibration.json"))
        self.declare_parameter("use_gst", True)
        self.declare_parameter("gst_latency_ms", 50)
        self.declare_parameter("invert_extrinsic", False)
        self.declare_parameter("axis_len_mm", 40.0)
        self.declare_parameter("draw_overlay", True)
        self.declare_parameter("publish_hz", 30.0)

        self.vega_ip   = self.get_parameter("vega_ip").get_parameter_value().string_value
        self.vega_port = int(self.get_parameter("vega_port").get_parameter_value().integer_value)
        self.roms      = list(self.get_parameter("roms").get_parameter_value().string_array_value)
        self.rtsp_url  = self.get_parameter("rtsp_url").get_parameter_value().string_value
        self.calib_json= self.get_parameter("calib_json").get_parameter_value().string_value
        self.use_gst   = bool(self.get_parameter("use_gst").get_parameter_value().bool_value)
        self.gst_lat   = int(self.get_parameter("gst_latency_ms").get_parameter_value().integer_value)
        self.invert_ex = bool(self.get_parameter("invert_extrinsic").get_parameter_value().bool_value)
        self.axis_len  = float(self.get_parameter("axis_len_mm").get_parameter_value().double_value)
        self.draw_overlay = bool(self.get_parameter("draw_overlay").get_parameter_value().bool_value)
        self.publish_hz   = float(self.get_parameter("publish_hz").get_parameter_value().double_value)

        # ---------- Publishers ----------
        qos = QoSProfile(depth=1)
        self.pub_tracking = self.create_publisher(Float64MultiArray, "/KUL/marker/marker_tracking", qos)
        self.img_pub      = self.create_publisher(Image, "/image_polaris_vega", qos)
        self.bridge       = CvBridge()

        # ---------- Camera model ----------
        try:
            cfg = load_calibration(self.calib_json)
            self.K, self.dist, self.rvec_CT, self.t_CT = build_camera_model(cfg, self.invert_ex)
            self.get_logger().info("Loaded camera model from JSON.")
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration JSON: {e}")
            raise

        # ---------- Tracker thread ----------
        self.pose_buf = PoseBuffer()
        self.tracker_thread = threading.Thread(
            target=self._tracker_loop, daemon=True
        )
        self.tracker_thread.start()

        # ---------- RTSP capture ----------
        self.cap = self._open_video(self.rtsp_url, self.use_gst, self.gst_lat)
        if not self.cap.isOpened():
            self.get_logger().error("Could not open RTSP stream.")
            raise RuntimeError("RTSP open failed")

        # ---------- Main timer ----------
        period = 1.0 / max(1.0, self.publish_hz)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info("Vega Polaris node started. Publishing pose and image.")

    # ----- tracker thread -----
    def _tracker_loop(self):
        settings = {
            "tracker type": "vega",
            "ip address": self.vega_ip,
            "port": self.vega_port,
            "romfiles": self.roms,
            "use quaternions": True,  # provides [qw,qx,qy,qz,x,y,z]
        }
        trk = None
        try:
            trk = NDITracker(settings)
            trk.start_tracking()
            self.get_logger().info("NDITracker started.")
            while rclpy.ok():
                ports, ts, frames, tracking, quality = trk.get_frame()
                if tracking:
                    self.pose_buf.update({
                        "time": time.time(),
                        "ports": ports,
                        "timestamps": ts,
                        "frames": frames,
                        "tracking": tracking,
                        "quality": quality,
                    })
                time.sleep(0.002)
        except Exception as e:
            self.get_logger().error(f"Tracker loop error: {e}")
        finally:
            if trk is not None:
                try:
                    trk.stop_tracking()
                except Exception:
                    pass
                try:
                    trk.close()
                except Exception:
                    pass
            self.get_logger().info("NDITracker stopped.")

    # ----- video -----
    def _gst_pipeline(self, url: str, latency_ms: int) -> str:
        return (f"rtspsrc location={url} latency={latency_ms} ! "
                f"rtph264depay ! h264parse ! avdec_h264 ! "
                f"videoconvert ! appsink drop=true sync=false max-buffers=1")

    def _open_video(self, rtsp_url: str, use_gst: bool, latency_ms: int) -> cv2.VideoCapture:
        if use_gst:
            return cv2.VideoCapture(self._gst_pipeline(rtsp_url, latency_ms), cv2.CAP_GSTREAMER)
        return cv2.VideoCapture(rtsp_url)

    # ----- timer: grab frame, overlay, publish pose+image -----
    def _on_timer(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return

        # latest pose
        sample = self.pose_buf.latest()
        pose_msg = Float64MultiArray()

        if sample and sample.get("tracking"):
            entry = np.asarray(sample["tracking"][0]).reshape(-1)
            if entry.size >= 7 and np.all(np.isfinite(entry[:7])):
                qw,qx,qy,qz,x,y,z = [float(entry[i]) for i in range(7)]

                # Convert quaternion -> T -> pose (x,y,z, r,p,y)
                T = quat_to_T(qw,qx,qy,qz,x,y,z)
                if T is not None and isRotationMatrix(T[:3,:3]):
                    pose = tranMat_to_pose(T)  # [x,y,z, roll,pitch,yaw]
                    pose_msg.data = pose.tolist()
                    self.pub_tracking.publish(pose_msg)

                    # Optional overlay at tool center in image:
                    if self.draw_overlay:
                        P = build_axes_points_in_tracker(qw,qx,qy,qz, x,y,z, self.axis_len)
                        if P is not None:
                            uv = project_points_tracker(P, self.rvec_CT, self.t_CT, self.K, self.dist)
                            if np.all(np.isfinite(uv)):
                                h,w = frame.shape[:2]
                                # avoid drawing if way offscreen (huge lines)
                                if not np.any((uv[:,0] < -2*w) | (uv[:,0] > 3*w) |
                                              (uv[:,1] < -2*h) | (uv[:,1] > 3*h)):
                                    draw_axes_at(frame, uv)

        # publish image
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.img_pub.publish(img_msg)

    # ----- clean up -----
    def destroy_node(self):
        try:
            if hasattr(self, "cap") and self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        super().destroy_node()


# ============================ main ============================

def main():
    rclpy.init()
    node = None
    try:
        node = VegaTrackerNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
