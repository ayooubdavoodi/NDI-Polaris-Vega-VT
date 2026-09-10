#!/usr/bin/env python3
from __future__ import annotations
import os
import math
import json
import threading
import time
from typing import Dict, Tuple, Optional

import numpy as np
import cv2
import trimesh

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ============================ math / calibration ============================

def is_rotation_matrix(R: np.ndarray) -> bool:
    Rt = R.T
    shouldBeI = Rt @ R
    I = np.eye(3, dtype=R.dtype)
    return np.linalg.norm(I - shouldBeI) < 1e-6

def euler_xyz_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic XYZ (roll, pitch, yaw), radians."""
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float64)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float64)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float64)
    R = Rz @ Ry @ Rx
    assert is_rotation_matrix(R)
    return R

def load_calibration(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def build_camera_model(cfg: Dict, invert_extrinsic: bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (K, dist, rvec_CT, t_CT, R_CT) for OpenCV projectPoints.
    - K: 3x3 intrinsics adjusted for ROI/binning.
    - dist: [k1,k2,p1,p2,k3]
    - rvec_CT: Rodrigues rotation for camera<-tracker
    - t_CT: 3x1 translation (mm) for camera<-tracker
    - R_CT: 3x3 rotation for camera<-tracker
    """
    fu, fv, u0, v0 = (cfg["intrinsic"][k] for k in ("fu","fv","u0","v0"))
    k1, k2, k3 = (cfg["distortion"][k] for k in ("k1","k2","k3"))
    p1, p2     = (cfg["distortion"][k] for k in ("p1","p2"))
    bin_x, bin_y = cfg["roi"]["binning_x"], cfg["roi"]["binning_y"]
    left, top    = cfg["roi"]["left"], cfg["roi"]["top"]

    fx = fu / bin_x
    fy = fv / bin_y
    cx = (u0 - left) / bin_x
    cy = (v0 - top)  / bin_y

    K    = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    dist = np.array([k1,k2,p1,p2,k3], dtype=np.float64)

    q0, qx, qy, qz = (cfg["extrinsic"][k] for k in ("q0","qx","qy","qz"))
    tx, ty, tz     = (cfg["extrinsic"][k] for k in ("tx","ty","tz"))

    q = np.array([q0,qx,qy,qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        raise RuntimeError("Invalid extrinsic quaternion in calibration JSON.")
    qw, qx, qy, qz = (q / n).tolist()
    R_CT = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),   1 - 2*(qx*qx+qy*qy)]
    ], dtype=np.float64)
    t_CT = np.array([[tx],[ty],[tz]], dtype=np.float64)

    if invert_extrinsic:
        R_CT = R_CT.T
        t_CT = -R_CT @ t_CT

    rvec_CT, _ = cv2.Rodrigues(R_CT)
    return K, dist, rvec_CT, t_CT, R_CT

# ============================ mesh decimation (Open3D) ============================

def decimate_with_open3d(vertices: np.ndarray, faces: np.ndarray, target_faces: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quadric decimation using Open3D. Returns (V,F) with ~target_faces faces.
    """
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh = mesh.simplify_quadric_decimation(int(target_faces))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.triangles, dtype=np.int32)
    return V, F

# ============================ GPU renderer (Open3D) ============================

class Open3DRenderer:
    """GPU renderer via Open3D OffscreenRenderer. Assumes pinhole camera (no distortion)."""
    def __init__(self, width: int, height: int, fx: float, fy: float, cx: float, cy: float,
                 face_color=(1.0, 0.0, 1.0, 0.35), near: float = 5.0, far: float = 3000.0):
        import open3d as o3d
        self.o3d = o3d
        self.width, self.height = width, height
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

        # Material record across versions
        self.mesh_name = "tool_mesh"
        self.mat = None
        try:
            MR = o3d.visualization.rendering.MaterialRecord
            self.mat = MR()
            if hasattr(self.mat, "shader"):
                self.mat.shader = "defaultUnlit"
            if hasattr(self.mat, "base_color"):
                self.mat.base_color = face_color
        except Exception:
            self.mat = None
        self._fallback_rgb = np.array(face_color[:3], dtype=np.float64)

        # Camera intrinsics (handle multiple Open3D versions)
        cam = self.renderer.scene.camera
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)

        ok = False
        # Overload #3: set_projection(K_3x3, width, height, near, far)
        try:
            cam.set_projection(K, float(width), float(height), float(near), float(far))
            ok = True
        except Exception:
            pass

        if not ok:
            # Try Projection object
            try:
                Projection = o3d.visualization.rendering.Camera.Projection
                proj = Projection(
                    Projection.PROJECTION_PERSPECTIVE,
                    float(near), float(far),
                    float(width), float(height),
                    float(fx), float(fy), float(cx), float(cy)
                )
                cam.set_projection(proj, float(width), float(height), float(near), float(far))
                ok = True
            except Exception:
                pass

        if not ok:
            # FOV-based (approximate, ignores principal point)
            fovy = 2.0 * math.degrees(math.atan2(0.5 * height, fy))
            FovType = o3d.visualization.rendering.Camera.FovType
            try:
                cam.set_projection(float(fovy), 0.0, 0.0, 0.0, FovType.Vertical)
                ok = True
            except Exception:
                pass

        if not ok:
            raise RuntimeError("Open3D camera.set_projection: no compatible overload found")

        # Transparent background for alpha compositing
        self.renderer.scene.set_background(np.array([0, 0, 0, 0], dtype=np.float32))

    def add_mesh(self, vertices: np.ndarray, faces: np.ndarray):
        mesh = self.o3d.geometry.TriangleMesh(
            self.o3d.utility.Vector3dVector(vertices.astype(np.float64)),
            self.o3d.utility.Vector3iVector(faces.astype(np.int32))
        )
        mesh.compute_vertex_normals()
        if self.mat is None:
            mesh.paint_uniform_color(self._fallback_rgb)
            self.renderer.scene.add_geometry(self.mesh_name, mesh, self.o3d.visualization.rendering.MaterialRecord())
        else:
            self.renderer.scene.add_geometry(self.mesh_name, mesh, self.mat)

    def set_mesh_pose_camframe(self, T_cam_tool: np.ndarray):
        self.renderer.scene.set_geometry_transform(self.mesh_name, T_cam_tool.astype(np.float32))

    def render_rgba(self) -> np.ndarray:
        img = self.renderer.render_to_image()
        arr = np.asarray(img)
        if arr.shape[-1] == 3:
            rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
            rgba[..., :3] = arr
            return rgba
        return arr

# ============================ Overlay Node (threaded) ============================

class FastOverlayNode(Node):
    def __init__(self,
                 calib_json: str,
                 stl_path: str,
                 pose_topic_name: str,
                 image_in_topic: str,
                 image_out_topic: str,
                 target_faces: int = 8000,
                 use_gpu: bool = True):
        super().__init__("fast_stl_overlay")

        # Topics
        self.declare_parameter("pose_topic", pose_topic_name)
        self.declare_parameter("image_in_topic", image_in_topic)
        self.declare_parameter("image_out_topic", image_out_topic)
        pose_topic      = self.get_parameter("pose_topic").value
        image_in_topic  = self.get_parameter("image_in_topic").value
        image_out_topic = self.get_parameter("image_out_topic").value

        # Calibration
        if not os.path.isfile(calib_json):
            raise FileNotFoundError(f"Calibration JSON not found: {calib_json}")
        cfg = load_calibration(calib_json)
        self.K, self.dist, self.rvec_CT, self.t_CT, self.R_CT = build_camera_model(cfg, invert_extrinsic=False)
        self.fx, self.fy, self.cx, self.cy = self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]

        # Load & decimate mesh
        if not os.path.isfile(stl_path):
            raise FileNotFoundError(f"STL not found: {stl_path}")
        mesh = trimesh.load(stl_path, force='mesh')
        V0 = np.asarray(mesh.vertices, dtype=np.float64)
        F0 = np.asarray(mesh.faces, dtype=np.int32)
        if target_faces is not None and F0.shape[0] > target_faces:
            try:
                V0, F0 = decimate_with_open3d(V0, F0, target_faces)
                self.get_logger().info(f"Decimated mesh to ~{F0.shape[0]} faces via Open3D.")
            except Exception as e:
                self.get_logger().warn(f"Open3D decimation failed; using original mesh: {e}")
        self.V_tool = V0
        self.F      = F0

        # I/O
        self.bridge = CvBridge()
        sensor_qos = QoSProfile(depth=1)
        self.create_subscription(Image, image_in_topic, self._on_image, sensor_qos)
        self.create_subscription(Float64MultiArray, pose_topic, self._on_pose, 10)
        self.image_pub = self.create_publisher(Image, image_out_topic, sensor_qos)

        # Buffers (latest-only)
        self._pose_lock = threading.Lock()
        self._pose: Optional[np.ndarray] = None  # [x,y,z,roll,pitch,yaw]
        self._img_lock  = threading.Lock()
        self._img: Optional[Tuple[np.ndarray, Image]] = None  # (bgr8 frame, original header)

        # GPU renderer (lazy init)
        self._gpu_enabled = bool(use_gpu)
        self._gpu: Optional[Open3DRenderer] = None

        # Processing thread
        self._stop = threading.Event()
        self._proc_th = threading.Thread(target=self._process_loop, daemon=True)
        self._proc_th.start()

        self.get_logger().info(
            f"Fast overlay node: pose={pose_topic}, in={image_in_topic}, out={image_out_topic}, "
            f"GPU={'on' if self._gpu_enabled else 'off'}."
        )

    # ---------- callbacks ----------
    def _on_pose(self, msg: Float64MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float64).reshape(-1)
        if data.size < 6 or not np.all(np.isfinite(data[:6])):
            return
        with self._pose_lock:
            self._pose = data[:6].copy()

    def _on_image(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge conversion failed: {e}")
            return
        with self._img_lock:
            self._img = (frame, msg)

    # ---------- processing thread ----------
    def _process_loop(self):
        last_warn_no_pose = 0.0
        while rclpy.ok() and not self._stop.is_set():
            with self._img_lock:
                item = self._img
            if item is None:
                time.sleep(0.001)
                continue
            frame, header_msg = item

            with self._pose_lock:
                pose = None if self._pose is None else self._pose.copy()

            if pose is None:
                t = time.time()
                if t - last_warn_no_pose > 1.0:
                    last_warn_no_pose = t
                    self.get_logger().debug("No pose yet; republishing passthrough.")
                self._publish(frame, header_msg)
                continue

            # Tool pose in camera frame (mm)
            x, y, z, roll, pitch, yaw = pose.tolist()
            R_tool = euler_xyz_to_R(roll, pitch, yaw)
            t_tool = np.array([x, y, z], dtype=np.float64)
            R_cam_tool = self.R_CT @ R_tool
            t_cam_tool = (self.R_CT @ t_tool.reshape(3,1)).reshape(3) + self.t_CT.reshape(3)

            # GPU overlay or CPU wireframe fallback
            try:
                out = self._overlay_gpu(frame, R_cam_tool, t_cam_tool)
            except Exception as e:
                if self._gpu_enabled:
                    self.get_logger().warn(f"GPU overlay failed ({e}); falling back to CPU wireframe.")
                    self._gpu_enabled = False
                out = self._overlay_cpu_wireframe(frame, R_cam_tool, t_cam_tool)

            self._publish(out, header_msg)

        self.get_logger().info("Processing thread stopped.")

    # ---------- overlay implementations ----------
    def _overlay_gpu(self, frame_bgr: np.ndarray, R_cam_tool: np.ndarray, t_cam_tool: np.ndarray) -> np.ndarray:
        if not self._gpu_enabled:
            raise RuntimeError("GPU disabled")
        h, w = frame_bgr.shape[:2]
        if self._gpu is None:
            self._gpu = Open3DRenderer(w, h, self.fx, self.fy, self.cx, self.cy,
                                       face_color=(1.0, 0.0, 1.0, 0.35),
                                       near=5.0, far=5000.0)
            self._gpu.add_mesh(self.V_tool, self.F)

        # 4x4 transform (camera <- tool)
        T = np.eye(4, dtype=np.float32)
        T[:3,:3] = R_cam_tool.astype(np.float32)
        T[:3, 3] = t_cam_tool.astype(np.float32)
        self._gpu.set_mesh_pose_camframe(T)

        rgba = self._gpu.render_rgba()  # HxWx4 uint8
        overlay_rgb = rgba[..., :3].astype(np.float32)
        alpha = (rgba[..., 3:4].astype(np.float32) / 255.0)  # HxWx1
        base = frame_bgr.astype(np.float32)
        comp = overlay_rgb * alpha + base * (1.0 - alpha)
        return comp.astype(np.uint8)

    def _overlay_cpu_wireframe(self, frame_bgr: np.ndarray, R_cam_tool: np.ndarray, t_cam_tool: np.ndarray) -> np.ndarray:
        V_cam = (R_cam_tool @ self.V_tool.T).T + t_cam_tool.reshape(1,3)
        pts2d, _ = cv2.projectPoints(V_cam.reshape(-1,1,3),
                                     np.zeros((3,1)), np.zeros((3,1)),
                                     self.K, self.dist)
        pts2d = pts2d.reshape(-1,2)
        overlay = frame_bgr.copy()
        for tri in self.F:
            poly = pts2d[tri]
            if not np.all(np.isfinite(poly)):
                continue
            cv2.polylines(overlay, [poly.astype(np.int32)], isClosed=True, color=(255,0,255), thickness=1)
        return overlay

    # ---------- publish ----------
    def _publish(self, frame_bgr: np.ndarray, src_header: Image) -> None:
        try:
            out_msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge encoding failed: {e}")
            return
        out_msg.header = src_header.header  # preserve timestamp & frame_id
        self.image_pub.publish(out_msg)

    # ---------- cleanup ----------
    def destroy_node(self):
        self._stop.set()
        try:
            if hasattr(self, "_proc_th") and self._proc_th.is_alive():
                self._proc_th.join(timeout=1.0)
        except Exception:
            pass
        super().destroy_node()

# ============================ main ============================

def main():
    rclpy.init()

    # Fixed paths
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CALIB_JSON = os.path.join(REPO_ROOT, "setting", "vega_calibration.json")
    STL_PATH = "/home/ayoob/Polaris_Vega_VT/phantom/Phantom_v1_V2_new.STL"

    # Topics
    pose_topic_name = "/marker/marker_tracking"
    image_in_topic  = "/image_polaris_vega"
    image_out_topic = "/image_polaris_vega_overlay"

    node = None
    try:
        node = FastOverlayNode(
            calib_json=CALIB_JSON,
            stl_path=STL_PATH,
            pose_topic_name=pose_topic_name,
            image_in_topic=image_in_topic,
            image_out_topic=image_out_topic,
            target_faces=2000,   # adjust lower/higher as needed
            use_gpu=True         # set False to force CPU wireframe
        )
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
