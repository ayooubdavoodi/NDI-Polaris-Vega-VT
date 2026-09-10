#!/usr/bin/env python3
from __future__ import annotations
import os
import time
import math
import json
import threading
import queue
from datetime import datetime
from typing import Dict, Tuple, Optional, Iterable, List

import numpy as np
import cv2
from sksurgerynditracker.nditracker import NDITracker


# ============================ Pure/utility functions ============================

def load_calibration(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def quat_to_R(qw: float, qx: float, qy: float, qz: float) -> Optional[np.ndarray]:
    """Quaternion (w,x,y,z) -> 3x3 rotation. Returns None if invalid."""
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

def build_camera_model(cfg: Dict, invert_extrinsic: bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (K, dist, rvec, tvec) for OpenCV projectPoints.
    rvec/tvec represent the transform used by projectPoints. By default this is camera<-tracker.
    If invert_extrinsic=True, we invert the VCU-0 6D so you can quickly check handedness.
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

    # Extrinsic (camera <- tracker) from VCU-0.Param.Lens.6D.*
    q0, qx, qy, qz = (cfg["extrinsic"][k] for k in ("q0","qx","qy","qz"))
    tx, ty, tz     = (cfg["extrinsic"][k] for k in ("tx","ty","tz"))

    R_CT = quat_to_R(q0,qx,qy,qz)
    if R_CT is None:
        raise RuntimeError("Invalid VCU 6D quaternion in JSON.")
    t_CT = np.array([[tx],[ty],[tz]], dtype=np.float64)

    if invert_extrinsic:
        # Invert the transform for debugging (tracker<-camera)
        R_use = R_CT.T
        t_use = -R_CT.T @ t_CT
    else:
        R_use = R_CT
        t_use = t_CT

    rvec, _ = cv2.Rodrigues(R_use)
    return K, dist, rvec, t_use

def project_points_tracker(points_xyz_tracker: np.ndarray,
                           rvec: np.ndarray,
                           tvec: np.ndarray,
                           K: np.ndarray,
                           dist: np.ndarray) -> np.ndarray:
    """
    Project 3D tracker points into the cropped/binned camera image.
    points_xyz_tracker: (N,3) mm in tracker coords matching rvec/tvec convention.
    Returns (N,2) pixel coords.
    """
    obj = np.asarray(points_xyz_tracker, dtype=np.float64).reshape(-1,1,3)
    img_pts, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return img_pts.reshape(-1,2)

def build_axes_points_in_tracker(qw: float, qx: float, qy: float, qz: float,
                                 x: float, y: float, z: float,
                                 axis_len_mm: float) -> Optional[np.ndarray]:
    """
    From tool pose (quaternion + translation), build 4 points in tracker frame:
    center P0 and endpoints of local X,Y,Z axes (Px,Py,Pz).
    Returns (4,3) or None.
    """
    R_tool = quat_to_R(qw,qx,qy,qz)
    if R_tool is None or not np.all(np.isfinite([x,y,z])):
        return None
    P0 = np.array([x,y,z], dtype=np.float64)
    Px = P0 + axis_len_mm * R_tool[:,0]
    Py = P0 + axis_len_mm * R_tool[:,1]
    Pz = P0 + axis_len_mm * R_tool[:,2]
    P  = np.vstack([P0, Px, Py, Pz])
    return P if np.all(np.isfinite(P)) else None

def draw_axes_at(frame: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """
    Draws axes given uv for [P0,Px,Py,Pz] in pixels.
    Mutates and returns the frame.
    """
    if uv.shape != (4,2) or not np.all(np.isfinite(uv)):
        return frame
    (u0,v0), (ux,vx), (uy,vy), (uz,vz) = uv
    O = (int(round(u0)), int(round(v0)))
    X = (int(round(ux)), int(round(vx)))
    Y = (int(round(uy)), int(round(vy)))
    Z = (int(round(uz)), int(round(vz)))
    cv2.circle(frame, O, 4, (255,255,255), -1)
    cv2.line(frame, O, X, (  0,  0,255), 2)  # X red
    cv2.line(frame, O, Y, (  0,255,  0), 2)  # Y green
    cv2.line(frame, O, Z, (255,  0,  0), 2)  # Z blue
    return frame

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

def gst_pipeline(url: str, latency_ms: int = 50) -> str:
    return (f"rtspsrc location={url} latency={latency_ms} ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! "
            f"videoconvert ! appsink drop=true sync=false max-buffers=1")


# ============================ IO / threading ============================

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

def tracker_loop(vega_ip: str,
                 vega_port: int,
                 roms: List[str],
                 pose_buffer: PoseBuffer,
                 use_quats: bool = True) -> None:
    """Continuously push latest frame to buffer."""
    settings = {
        "tracker type": "vega",
        "ip address": vega_ip,
        "port": vega_port,
        "romfiles": roms,
        "use quaternions": use_quats,
    }
    trk = NDITracker(settings)
    trk.start_tracking()
    try:
        while True:
            ports, ts, frames, tracking, quality = trk.get_frame()
            if tracking:
                pose_buffer.update({
                    "time": time.time(),
                    "ports": ports,
                    "timestamps": ts,
                    "frames": frames,
                    "tracking": tracking,   # list of [qw,qx,qy,qz,x,y,z]
                    "quality": quality,
                })
            time.sleep(0.002)
    except Exception as e:
        print(f"[tracker_loop] error: {e}")
    finally:
        try:
            trk.stop_tracking()
        finally:
            trk.close()


# ============================ Orchestration ============================

def overlay_one_tool(frame: np.ndarray,
                     sample: Dict,
                     K: np.ndarray,
                     dist: np.ndarray,
                     rvec: np.ndarray,
                     tvec: np.ndarray,
                     axis_len_mm: float = 40.0) -> Tuple[np.ndarray, bool]:
    """
    Projects the first tool’s center & axes. Returns (frame, drew).
    """
    if not sample or not sample.get("tracking"):
        return frame, False

    qtx = to_qtx(sample["tracking"][0])
    if qtx is None:
        return frame, False
    qw, qx, qy, qz, x, y, z = qtx

    P = build_axes_points_in_tracker(qw,qx,qy,qz, x,y,z, axis_len_mm)
    if P is None:
        return frame, False

    uv = project_points_tracker(P, rvec, tvec, K, dist)
    if not np.all(np.isfinite(uv)):
        return frame, False

    # Optionally reject projections far outside the image to avoid giant lines
    h, w = frame.shape[:2]
    if np.any((uv[:,0] < -2*w) | (uv[:,0] > 3*w) | (uv[:,1] < -2*h) | (uv[:,1] > 3*h)):
        return frame, False

    frame = draw_axes_at(frame, uv)

    # HUD (optional)
    ts  = datetime.fromtimestamp(sample["time"]).strftime("%H:%M:%S.%f")[:-3]
    frm = sample["frames"][0] if len(sample["frames"]) else "?"
    qlt = sample["quality"][0] if len(sample["quality"]) else "?"
    cv2.putText(frame, f"t={ts} frame={frm} qual={qlt:3.4f}",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"pos(mm): x={x:7.2f} y={y:7.2f} z={z:7.2f}",
                (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    return frame, True

def open_video(rtsp_url: str, use_gst: bool = True, latency_ms: int = 50) -> cv2.VideoCapture:
    if use_gst:
        return cv2.VideoCapture(gst_pipeline(rtsp_url, latency_ms), cv2.CAP_GSTREAMER)
    return cv2.VideoCapture(rtsp_url)

def run_app(calib_json: str,
            vega_ip: str,
            vega_port: int,
            roms: List[str],
            rtsp_url: str,
            use_gst: bool = True,
            latency_ms: int = 50,
            invert_extrinsic: bool = False,
            axis_len_mm: float = 40.0) -> None:

    # Camera model
    cfg = load_calibration(calib_json)
    K, dist, rvec, tvec = build_camera_model(cfg, invert_extrinsic)

    # Tracker thread
    buf = PoseBuffer()
    threading.Thread(target=tracker_loop,
                     args=(vega_ip, vega_port, roms, buf, True),
                     daemon=True).start()

    # Video
    cap = open_video(rtsp_url, use_gst, latency_ms)
    if not cap.isOpened():
        print("ERROR: Could not open RTSP stream.")
        return

    cv2.setUseOptimized(True)
    print("Press ESC or q to quit.")
    last_status = "MISSING"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            sample = buf.latest()
            frame, drew = overlay_one_tool(frame, sample, K, dist, rvec, tvec, axis_len_mm)

            if not drew:
                cv2.putText(frame, last_status, (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            else:
                last_status = "LIVE"

            cv2.imshow("Vega RTSP + Tool Axes (projected)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ============================ Entry point ============================

if __name__ == "__main__":
    # ---- set your paths/addresses here ----
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CALIB_JSON = os.path.join(REPO_ROOT, "setting", "vega_calibration.json")
    VEGA_IP, VEGA_PORT = "169.254.7.143", 8765
    ROMS: List[str] = ["/home/ayoob/Polaris_Vega_VT/marker_definition/Kia_phantom_marker.rom"]
    RTSP_URL = "rtsp://169.254.7.143:554/video"

    run_app(
        calib_json=CALIB_JSON,
        vega_ip=VEGA_IP,
        vega_port=VEGA_PORT,
        roms=ROMS,
        rtsp_url=RTSP_URL,
        use_gst=True,
        latency_ms=50,
        invert_extrinsic=False,   # set True if overlay looks mirrored/flipped
        axis_len_mm=40.0
    )
