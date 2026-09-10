#!/usr/bin/env python3
import os
import time, threading, queue, math
from datetime import datetime
import numpy as np
import cv2
from sksurgerynditracker.nditracker import NDITracker
import json
import numpy as np
import cv2


# ========= Calibration from your Vega (edit if different) =========
# Load JSON
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_REPO_ROOT, "setting", "vega_calibration.json"), "r") as f:
    cfg = json.load(f)


# Extract
q0, qx, qy, qz = (cfg["extrinsic"][k] for k in ("q0","qx","qy","qz"))
tx, ty, tz     = (cfg["extrinsic"][k] for k in ("tx","ty","tz"))

fu, fv, u0, v0 = (cfg["intrinsic"][k] for k in ("fu","fv","u0","v0"))

k1, k2, k3 = (cfg["distortion"][k] for k in ("k1","k2","k3"))
p1, p2     = (cfg["distortion"][k] for k in ("p1","p2"))

bin_x, bin_y = cfg["roi"]["binning_x"], cfg["roi"]["binning_y"]
left, top    = cfg["roi"]["left"], cfg["roi"]["top"]

# ---- Build matrices ----
def quat_to_R(qw,qx,qy,qz):
    q = np.array([qw,qx,qy,qz], dtype=np.float64)
    n = np.linalg.norm(q)
    qw,qx,qy,qz = (q/n).tolist()
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)]
    ], dtype=np.float64)

R_CT = quat_to_R(q0,qx,qy,qz)
t_CT = np.array([[tx],[ty],[tz]], dtype=np.float64)

# Invert to tracker <- camera, then back to camera <- tracker to test direction
# R_TC = R_CT.T
# t_TC = -R_CT.T @ t_CT
# If you discover you actually needed tracker<-camera, use rvec/tvec from R_TC,t_TC instead.


rvec_CT, _ = cv2.Rodrigues(R_CT)

# Adjust intrinsics for ROI + binning
fx = fu / bin_x
fy = fv / bin_y
u0_adj = (u0 - left) / bin_x
v0_adj = (v0 - top)  / bin_y

K = np.array([[fx,0,u0_adj],
              [0,fy,v0_adj],
              [0,0,1]], dtype=np.float64)

dist = np.array([k1,k2,p1,p2,k3], dtype=np.float64)

# ================================================================

VEGA_IP, VEGA_PORT = "169.254.7.143", 8765
ROMS = ["/home/ayoob/Polaris_Vega_VT/marker_definition/Kia_phantom_marker.rom"]
RTSP_URL = "rtsp://169.254.7.143:554/video"
USE_GSTREAMER = True
GST_LATENCY_MS = 50

def gst_pipeline(url: str) -> str:
    return (f"rtspsrc location={url} latency={GST_LATENCY_MS} ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! "
            f"videoconvert ! appsink drop=true sync=false max-buffers=1")

def quat_to_R(qw,qx,qy,qz):
    q = np.array([qw,qx,qy,qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12: return None
    qw,qx,qy,qz = (q/n).tolist()
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)]
    ], dtype=np.float64)
    return R if np.all(np.isfinite(R)) else None

# Camera intrinsics/distortion matrices for OpenCV
CAM_MTX = np.array([[FX, 0, CX],
                    [0, FY, CY],
                    [0,  0,  1]], dtype=np.float64)
DIST_COEFFS = np.array([K1, K2, P1, P2, K3], dtype=np.float64)

# R_cam_tracker, t_cam_tracker (mm)
R_CT = quat_to_R(Q0, QX, QY, QZ)
t_CT = np.array([[TX],[TY],[TZ]], dtype=np.float64)
if R_CT is None:
    raise RuntimeError("Invalid camera<-tracker quaternion from VCU params.")
rvec_CT, _ = cv2.Rodrigues(R_CT)  # OpenCV wants Rodrigues

class PoseBuffer:
    def __init__(self):
        self._q = queue.Queue(maxsize=1); self._last=None
    def update(self, item):
        try:
            if self._q.full(): _ = self._q.get_nowait()
        except queue.Empty: pass
        self._q.put(item); self._last=item
    def latest(self):
        try:
            while True: self._last = self._q.get_nowait()
        except queue.Empty: pass
        return self._last

def tracker_thread(buf: PoseBuffer):
    settings = {"tracker type":"vega","ip address":VEGA_IP,"port":VEGA_PORT,
                "romfiles":ROMS,"use quaternions":True}
    trk = NDITracker(settings); trk.start_tracking()
    print("Tools:", trk.get_tool_descriptions())
    try:
        while True:
            ports, ts, frames, tracking, quality = trk.get_frame()
            if tracking:
                buf.update({"time":time.time(),"ports":ports,"timestamps":ts,
                            "frames":frames,"tracking":tracking,"quality":quality})
            time.sleep(0.002)
    finally:
        trk.stop_tracking(); trk.close()

def to_qtx(entry):
    try:
        v = np.asarray(entry).reshape(-1)
        if v.size >= 7:
            vals = [float(v[i]) for i in range(7)]
            return vals if np.all(np.isfinite(vals)) else None
        if np.asarray(entry).shape==(4,4):
            T = np.asarray(entry, dtype=np.float64)
            R, t = T[:3,:3], T[:3,3]
            if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))): return None
            tr = float(np.trace(R))
            if tr>0:
                S = math.sqrt(tr+1.0)*2; qw=0.25*S
                qx=(R[2,1]-R[1,2])/S; qy=(R[0,2]-R[2,0])/S; qz=(R[1,0]-R[0,1])/S
            else:
                if (R[0,0]>R[1,1]) and (R[0,0]>R[2,2]):
                    S=math.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2
                    qw=(R[2,1]-R[1,2])/S; qx=0.25*S
                    qy=(R[0,1]+R[1,0])/S; qz=(R[0,2]+R[2,0])/S
                elif R[1,1]>R[2,2]:
                    S=math.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2
                    qw=(R[0,2]-R[2,0])/S; qx=(R[0,1]+R[1,0])/S
                    qy=0.25*S; qz=(R[1,2]+R[2,1])/S
                else:
                    S=math.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2
                    qw=(R[1,0]-R[0,1])/S; qx=(R[0,2]+R[2,0])/S
                    qy=(R[1,2]+R[2,1])/S; qz=0.25*S
            q = np.array([qw,qx,qy,qz]); q/=np.linalg.norm(q)
            x,y,z = float(t[0]),float(t[1]),float(t[2])
            vals = [q[0],q[1],q[2],q[3],x,y,z]
            return vals if np.all(np.isfinite(vals)) else None
    except Exception:
        return None
    return None

def project_points_tracker(points_xyz_tracker):
    """
    points_xyz_tracker: (N,3) in tracker coords [mm].
    Uses camera<-tracker extrinsic to project into image.
    Returns (N,2) float pixel coords.
    """
    obj = np.asarray(points_xyz_tracker, dtype=np.float64).reshape(-1,1,3)
    img_pts, _ = cv2.projectPoints(obj, rvec_CT, t_CT, CAM_MTX, DIST_COEFFS)
    return img_pts.reshape(-1,2)

def main():
    buf = PoseBuffer()
    threading.Thread(target=tracker_thread, args=(buf,), daemon=True).start()

    cap = (cv2.VideoCapture(gst_pipeline(RTSP_URL), cv2.CAP_GSTREAMER)
           if USE_GSTREAMER else cv2.VideoCapture(RTSP_URL))
    if not cap.isOpened():
        print("ERROR: Could not open RTSP stream."); return

    print("Press ESC or q to quit.")
    axis_len_mm = 40.0  # how long to draw axes in 3D (mm)
    last_txt = "MISSING"

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02); continue

        sample = buf.latest()
        drew = False
        if sample and sample.get("tracking"):
            qtx = to_qtx(sample["tracking"][0])
            if qtx is not None:
                qw,qx,qy,qz,x,y,z = qtx
                R_tool = quat_to_R(qw,qx,qy,qz)
                if R_tool is not None:
                    # Build 3D points in tracker coords: origin and axis endpoints
                    P0 = np.array([x,y,z], dtype=np.float64)
                    Px = P0 + axis_len_mm * R_tool[:,0]
                    Py = P0 + axis_len_mm * R_tool[:,1]
                    Pz = P0 + axis_len_mm * R_tool[:,2]

                    # Project to pixels
                    uv = project_points_tracker(np.vstack([P0,Px,Py,Pz]))
                    u0,v0 = uv[0]; ux,vx = uv[1]; uy,vy = uv[2]; uz,vz = uv[3]

                    # Draw axes at tool center in image
                    O = (int(round(u0)), int(round(v0)))
                    X = (int(round(ux)), int(round(vx)))
                    Y = (int(round(uy)), int(round(vy)))
                    Z = (int(round(uz)), int(round(vz)))
                    if all(np.isfinite(uv).ravel()):
                        cv2.circle(frame, O, 4, (255,255,255), -1)
                        cv2.line(frame, O, X, (0,0,255), 2)   # X red
                        cv2.line(frame, O, Y, (0,255,0), 2)   # Y green
                        cv2.line(frame, O, Z, (255,0,0), 2)   # Z blue
                        drew = True
                        last_txt = "LIVE"

                        # Optional HUD text
                        ts = datetime.fromtimestamp(sample["time"]).strftime("%H:%M:%S.%f")[:-3]
                        port = sample["ports"][0] if sample["ports"] else "?"
                        frm  = sample["frames"][0] if len(sample["frames"]) else "?"
                        qlt  = sample["quality"][0] if len(sample["quality"]) else "?"
                        cv2.putText(frame, f"t={ts} port={port} frame={frm}",
                                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"pos(mm): x={x:7.2f} y={y:7.2f} z={z:7.2f}",
                                    (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        if not drew:
            cv2.putText(frame, last_txt, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow("Vega RTSP + Tool Axes (projected)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
