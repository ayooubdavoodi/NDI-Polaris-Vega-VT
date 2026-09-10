# Polaris Vega VT

Tools for working with the NDI **Polaris Vega VT** optical tracking system: reading live tool poses, streaming the Vega's built-in camera over RTSP, and overlaying tracked-tool axes or a 3D STL mesh onto that video feed in real time. Several scripts are wired into ROS 2 for use in a larger robotics/teleoperation pipeline.

## Hardware / network

- NDI Polaris Vega VT unit, reachable at `169.254.7.143` (host machine expected at `169.254.7.240`).
- Tracker control port: `8765`.
- Onboard camera RTSP stream: `rtsp://169.254.7.143:554/video`.

See [shortcuts.sh](shortcuts.sh) for the ping/build/run command reference.

## Repository layout

| Path | Contents |
|---|---|
| [src/vega_track.py](src/vega_track.py) | Minimal example: connect to the tracker and print raw tool poses. |
| [src/Transformation_fun.py](src/Transformation_fun.py) | Shared math helpers (rotation matrices, Euler angles, quaternions, etc.). |
| [src/vega_rtsp_overlay.py](src/vega_rtsp_overlay.py) | RTSP video + tracked-tool XYZ axes overlay. |
| [src/vega_rtsp_combo.py](src/vega_rtsp_combo.py) | Earlier/alternate version of the axes overlay. |
| [src/vega_overlay_mesh.py](src/vega_overlay_mesh.py) | RTSP video + 3D STL mesh overlay, ROS 2 node. |
| [src/vega_rtsp_overlay_ros.py](src/vega_rtsp_overlay_ros.py) | Axes overlay as a ROS 2 node (publishes pose + image). |
| [src/vega_rtsp_overlay_ros_mesh.py](src/vega_rtsp_overlay_ros_mesh.py) | Mesh overlay as a ROS 2 node. |
| [src/vega_rtsp_overlay_ros_mesh_low_delay.py](src/vega_rtsp_overlay_ros_mesh_low_delay.py) | Low-latency variant of the ROS 2 mesh overlay. |
| [marker_definition/](marker_definition/) | Custom rigid-body marker definition ("Kia phantom marker"): `.rom`, `.stl`, calibration test results. |
| [phantom/](phantom/) | STL of the physical test phantom. |
| [setting/](setting/) | Camera calibration ([vega_calibration.json](setting/vega_calibration.json)) and a reference screenshot of the Vega's settings page. |
| [Combined_API_Sample_C++_v1.9.7/](Combined_API_Sample_C++_v1.9.7/) | NDI's official C++ sample app (CAPIsample / ARDemo) for direct bring-up/testing of the Vega. |
| [installation_files/](installation_files/) | Vendor installers (Cygna-6D, ToolBox) and sample archives from NDI. |
| [shortcuts.sh](shortcuts.sh) | Quick-reference commands for pinging, building, and running everything above. |

## Setup

```bash
pip install scikit-surgerynditracker opencv-python numpy trimesh
```

ROS 2 scripts additionally require a sourced ROS 2 install (`rclpy`, `sensor_msgs`, `std_msgs`, `cv_bridge`).

1. Confirm the Vega is reachable: `ping 169.254.7.143`
2. Check/update the tracker IP, port, and `.rom` marker path at the bottom of whichever script you run.
3. Check/update the `STL_PATH` (and the marker `.rom` path) if you've moved the repo off `/home/ayoob/Polaris_Vega_VT` — these are still absolute paths. `CALIB_JSON` is resolved automatically relative to the repo root, so it doesn't need editing.

## Running

All scripts live in [src/](src/) and can be run from anywhere (calibration/marker paths are resolved relative to the script location, not the current directory):

```bash
python3 src/vega_track.py              # raw pose printout, no video
python3 src/vega_rtsp_overlay.py       # RTSP video + axes overlay
python3 src/vega_overlay_mesh.py       # RTSP video + STL mesh overlay (ROS 2 node)
```

Press `q` or `Esc` in the video window to quit.

## Calibration

[setting/vega_calibration.json](setting/vega_calibration.json) holds the Vega camera's intrinsics, distortion coefficients, ROI/binning, and the `camera <- tracker` extrinsic (as a quaternion + translation in mm), copied from the device's ARDemo `GET` output. The overlay scripts use this to project 3D tracker-space points into the 2D camera image via `cv2.projectPoints`. If an overlay looks mirrored or flipped, try `invert_extrinsic=True` where exposed.

## C++ sample app

For low-level bring-up/testing against the Vega directly (bypassing Python):

```bash
cd Combined_API_Sample_C++_v1.9.7/CombinedAPIsample
make
./bin/linux/capisample 169.254.7.143 --tools="$(pwd)/sroms/Kia_phantom_marker.rom"
```
