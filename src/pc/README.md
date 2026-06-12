The PC folder contains code which runs on the PC and sends commands to the EV3 robot.

Main entrypoints:
- `main.py` runs the autonomous camera/pickup/delivery loop.
- `robot_console.py` runs the manual EV3 control console.
- `test_camera.py` probes available camera indexes.
- `vision_live_view.py` shows the YOLO model detections without connecting to the EV3.

Core modules:
- `camera.py` opens the camera, prepares map-sized frames, saves frames, and builds color matrices.
- `robot_sync.py` syncs camera-detected robot poses to the EV3 and sends movement helpers.
- `pickup.py` handles ball pickup planning and final camera-servo pickup.
- `delivery.py` handles goal marker selection and delivery.
- `map_utils.py` contains shared map/coordinate/path helpers.
- `settings.py` contains shared constants and tuning values.
- `com_protocol.py` builds and sends EV3 protocol packets.
- `Imagesplitter.py` converts images to color-label matrices.
- `id_color.py` detects balls, robot markers, grappler, and goals in color matrices.
- `vision_detection.py` optionally runs the trained YOLO model in `models/robotvision_v2_best.pt` and maps detections into the same 640 x 480 arena coordinates.
- `calibrate_warp.py` lets you click the four arena corners and writes `warp_calibration.txt`, which `camera.py` loads automatically.

Vision detection is enabled by default on the PC side. Install `ultralytics` in the PC Python environment to use it. Set `GOLFBOT_USE_VISION=0` to force the old color detector, or set `GOLFBOT_VISION_MODEL_PATH=/path/to/best.pt` to test another model. The bundled default is the `RobotVision_V2` checkpoint. Color coordinate fallback is disabled by default; set `GOLFBOT_ALLOW_COLOR_FALLBACK=1` to let color detection fill in missing balls, robot pose, grappler/claw, or goal markers. Robot pose reads retry for a few frames by default; tune this with `GOLFBOT_ROBOT_POSE_RETRY_FRAMES` and `GOLFBOT_ROBOT_POSE_RETRY_DELAY_SECONDS`. Perspective warping is disabled by default; set `GOLFBOT_USE_WARP=1` to use `warp_calibration.txt`. The model live view is enabled by default; set `GOLFBOT_VISION_LIVE_VIEW=0` to hide it or `GOLFBOT_VISION_LIVE_VIEW_MAX_WIDTH=1280` to make the window larger.

To inspect what the model sees without running the robot loop:

```bash
python vision_live_view.py
```

Camera warp calibration only matters when `GOLFBOT_USE_WARP=1`. Rerun it whenever the camera is moved:

```bash
python calibrate_warp.py
```

Click top-left, top-right, bottom-right, bottom-left on the raw camera view, then press Enter to save.

