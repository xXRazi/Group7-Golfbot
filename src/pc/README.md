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
- `vision_detection.py` optionally runs the trained YOLO model in `models/robotvision_v3_best.pt` on the prepared 640 x 480 arena frame.
- `calibrate_warp.py` lets you click the four arena corners and writes `warp_calibration.txt`, which `camera.py` loads automatically.

Vision detection is enabled by default on the PC side. Install `ultralytics` in the PC Python environment to use it. Set `GOLFBOT_USE_VISION=0` to force the old color detector, or set `GOLFBOT_VISION_MODEL_PATH=/path/to/best.pt` to test another model. The bundled default is the `RobotVision_V3` checkpoint. Color coordinate fallback is disabled by default; set `GOLFBOT_ALLOW_COLOR_FALLBACK=1` to let color detection fill in missing balls, robot pose, grappler/claw, or goal markers. Robot pose reads retry for a few frames by default; tune this with `GOLFBOT_ROBOT_POSE_RETRY_FRAMES` and `GOLFBOT_ROBOT_POSE_RETRY_DELAY_SECONDS`. Perspective warping is disabled by default; set `GOLFBOT_USE_WARP=1` to use `warp_calibration.txt`. The model now runs on that prepared arena frame, so the live model view shows the same image used for robot coordinates. The model live view is enabled by default; set `GOLFBOT_VISION_LIVE_VIEW=0` to hide it or `GOLFBOT_VISION_LIVE_VIEW_MAX_WIDTH=1280` to make the window larger.

After a successful delivery, the autonomous loop keeps searching for more balls by default. Set `GOLFBOT_STOP_AFTER_SUCCESSFUL_DELIVERY=1` to stop after one delivery. Pickup considers white and orange balls by default; set `GOLFBOT_PICKUP_BALL_COLORS=W`, `O`, or `W,O` to choose which colors it should collect.

When the model misses the claw or robot pose, the PC saves one prepared arena frame per stationary robot position in `images/missing_detections`. Set `GOLFBOT_SAVE_MISSING_DETECTION_FRAMES=0` to disable this, or `GOLFBOT_MISSING_DETECTION_DIR=/path/to/dir` to save somewhere else. The duplicate suppression resets after successful robot movement commands.

The PC vision map is 640 x 480 by default. On startup, the PC sends the configured EV3 coordinate bounds with a `MAPSIZE` command before any `POSSYNC` or `GOTO`. Keep `EV3_MAP_WIDTH` and `EV3_MAP_HEIGHT` equal to the PC map unless you intentionally want PC coordinates scaled before they are sent.

To inspect what the model sees without running the robot loop:

```bash
python vision_live_view.py
```

Delivery uses fixed map goal openings by default: Goal_B is halfway up the left edge, Goal_A is halfway up the right edge, each offset inward by `GOLFBOT_DELIVERY_FIXED_GOAL_SIDE_OFFSET_RATIO` (`0.05` by default). Set `GOLFBOT_DELIVERY_PREFER_VISION_GOALS=1` to prefer model-detected goal openings while still falling back to the fixed map positions, or set `GOLFBOT_DELIVERY_USE_FIXED_GOALS=0` to require detected goals. Delivery verifies a fresh vision frame before pushing the ball. The claw position is the important final check: `GOLFBOT_DELIVERY_CLAW_POSITION_TOLERANCE` defaults to `45`, while the robot center position is a soft check unless `GOLFBOT_DELIVERY_REQUIRE_CENTER_POSITION=1`. `GOLFBOT_DELIVERY_POSITION_TOLERANCE` defaults to `60`, and `GOLFBOT_DELIVERY_FINAL_CORRECTION_ATTEMPTS` defaults to `1`.

When the model sees a `redcross`, PC path planning marks the detected box plus `GOLFBOT_RED_CROSS_OBSTACLE_MARGIN` (`45` map units by default) as blocked. Delivery then follows A* waypoints around it instead of sending one straight `GOTO`; tune waypoint spacing with `GOLFBOT_DELIVERY_WAYPOINT_STEP_SIZE` (`70` by default), or set `GOLFBOT_AVOID_RED_CROSS=0` to disable this. Red-cross path waypoints do a verified turn before moving forward; tune that heading tolerance with `GOLFBOT_PATH_PRETURN_HEADING_TOLERANCE` (`7` degrees by default).

Camera warp calibration only matters when `GOLFBOT_USE_WARP=1`. Rerun it whenever the camera is moved:

```bash
python calibrate_warp.py
```

Click top-left, top-right, bottom-right, bottom-left on the raw camera view, then press Enter to save.

