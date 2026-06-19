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
- `vision_detection.py` optionally runs the trained YOLO model in `models/robotvision_v4_best.pt` on the prepared 640 x 480 arena frame.
- `calibrate_warp.py` lets you click the four arena corners and writes `warp_calibration.txt`, which `camera.py` loads automatically.
- `calibrate_motion.py` runs an automated camera-guided turn/drive calibration and saves `motion_calibration.txt` for the EV3.

Vision detection is enabled by default on the PC side. Install `ultralytics` in the PC Python environment to use it. Set `GOLFBOT_USE_VISION=0` to force the old color detector, or set `GOLFBOT_VISION_MODEL_PATH=/path/to/best.pt` to test another model. The bundled default is the `RobotVision_V4` checkpoint. Color coordinate fallback is disabled by default; set `GOLFBOT_ALLOW_COLOR_FALLBACK=1` to let color detection fill in missing balls, robot pose, grappler/claw, or goal markers. Robot pose reads retry for a few frames by default; tune this with `GOLFBOT_ROBOT_POSE_RETRY_FRAMES` and `GOLFBOT_ROBOT_POSE_RETRY_DELAY_SECONDS`. Perspective warping is disabled by default; set `GOLFBOT_USE_WARP=1` to use `warp_calibration.txt`. The model now runs on that prepared arena frame, so the live model view shows the same image used for robot coordinates. The model live view is enabled by default and scales the preview to `GOLFBOT_VISION_LIVE_VIEW_MAX_WIDTH` (`960` by default); set `GOLFBOT_VISION_LIVE_VIEW=0` to hide it or `GOLFBOT_VISION_LIVE_VIEW_MAX_WIDTH=1280` to make the window larger.

After a successful delivery, the autonomous loop keeps searching for more balls by default. Set `GOLFBOT_STOP_AFTER_SUCCESSFUL_DELIVERY=1` to stop after one delivery. Pickup considers white and orange balls by default; set `GOLFBOT_PICKUP_BALL_COLORS=W`, `O`, or `W,O` to choose which colors it should collect. Orange balls are prioritized by default with `GOLFBOT_PICKUP_BALL_COLOR_PRIORITY=O,W`, and delivered to Goal_A, the right/small goal, by default with `GOLFBOT_DELIVERY_ORANGE_BALL_GOAL=A`.

When the model misses the claw or robot pose, the PC saves one prepared arena frame per stationary robot position in `images/missing_detections`. Set `GOLFBOT_SAVE_MISSING_DETECTION_FRAMES=0` to disable this, or `GOLFBOT_MISSING_DETECTION_DIR=/path/to/dir` to save somewhere else. The duplicate suppression resets after successful robot movement commands. If the robot body is visible but the claw/grappler is missing, the PC sends a short reverse pulse before trying again; disable it with `GOLFBOT_MISSING_GRAPPLER_REVERSE=0`, or tune it with `GOLFBOT_MISSING_GRAPPLER_REVERSE_SPEED` and `GOLFBOT_MISSING_GRAPPLER_REVERSE_SECONDS`.

The PC vision map is 640 x 480 by default. On startup, the PC sends the configured EV3 coordinate bounds with a `MAPSIZE` command before any `POSSYNC` or `GOTO`. Keep `EV3_MAP_WIDTH` and `EV3_MAP_HEIGHT` equal to the PC map unless you intentionally want PC coordinates scaled before they are sent.

To inspect what the model sees without running the robot loop:

```bash
python vision_live_view.py
```

Delivery uses fixed map goal openings by default: Goal_B is halfway up the left edge, Goal_A is halfway up the right edge, each offset inward by `GOLFBOT_DELIVERY_FIXED_GOAL_SIDE_OFFSET_RATIO` (`0.05` by default). Set `GOLFBOT_DELIVERY_PREFER_VISION_GOALS=1` to prefer model-detected goal openings while still falling back to the fixed map positions, or set `GOLFBOT_DELIVERY_USE_FIXED_GOALS=0` to require detected goals. While carrying a ball, delivery sends `CLAW_CLOSE` again if the model sees `open_claw`; disable that with `GOLFBOT_RECLOSE_OPEN_CLAW_WHEN_HELD=0`, or tune the post-close wait with `GOLFBOT_HELD_CLAW_RECLOSE_DELAY_SECONDS` (`0.25` by default). Delivery verifies a fresh vision frame before pushing the ball. The claw position is the important final check: `GOLFBOT_DELIVERY_CLAW_POSITION_TOLERANCE` defaults to `45`, while the robot center position is a soft check unless `GOLFBOT_DELIVERY_REQUIRE_CENTER_POSITION=1`. `GOLFBOT_DELIVERY_POSITION_TOLERANCE` defaults to `60`, and `GOLFBOT_DELIVERY_FINAL_CORRECTION_ATTEMPTS` defaults to `1`. Delivery also refuses to push when it is too close to the side goal wall; tune this with `GOLFBOT_DELIVERY_MIN_CENTER_GOAL_DISTANCE` (`130`), `GOLFBOT_DELIVERY_MIN_CLAW_GOAL_DISTANCE` (`70`), and `GOLFBOT_DELIVERY_GOAL_DISTANCE_CORRECTION_MARGIN` (`8`). When it is too close and already facing the goal, it reverses instead of doing a full `GOTO`; disable this with `GOLFBOT_DELIVERY_GOAL_DISTANCE_REVERSE=0`, or tune `GOLFBOT_DELIVERY_GOAL_DISTANCE_REVERSE_SPEED`, `GOLFBOT_DELIVERY_GOAL_DISTANCE_REVERSE_SECONDS_PER_MAP_UNIT`, and `GOLFBOT_DELIVERY_GOAL_DISTANCE_CORRECTION_ATTEMPTS`.

When the model sees a `redcross`, PC path planning marks a plus-shaped obstacle from the detected box: one vertical arm and one horizontal arm, padded by `GOLFBOT_RED_CROSS_OBSTACLE_MARGIN` (`30` map units by default). Tune the arm thickness with `GOLFBOT_RED_CROSS_OBSTACLE_ARM_RATIO` (`0.25` by default) and `GOLFBOT_RED_CROSS_OBSTACLE_MIN_ARM_WIDTH` (`12` by default), or set `GOLFBOT_AVOID_RED_CROSS=0` to disable this. Pickup uses a larger grappler safety margin, `GOLFBOT_PICKUP_RED_CROSS_CLEARANCE_MARGIN` (`40` by default), and skips balls inside that protected area. The final camera-servo scoop also refuses to move or close the claw if the only visible ball is inside that protected area or if the scoop path would cross it. Pickup clears a soft approach pocket around safe selected balls with `GOLFBOT_PICKUP_BALL_ENDPOINT_CLEAR_RADIUS` (`30` by default), without clearing cells that were originally red cross pixels or inside the grappler safety area. Red-cross A* routes use a lookahead point instead of chasing every exact grid corner; tune this with `GOLFBOT_RED_CROSS_WAYPOINT_ACCEPTANCE_RADIUS`, `GOLFBOT_PICKUP_RED_CROSS_LOOKAHEAD_DISTANCE`, and `GOLFBOT_DELIVERY_RED_CROSS_LOOKAHEAD_DISTANCE`. Delivery then follows A* waypoints around the obstacle only when needed; tune waypoint spacing with `GOLFBOT_DELIVERY_WAYPOINT_STEP_SIZE` (`70` by default). Red-cross path waypoints do a verified turn before moving forward; tune that heading tolerance with `GOLFBOT_PATH_PRETURN_HEADING_TOLERANCE` (`7` degrees by default).

Camera warp calibration only matters when `GOLFBOT_USE_WARP=1`. Rerun it whenever the camera is moved:

```bash
python calibrate_warp.py
```

Click top-left, top-right, bottom-right, bottom-left on the raw camera view, then press Enter to save.

Motion calibration can be rerun whenever the robot starts under/over-turning or driving the wrong distance:

```bash
python calibrate_motion.py
```

Keep the robot clear of balls, walls, and the red cross while it runs. The script updates both the EV3 and the local `src/ev3/motion_calibration.txt` file.

