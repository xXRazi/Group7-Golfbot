import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False


load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")


def _env_bool(name, default):
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() not in ("0", "false", "no", "off")


def _env_float(name, default):
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name, default):
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return default


CAMERA_INDEX = 0
MAP_WIDTH = 640
MAP_HEIGHT = 480
# EV3 odometry coordinate bounds. The PC sends these to the EV3 at startup.
# Keep them equal to MAP_WIDTH/MAP_HEIGHT unless you intentionally want the PC
# to scale camera-map coordinates into a different EV3 coordinate system.
EV3_MAP_WIDTH = _env_int("GOLFBOT_EV3_MAP_WIDTH", MAP_WIDTH)
EV3_MAP_HEIGHT = _env_int("GOLFBOT_EV3_MAP_HEIGHT", MAP_HEIGHT)

# Raw camera arena corners, ordered top-left, top-right, bottom-right, bottom-left.
DEFAULT_PERSPECTIVE_SOURCE_POINTS = (
    (1, 0),
    (1916, 1),
    (1919, 1076),
    (1, 1078),
)
WARP_CALIBRATION_PATH = os.getenv(
    "GOLFBOT_WARP_CALIBRATION_PATH",
    os.path.join(BASE_DIR, "warp_calibration.txt"),
)
CAMERA_WARP_ENABLED = _env_bool("GOLFBOT_USE_WARP", True)


def _load_perspective_source_points(path, default_points):
    try:
        with open(path, "r") as calibration_file:
            lines = calibration_file.readlines()
    except OSError:
        return default_points, "default"

    points = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        parts = stripped.replace(",", " ").split()

        if len(parts) != 2:
            return default_points, "invalid"

        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            return default_points, "invalid"

        points.append((x, y))

    if len(points) != 4:
        return default_points, "invalid"

    return tuple(points), path


PERSPECTIVE_SOURCE_POINTS, PERSPECTIVE_SOURCE_POINTS_SOURCE = _load_perspective_source_points(
    WARP_CALIBRATION_PATH,
    DEFAULT_PERSPECTIVE_SOURCE_POINTS,
)

SYNC_DELAY_SECONDS = 0.2
SYNC_IMAGE_PATH = os.path.join(IMAGE_DIR, "robot_sync_frame.png")

FRAME_CAPTURE_INTERVAL_SECONDS = 1
STARTUP_DELAY_SECONDS = 2

# YOLO vision detector. The model runs on the prepared 640 x 480 arena frame.
# With GOLFBOT_USE_WARP=1 that frame is perspective-warped; otherwise it is a
# resized raw camera frame.
VISION_DETECTION_ENABLED = _env_bool("GOLFBOT_USE_VISION", True)
VISION_MODEL_PATH = os.getenv(
    "GOLFBOT_VISION_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "robotvision_v3_best.pt"),
)
VISION_CONFIDENCE = _env_float("GOLFBOT_VISION_CONFIDENCE", 0.45)
VISION_IOU = _env_float("GOLFBOT_VISION_IOU", 0.7)
VISION_MODEL_IMAGE_SIZE = _env_int("GOLFBOT_VISION_IMAGE_SIZE", 640)
VISION_DEBUG = _env_bool("GOLFBOT_VISION_DEBUG", False)
VISION_LIVE_VIEW_ENABLED = _env_bool("GOLFBOT_VISION_LIVE_VIEW", True)
VISION_LIVE_VIEW_MAX_WIDTH = _env_int("GOLFBOT_VISION_LIVE_VIEW_MAX_WIDTH", 960)
VISION_LIVE_VIEW_WINDOW_NAME = os.getenv("GOLFBOT_VISION_LIVE_VIEW_WINDOW", "Golfbot model view")
VISION_MISSING_DETECTION_SAVE_ENABLED = _env_bool("GOLFBOT_SAVE_MISSING_DETECTION_FRAMES", True)
VISION_MISSING_DETECTION_DIR = os.getenv(
    "GOLFBOT_MISSING_DETECTION_DIR",
    os.path.join(IMAGE_DIR, "missing_detections"),
)

# Red cross avoidance. The model detects the center cross. Path planning stamps
# that detection as two padded arms instead of one full bounding-box square.
RED_CROSS_AVOIDANCE_ENABLED = _env_bool("GOLFBOT_AVOID_RED_CROSS", True)
RED_CROSS_OBSTACLE_MARGIN = _env_int("GOLFBOT_RED_CROSS_OBSTACLE_MARGIN", 30)
RED_CROSS_OBSTACLE_ARM_RATIO = _env_float("GOLFBOT_RED_CROSS_OBSTACLE_ARM_RATIO", 0.25)
RED_CROSS_OBSTACLE_MIN_ARM_WIDTH = _env_int("GOLFBOT_RED_CROSS_OBSTACLE_MIN_ARM_WIDTH", 12)
PATH_PRETURN_HEADING_TOLERANCE = _env_float("GOLFBOT_PATH_PRETURN_HEADING_TOLERANCE", 7.0)

# Keep coordinate decisions vision-only by default. Set this to True or export
# GOLFBOT_ALLOW_COLOR_FALLBACK=1 to let old color detection fill in missing
# balls, robot pose, grappler/claw, or goal markers.
ALLOW_COLOR_DETECTION_FALLBACK = _env_bool("GOLFBOT_ALLOW_COLOR_FALLBACK", False)
ROBOT_POSE_RETRY_FRAMES = _env_int("GOLFBOT_ROBOT_POSE_RETRY_FRAMES", 6)
ROBOT_POSE_RETRY_DELAY_SECONDS = _env_float("GOLFBOT_ROBOT_POSE_RETRY_DELAY_SECONDS", 0.12)
ROBOT_POSE_HOLD_FRAMES = _env_int("GOLFBOT_ROBOT_POSE_HOLD_FRAMES", 3)
PICKUP_BALL_COLORS = tuple(
    color.strip().upper()
    for color in os.getenv("GOLFBOT_PICKUP_BALL_COLORS", "W,O").split(",")
    if color.strip()
)

# Coarse pickup path tuning.
PICKUP_STOP_DISTANCE = 10
PICKUP_WAYPOINT_STEP_SIZE = 50
PICKUP_SETTLE_SECONDS = 0.15
PICKUP_PREAPPROACH_DISTANCE = 55.0
PICKUP_BALL_ENDPOINT_CLEAR_RADIUS = _env_int("GOLFBOT_PICKUP_BALL_ENDPOINT_CLEAR_RADIUS", 30)
PICKUP_RED_CROSS_CLEARANCE_MARGIN = _env_int("GOLFBOT_PICKUP_RED_CROSS_CLEARANCE_MARGIN", 40)
USE_COARSE_PICKUP_PREAPPROACH = False

# Grappler marker fallback offset from robot center.
GRAPPLER_FORWARD_OFFSET_FALLBACK = 40.0
GRAPPLER_LATERAL_OFFSET_FALLBACK = 0.0

# Final pickup camera-servo tuning.
PICKUP_SERVO_MAX_ITERATIONS = 24
PICKUP_SERVO_MAX_FORWARD_STEP = 32.0
PICKUP_SERVO_FAR_FORWARD_STEP = 32.0
PICKUP_SERVO_MID_FORWARD_STEP = 22.0
PICKUP_SERVO_NEAR_FORWARD_STEP = 12.0
PICKUP_SERVO_MIN_FORWARD_STEP = 3.0
PICKUP_CENTER_TO_BALL_CLOSE_DISTANCE = 36.0
PICKUP_CENTER_TO_BALL_MARGIN = 3.0
PICKUP_FINAL_HEADING_CLOSE_TOLERANCE = 14.0
PICKUP_DIRECT_FALLBACK_MAX_DISTANCE = 90.0
PICKUP_FINAL_SCOOP_DISTANCE = 7.0
PICKUP_GRAPPLER_CLOSE_DISTANCE = 23.0
PICKUP_FINAL_HEADING_TOLERANCE = 5.0
PICKUP_FINAL_SYNC_DELAY_SECONDS = 0.15
PICKUP_FINAL_NUDGE_MAX_DISTANCE = 28.0
PICKUP_FINAL_NUDGE_MARGIN = 5.0
# Inflate apparent pickup distances near the image edges, where perspective can
# make the ball look closer to the claw than it physically is.
PICKUP_OFFCENTER_DISTANCE_SCALE = _env_float("GOLFBOT_PICKUP_OFFCENTER_DISTANCE_SCALE", 0.55)
PICKUP_OFFCENTER_SCOOP_SCALE_LIMIT = _env_float("GOLFBOT_PICKUP_OFFCENTER_SCOOP_SCALE_LIMIT", 1.45)

# Delivery settings.
DELIVERY_GOAL_PREFERENCE = "nearest"
DELIVERY_USE_FIXED_GOALS = _env_bool("GOLFBOT_DELIVERY_USE_FIXED_GOALS", True)
DELIVERY_PREFER_VISION_GOALS = _env_bool("GOLFBOT_DELIVERY_PREFER_VISION_GOALS", False)
DELIVERY_FIXED_GOAL_ROW_RATIO = _env_float("GOLFBOT_DELIVERY_FIXED_GOAL_ROW_RATIO", 0.5)
DELIVERY_FIXED_GOAL_SIDE_OFFSET_RATIO = _env_float(
    "GOLFBOT_DELIVERY_FIXED_GOAL_SIDE_OFFSET_RATIO",
    0.05,
)
DELIVERY_CENTER_TO_MARKER_DISTANCE = 105.0
DELIVERY_CLAW_TO_MARKER_DISTANCE = _env_float("GOLFBOT_DELIVERY_CLAW_TO_MARKER_DISTANCE", 25.0)
DELIVERY_CENTER_TO_CLAW_DISTANCE = _env_float("GOLFBOT_DELIVERY_CENTER_TO_CLAW_DISTANCE", 80.0)
DELIVERY_ROBOT_EDGE_MARGIN = 65
DELIVERY_CLAW_EDGE_MARGIN = _env_int("GOLFBOT_DELIVERY_CLAW_EDGE_MARGIN", 45)
DELIVERY_PREAPPROACH_EXTRA_DISTANCE = 45.0
DELIVERY_POSITION_TOLERANCE = _env_float("GOLFBOT_DELIVERY_POSITION_TOLERANCE", 60.0)
DELIVERY_REQUIRE_CENTER_POSITION = _env_bool("GOLFBOT_DELIVERY_REQUIRE_CENTER_POSITION", False)
DELIVERY_CLAW_POSITION_TOLERANCE = _env_float("GOLFBOT_DELIVERY_CLAW_POSITION_TOLERANCE", 45.0)
DELIVERY_HEADING_TOLERANCE = 7.0
DELIVERY_FINAL_CORRECTION_ATTEMPTS = _env_int("GOLFBOT_DELIVERY_FINAL_CORRECTION_ATTEMPTS", 1)
DELIVERY_EDGE_ESCAPE_REVERSE_SPEED = _env_int("GOLFBOT_DELIVERY_EDGE_ESCAPE_REVERSE_SPEED", -20)
DELIVERY_EDGE_ESCAPE_SECONDS = _env_float("GOLFBOT_DELIVERY_EDGE_ESCAPE_SECONDS", 0.45)
DELIVERY_WAYPOINT_STEP_SIZE = _env_int("GOLFBOT_DELIVERY_WAYPOINT_STEP_SIZE", 70)
HELD_CLAW_RECLOSE_ENABLED = _env_bool("GOLFBOT_RECLOSE_OPEN_CLAW_WHEN_HELD", True)
HELD_CLAW_RECLOSE_DELAY_SECONDS = _env_float("GOLFBOT_HELD_CLAW_RECLOSE_DELAY_SECONDS", 0.25)
DELIVERY_GOAL_A_MARKER_FALLBACK = (
    int(round((MAP_HEIGHT - 1) * DELIVERY_FIXED_GOAL_ROW_RATIO)),
    int(round((MAP_WIDTH - 1) * (1.0 - DELIVERY_FIXED_GOAL_SIDE_OFFSET_RATIO))),
)
DELIVERY_GOAL_B_MARKER_FALLBACK = (
    int(round((MAP_HEIGHT - 1) * DELIVERY_FIXED_GOAL_ROW_RATIO)),
    int(round((MAP_WIDTH - 1) * DELIVERY_FIXED_GOAL_SIDE_OFFSET_RATIO)),
)
STOP_AFTER_SUCCESSFUL_DELIVERY = _env_bool("GOLFBOT_STOP_AFTER_SUCCESSFUL_DELIVERY", False)
