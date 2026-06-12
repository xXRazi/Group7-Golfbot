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
CAMERA_WARP_ENABLED = _env_bool("GOLFBOT_USE_WARP", False)


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

# YOLO vision detector. The model runs on the raw camera frame and its centers
# are mapped into the shared 640 x 480 coordinate system.
VISION_DETECTION_ENABLED = _env_bool("GOLFBOT_USE_VISION", True)
VISION_MODEL_PATH = os.getenv(
    "GOLFBOT_VISION_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "robotvision_v2_best.pt"),
)
VISION_CONFIDENCE = _env_float("GOLFBOT_VISION_CONFIDENCE", 0.45)
VISION_IOU = _env_float("GOLFBOT_VISION_IOU", 0.7)
VISION_MODEL_IMAGE_SIZE = _env_int("GOLFBOT_VISION_IMAGE_SIZE", 640)
VISION_DEBUG = _env_bool("GOLFBOT_VISION_DEBUG", False)
VISION_LIVE_VIEW_ENABLED = _env_bool("GOLFBOT_VISION_LIVE_VIEW", True)
VISION_LIVE_VIEW_MAX_WIDTH = _env_int("GOLFBOT_VISION_LIVE_VIEW_MAX_WIDTH", 960)
VISION_LIVE_VIEW_WINDOW_NAME = os.getenv("GOLFBOT_VISION_LIVE_VIEW_WINDOW", "Golfbot model view")

# Keep coordinate decisions vision-only by default. Set this to True or export
# GOLFBOT_ALLOW_COLOR_FALLBACK=1 to let old color detection fill in missing
# balls, robot pose, grappler/claw, or goal markers.
ALLOW_COLOR_DETECTION_FALLBACK = _env_bool("GOLFBOT_ALLOW_COLOR_FALLBACK", False)
ROBOT_POSE_RETRY_FRAMES = _env_int("GOLFBOT_ROBOT_POSE_RETRY_FRAMES", 6)
ROBOT_POSE_RETRY_DELAY_SECONDS = _env_float("GOLFBOT_ROBOT_POSE_RETRY_DELAY_SECONDS", 0.12)

# Coarse pickup path tuning.
PICKUP_STOP_DISTANCE = 10
PICKUP_WAYPOINT_STEP_SIZE = 40
PICKUP_SETTLE_SECONDS = 0.15
PICKUP_PREAPPROACH_DISTANCE = 55.0
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
PICKUP_FINAL_SCOOP_DISTANCE = 13.0
PICKUP_GRAPPLER_CLOSE_DISTANCE = 23.0
PICKUP_FINAL_HEADING_TOLERANCE = 5.0
PICKUP_FINAL_SYNC_DELAY_SECONDS = 0.15
PICKUP_FINAL_NUDGE_MAX_DISTANCE = 28.0
PICKUP_FINAL_NUDGE_MARGIN = 5.0

# Delivery settings.
DELIVERY_GOAL_PREFERENCE = "nearest"
DELIVERY_CENTER_TO_MARKER_DISTANCE = 42.0
DELIVERY_PREAPPROACH_EXTRA_DISTANCE = 45.0
DELIVERY_POSITION_TOLERANCE = 9.0
DELIVERY_HEADING_TOLERANCE = 7.0
DELIVERY_GOAL_A_MARKER_FALLBACK = (MAP_HEIGHT // 2, 505)
DELIVERY_GOAL_B_MARKER_FALLBACK = (MAP_HEIGHT // 2, 184)
STOP_AFTER_SUCCESSFUL_DELIVERY = True
