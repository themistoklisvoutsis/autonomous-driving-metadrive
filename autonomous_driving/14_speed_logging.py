import cv2
import math
import statistics
import csv
from pathlib import Path

from ultralytics import YOLO

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.obs.state_obs import LidarStateObservation


# ==========================================================
# GENERAL SETTINGS
# ==========================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CAMERA_FOV = 66

CONFIDENCE_THRESHOLD = 0.50

# COCO classes:
# 0 = person
# 1 = bicycle
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
RELEVANT_CLASSES = [0, 1, 2, 3, 5, 7]

PRINT_EVERY = 20

# ==========================================================
# SPEED / SENSOR LOGGING
# ==========================================================

# Καταγράφουμε μόνο μετρήσεις. Δεν κάνουμε ακόμη risk labels.
LOG_FILE = Path(__file__).resolve().with_name("speed_measurements.csv")
#καταγραφη τρακαρισματων/εκτος δρομου οδηγηση κλπ
EVENT_LOG_FILE=Path(__file__).resolve().with_name("episode_events.csv")


LOG_COLUMNS = [
    "episode_id",
    "simulation_time_s",
    "track_id",
    "class",
    "ego_speed_kmh",
    "lidar_distance_m",
    "closing_speed_kmh",
    "camera_angle_deg",
    "lidar_angle_deg",
    "yolo_confidence",
    "lidar_confidence",
    "lidar_match_status",
]


def initialize_log_file():
    """Create CSV with header if it does not already exist."""
    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()


def log_speed_measurements(camera_tracks, ego_speed_kmh, current_time,episode_id):
    """
    Append one row per visible camera track that currently has
    a valid LiDAR distance measurement.

    closing_speed_kmh > 0  -> ego is closing the gap
    closing_speed_kmh < 0  -> gap is increasing

    IMPORTANT: closing speed is relative speed, not the absolute
    speed of the target vehicle.
    """
    rows = []

    for obj in camera_tracks:
        if obj["lidar_distance"] is None:
            continue

        rows.append(
            {
                "episode_id":episode_id,
                "simulation_time_s": round(current_time, 3),
                "track_id": obj["track_id"],
                "class": obj["class"],
                "ego_speed_kmh": round(float(ego_speed_kmh), 3),
                "lidar_distance_m": round(float(obj["lidar_distance"]), 3),
                "closing_speed_kmh": round(float(obj["closing_speed_kmh"]), 3),
                "camera_angle_deg": round(float(obj["camera_angle"]), 3),
                "lidar_angle_deg": round(float(obj["lidar_angle"]), 3),
                "yolo_confidence": round(float(obj["confidence"]), 4),
                "lidar_confidence": obj["lidar_confidence"],
                "lidar_match_status": obj["lidar_match_status"],
            }
        )

    if not rows:
        return

    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerows(rows)


initialize_log_file()


# ==========================================================
# LIDAR SETTINGS
# ==========================================================

NUM_LASERS = 72
LIDAR_DISTANCE = 40
ANGLE_PER_RAY = 360 / NUM_LASERS

# Μας ενδιαφέρει μόνο η περιοχή που βλέπει περίπου η camera.
FUSION_HALF_ANGLE = 33

MAX_DISTANCE_GAP = 3.0
MAX_ANGLE_GAP = ANGLE_PER_RAY * 1.5

# Αν το LiDAR cluster πέφτει λίγο έξω από το angular range του
# YOLO bbox, επιτρέπουμε μικρό περιθώριο.
BBOX_ANGLE_MARGIN = 3.0


# ==========================================================
# CAMERA-FIRST TRACKING SETTINGS
# ==========================================================

# Για πόσα frames ο BoT-SORT κρατά lost camera tracks alive.
BOT_SORT_TRACK_BUFFER = 45

# Μεγαλύτερο appearance threshold = αυστηρότερη ReID ομοιότητα.
BOT_SORT_APPEARANCE_THRESHOLD = 0.85

# Distance history για closing speed.
DISTANCE_HISTORY_LENGTH = 8

# Κρατάμε ξεχωριστό ιστορικό instantaneous closing-speed estimates
# και χρησιμοποιούμε median για να μην επηρεάζει το αποτέλεσμα ένα spike.
SPEED_HISTORY_LENGTH = 5

# Αν μεσολαβήσει μεγαλύτερο κενό LiDAR, δεν συγκρίνουμε τις δύο
# αποστάσεις για speed γιατί το dt είναι πλέον πολύ μεγάλο/αβέβαιο.
MAX_LIDAR_GAP_FOR_SPEED = 0.5

# Prediction-based LiDAR association. Δεν είναι safety threshold.
MAX_PREDICTION_GAP = 0.8
MAX_PREDICTION_ERROR_M = 4.0

# Sanity check μόνο για να κόβουμε προφανή spikes από λάθος association.
MAX_ABS_CLOSING_SPEED_KMH = 150.0

# Καθαρίζουμε sensor history μόνο αν το camera ID έχει χαθεί
# για αρκετό simulated χρόνο.
SENSOR_STATE_MAX_AGE = 5.0

# MetaDrive ~0.1 simulated sec ανά env.step() στο setup μας.
SIMULATION_DT = 0.1


# ==========================================================
# CREATE CUSTOM BOT-SORT CONFIG
# ==========================================================

TRACKER_CONFIG_PATH = Path(__file__).resolve().with_name(
    "botsort_metadrive_reid.yaml"
)


def ensure_tracker_config():
    """
    Φτιάχνει custom BoT-SORT config δίπλα στο script.

    Camera-first identity:
      - BoT-SORT κρατά τα IDs
      - sparseOptFlow κάνει camera-motion compensation
      - ReID βοηθά όταν δύο οχήματα είναι κοντά/διασταυρώνονται
    """

    config_text = f"""tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.25
track_buffer: {BOT_SORT_TRACK_BUFFER}
match_thresh: 0.80
fuse_score: True

gmc_method: sparseOptFlow

proximity_thresh: 0.50
appearance_thresh: {BOT_SORT_APPEARANCE_THRESHOLD}
with_reid: True
model: auto
"""

    # Γράφουμε πάντα το config ώστε οι παραπάνω παράμετροι να
    # αντιστοιχούν στην τρέχουσα έκδοση του script.
    TRACKER_CONFIG_PATH.write_text(config_text, encoding="utf-8")


ensure_tracker_config()


# ==========================================================
# LOAD YOLO
# ==========================================================

model = YOLO("yolo26n.pt")


# ==========================================================
# SENSOR STATE PER CAMERA TRACK ID
# ==========================================================

# ΣΗΜΑΝΤΙΚΟ:
# Τα IDs ΔΕΝ δημιουργούνται εδώ.
# Τα IDs προέρχονται αποκλειστικά από BoT-SORT/RGB camera.
# Εδώ αποθηκεύουμε μόνο LiDAR measurements ανά camera ID.
sensor_states = {}


# ==========================================================
# CAMERA FUNCTIONS
# ==========================================================


def pixel_x_to_angle(pixel_x):
    """
    Horizontal pixel -> camera-relative angle.

    negative = αριστερά
    0        = κέντρο
    positive = δεξιά
    """

    center_x = CAMERA_WIDTH / 2

    focal_length = (CAMERA_WIDTH / 2) / math.tan(
        math.radians(CAMERA_FOV / 2)
    )

    return math.degrees(
        math.atan((pixel_x - center_x) / focal_length)
    )



def get_direction(angle):
    if angle < -5:
        return "FRONT-LEFT"
    if angle > 5:
        return "FRONT-RIGHT"
    return "FRONT-CENTER"


# ==========================================================
# LIDAR CLUSTERING
# ==========================================================


def get_lidar_clusters(env, ego):
    """
    Raw LiDAR rays -> clusters.

    Δεν δημιουργεί IDs.
    Δεν αποφασίζει identity.
    Δίνει μόνο γεωμετρικά measurements.
    """

    lidar = env.engine.get_sensor("lidar")

    cloud_points, _ = lidar.perceive(
        ego,
        physics_world=env.engine.physics_world.dynamic_world,
        num_lasers=NUM_LASERS,
        distance=LIDAR_DISTANCE,
        show=True,
    )

    hits = []

    for ray_index, normalized_distance in enumerate(cloud_points):
        angle = ray_index * ANGLE_PER_RAY

        if angle > 180:
            signed_angle = angle - 360
        else:
            signed_angle = angle

        if abs(signed_angle) > FUSION_HALF_ANGLE:
            continue

        # ~1.0 => no hit
        if normalized_distance >= 0.999:
            continue

        hits.append(
            {
                "ray": ray_index,
                "angle": signed_angle,
                "distance": normalized_distance * LIDAR_DISTANCE,
            }
        )

    hits.sort(key=lambda h: h["angle"])

    if not hits:
        return []

    clusters = []
    current_cluster = [hits[0]]

    for hit in hits[1:]:
        previous_hit = current_cluster[-1]

        angle_gap = hit["angle"] - previous_hit["angle"]
        distance_gap = abs(hit["distance"] - previous_hit["distance"])

        if (
            angle_gap <= MAX_ANGLE_GAP
            and distance_gap <= MAX_DISTANCE_GAP
        ):
            current_cluster.append(hit)
        else:
            clusters.append(current_cluster)
            current_cluster = [hit]

    clusters.append(current_cluster)

    detected_objects = []

    for cluster in clusters:
        distances = [hit["distance"] for hit in cluster]
        angles = [hit["angle"] for hit in cluster]
        rays = [hit["ray"] for hit in cluster]

        center_angle = statistics.mean(angles)

        detected_objects.append(
            {
                "rays": rays,
                "distance": statistics.median(distances),
                "closest_distance": min(distances),
                "angle": center_angle,
                "direction": get_direction(center_angle),
                "lidar_confidence": (
                    "STRONG" if len(cluster) >= 2 else "WEAK"
                ),
            }
        )

    detected_objects.sort(key=lambda obj: obj["distance"])
    return detected_objects


# ==========================================================
# CAMERA TRACKS FROM BOT-SORT
# ==========================================================


def get_camera_tracks(frame_bgr):
    """
    RGB CAMERA -> YOLO -> BoT-SORT.

    Εδώ δημιουργείται και διατηρείται το Track ID.
    Το LiDAR δεν συμμετέχει καθόλου σε αυτή την απόφαση.
    """

    results = model.track(
        frame_bgr,
        persist=True,
        tracker=str(TRACKER_CONFIG_PATH),
        conf=CONFIDENCE_THRESHOLD,
        imgsz=640,
        classes=RELEVANT_CLASSES,
        verbose=False,
    )

    result = results[0]
    camera_tracks = []

    if result.boxes is None:
        return result, camera_tracks

    for box in result.boxes:
        # Unconfirmed detections μπορεί προσωρινά να μην έχουν ID.
        if box.id is None:
            continue

        track_id = int(box.id.item())
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        class_name = result.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

        center_x = (x1 + x2) / 2
        center_angle = pixel_x_to_angle(center_x)

        # Αντί να κάνουμε Camera↔LiDAR fusion μόνο από το κέντρο,
        # μετατρέπουμε ΟΛΟ το horizontal bbox σε angular interval.
        left_angle = pixel_x_to_angle(x1)
        right_angle = pixel_x_to_angle(x2)

        bbox_angle_min = min(left_angle, right_angle)
        bbox_angle_max = max(left_angle, right_angle)

        camera_tracks.append(
            {
                "track_id": track_id,
                "class": class_name,
                "confidence": confidence,
                "bbox": (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ),
                "camera_angle": center_angle,
                "bbox_angle_min": bbox_angle_min,
                "bbox_angle_max": bbox_angle_max,
                "direction": get_direction(center_angle),
            }
        )

    # Πρώτα τα πιο confident camera tracks όταν δύο bbox angular
    # intervals ανταγωνίζονται για το ίδιο LiDAR cluster.
    camera_tracks.sort(
        key=lambda obj: obj["confidence"],
        reverse=True,
    )

    return result, camera_tracks


# ==========================================================
# LIDAR MEASUREMENT -> EXISTING CAMERA ID
# ==========================================================


def predict_distance_for_track(track_id, current_time):
    """
    Πρόβλεψη της τωρινής απόστασης ενός camera Track ID.

    d_pred = d_last - closing_speed * dt

    Χρησιμοποιείται μόνο ως βοήθεια για το LiDAR association.
    """

    state = sensor_states.get(track_id)

    if state is None:
        return None

    if state["last_distance"] is None or state["last_lidar_seen"] is None:
        return None

    dt = current_time - state["last_lidar_seen"]

    if dt <= 0 or dt > MAX_PREDICTION_GAP:
        return None

    closing_speed_mps = state["closing_speed_kmh"] / 3.6

    predicted_distance = (
        state["last_distance"]
        - closing_speed_mps * dt
    )

    return max(0.0, predicted_distance)


def attach_lidar_measurements(camera_tracks, lidar_objects, current_time):
    """
    Κουμπώνει LiDAR measurements πάνω σε υπάρχον BoT-SORT camera ID.

    Το camera ID ΔΕΝ αλλάζει.

    Για κάθε πιθανό Camera↔LiDAR ζευγάρι χρησιμοποιούμε:
      1) το angular interval του YOLO bbox
      2) την απόσταση από το optical center
      3) όταν υπάρχει history, το predicted distance του ίδιου ID

    Έπειτα κάνουμε global greedy assignment στα καλύτερα candidate pairs,
    ώστε ένα LiDAR cluster και ένα camera ID να χρησιμοποιούνται μία φορά.
    """

    for track in camera_tracks:
        track["lidar_distance"] = None
        track["lidar_angle"] = None
        track["lidar_confidence"] = None
        track["lidar_angle_error"] = None
        track["lidar_prediction_error"] = None
        track["predicted_distance_before_match"] = predict_distance_for_track(
            track["track_id"],
            current_time,
        )
        track["lidar_match_status"] = "NO_LIDAR_MATCH"

    candidate_pairs = []

    for track_index, track in enumerate(camera_tracks):
        interval_min = track["bbox_angle_min"] - BBOX_ANGLE_MARGIN
        interval_max = track["bbox_angle_max"] + BBOX_ANGLE_MARGIN

        half_span = max(
            1.0,
            (track["bbox_angle_max"] - track["bbox_angle_min"]) / 2
            + BBOX_ANGLE_MARGIN,
        )

        predicted_distance = track["predicted_distance_before_match"]

        for lidar_index, lidar_obj in enumerate(lidar_objects):
            lidar_angle = lidar_obj["angle"]

            if not (interval_min <= lidar_angle <= interval_max):
                continue

            center_error = abs(track["camera_angle"] - lidar_angle)
            angle_score = center_error / half_span

            prediction_error = None

            if predicted_distance is not None:
                prediction_error = abs(
                    lidar_obj["distance"] - predicted_distance
                )

                # Αν η μέτρηση είναι πολύ μακριά από αυτό που περιμένουμε
                # για το ίδιο Track ID, την απορρίπτουμε ως πιθανό wrong object.
                if prediction_error > MAX_PREDICTION_ERROR_M:
                    continue

                prediction_score = (
                    prediction_error / MAX_PREDICTION_ERROR_M
                )

                # Η prediction μετρά περισσότερο από το center angle,
                # γιατί θέλουμε συνέχεια της ίδιας απόστασης/κίνησης.
                match_score = (
                    0.35 * angle_score
                    + 0.65 * prediction_score
                )
            else:
                # Χωρίς history βασιζόμαστε μόνο στη γεωμετρία του bbox.
                match_score = angle_score

            candidate_pairs.append(
                (
                    match_score,
                    center_error,
                    prediction_error,
                    track_index,
                    lidar_index,
                )
            )

    candidate_pairs.sort(key=lambda item: item[0])

    used_track_indices = set()
    used_lidar_indices = set()

    for (
        match_score,
        center_error,
        prediction_error,
        track_index,
        lidar_index,
    ) in candidate_pairs:

        if track_index in used_track_indices:
            continue

        if lidar_index in used_lidar_indices:
            continue

        track = camera_tracks[track_index]
        lidar_obj = lidar_objects[lidar_index]

        track["lidar_distance"] = lidar_obj["distance"]
        track["lidar_angle"] = lidar_obj["angle"]
        track["lidar_confidence"] = lidar_obj["lidar_confidence"]
        track["lidar_angle_error"] = center_error
        track["lidar_prediction_error"] = prediction_error
        track["lidar_match_score"] = match_score
        track["lidar_match_status"] = "MATCHED_TO_CAMERA_ID"

        used_track_indices.add(track_index)
        used_lidar_indices.add(lidar_index)

    return camera_tracks


# ==========================================================
# CLOSING SPEED PER CAMERA ID
# ==========================================================


def update_sensor_states(camera_tracks, current_time, ego_speed_kmh):
    """
    Κρατά LiDAR distance / closing-speed history ανά CAMERA Track ID.

    Closing speed:
      > 0 : η απόσταση κλείνει
      < 0 : η απόσταση ανοίγει

    Το stable closing speed είναι median των τελευταίων αποδεκτών
    instantaneous estimates ώστε ένα μεμονωμένο spike να μην χαλάει
    όλο το history.
    """

    for obj in camera_tracks:
        track_id = obj["track_id"]
        state = sensor_states.get(track_id)

        if state is None:
            state = {
                "history": [],
                "speed_history": [],
                "last_camera_seen": current_time,
                "last_lidar_seen": None,
                "last_distance": None,
                "closing_speed_kmh": 0.0,
            }
            sensor_states[track_id] = state

        state["last_camera_seen"] = current_time

        distance = obj["lidar_distance"]

        if distance is not None:
            previous_distance = state["last_distance"]
            previous_lidar_time = state["last_lidar_seen"]

            if previous_distance is not None and previous_lidar_time is not None:
                dt = current_time - previous_lidar_time

                if 0 < dt <= MAX_LIDAR_GAP_FOR_SPEED:
                    instantaneous_closing_speed_kmh = (
                        (previous_distance - distance) / dt
                    ) * 3.6

                    # Απορρίπτουμε μόνο προφανή spikes. Η τιμή αυτή
                    # δεν χαρακτηρίζει safe/danger, είναι sensor sanity check.
                    if (
                        abs(instantaneous_closing_speed_kmh)
                        <= MAX_ABS_CLOSING_SPEED_KMH
                    ):
                        state["speed_history"].append(
                            instantaneous_closing_speed_kmh
                        )

                        if len(state["speed_history"]) > SPEED_HISTORY_LENGTH:
                            state["speed_history"].pop(0)

                        state["closing_speed_kmh"] = statistics.median(
                            state["speed_history"]
                        )
                else:
                    # Μεγάλο gap: δεν συγκρίνουμε παλιά και νέα μέτρηση
                    # για speed. Ξεκινάμε φρέσκο speed history.
                    state["speed_history"].clear()
                    state["closing_speed_kmh"] = 0.0

            state["last_distance"] = distance
            state["last_lidar_seen"] = current_time

            state["history"].append((current_time, distance))

            if len(state["history"]) > DISTANCE_HISTORY_LENGTH:
                state["history"].pop(0)

        closing_speed_mps = state["closing_speed_kmh"] / 3.6

        reference_distance = (
            distance
            if distance is not None
            else state["last_distance"]
        )

        if reference_distance is not None:
            predicted_next_distance = max(
                0.0,
                reference_distance
                - closing_speed_mps * SIMULATION_DT,
            )
        else:
            predicted_next_distance = None

        # Approximation για όχημα που κινείται περίπου στην ίδια
        # longitudinal κατεύθυνση με το ego. Δεν είναι ground truth.
        estimated_target_speed_kmh = (
            float(ego_speed_kmh) - state["closing_speed_kmh"]
        )

        obj["closing_speed_kmh"] = state["closing_speed_kmh"]
        obj["last_known_distance"] = state["last_distance"]
        obj["predicted_next_distance"] = predicted_next_distance
        obj["estimated_target_speed_kmh"] = estimated_target_speed_kmh

    # Local measurement state cleanup. Αυτό δεν ελέγχει BoT-SORT IDs.
    expired_ids = []

    for track_id, state in sensor_states.items():
        if current_time - state["last_camera_seen"] > SENSOR_STATE_MAX_AGE:
            expired_ids.append(track_id)

    for track_id in expired_ids:
        del sensor_states[track_id]

    return camera_tracks


# ==========================================================
# RESET BOT-SORT BETWEEN EPISODES
# ==========================================================


def reset_camera_tracker():
    """Καθαρίζει το persistent BoT-SORT state μετά από env.reset()."""

    predictor = getattr(model, "predictor", None)

    if predictor is None:
        return

    trackers = getattr(predictor, "trackers", None)

    if trackers is None:
        return

    for tracker in trackers:
        if hasattr(tracker, "reset"):
            tracker.reset()


# ==========================================================
# METADRIVE CONFIG
# ==========================================================

config = {
    "use_render": True,
    "image_observation": True,
    "agent_observation": LidarStateObservation,
    "manual_control": True,
    "controller": "keyboard",
    "traffic_density": 0.20,
    "map": 5,
    "camera_fov": CAMERA_FOV,
    "sensors": {
        "rgb_camera": [
            RGBCamera,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
        ]
    },
    "vehicle_config": {
        "show_lidar": True,
        "lidar": {
            "num_lasers": NUM_LASERS,
            "distance": LIDAR_DISTANCE,
            "num_others": 0,
        },
    },
}


# ==========================================================
# CREATE ENVIRONMENT
# ==========================================================

env = MetaDriveEnv(config)
obs, info = env.reset()

frame_counter = 0
simulation_time = 0.0
episode_id = 0

# ==========================================================
# MAIN LOOP
# ==========================================================

try:
        
    while True:
        obs, reward, terminated, truncated, info = env.step([0, 0])

        # Εμφάνιση πληροφοριών κάθε 20 frames
        if frame_counter % 20 == 0:
            print("METADRIVE INFO:", info)

        # Εμφάνιση σημαντικών συμβάντων
        if info.get("crash") or terminated or truncated:
            print("\n--- EPISODE EVENT ---")
            print("Crash:", info.get("crash"))
            print("Crash Vehicle:", info.get("crash_vehicle"))
            print("Crash Object:", info.get("crash_object"))
            print("Out of Road:", info.get("out_of_road"))
            print("Terminated:", terminated)
            print("Truncated:", truncated)
            print("Episode Length:", info.get("episode_length"))

        # Αποθήκευση αποτελέσματος όταν τελειώνει το επεισόδιο
        if terminated or truncated:
            with EVENT_LOG_FILE.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                if f.tell() == 0:
                    writer.writerow([
                        "episode_id",
                        "episode_length",
                        "crash",
                        "crash_vehicle",
                        "out_of_road"
                    ])

                writer.writerow([
                    episode_id,
                    info.get("episode_length"),
                    info.get("crash"),
                    info.get("crash_vehicle"),
                    info.get("out_of_road")
                ])

    # Εδώ συνεχίζει ο υπόλοιπος κώδικας της προσομοίωσης
        ego = env.agent

        simulation_time += SIMULATION_DT
        current_time = simulation_time

        # ==================================================
        # RGB CAMERA
        # ==================================================

        camera = env.engine.get_sensor("rgb_camera")
        frame_rgb = camera.get_rgb_array_cpu()

        frame_bgr = cv2.cvtColor(
            frame_rgb,
            cv2.COLOR_RGB2BGR,
        )

        # ==================================================
        # 1) CAMERA-FIRST TRACKING
        # ==================================================

        result, camera_tracks = get_camera_tracks(frame_bgr)

        # ==================================================
        # 2) LIDAR MEASUREMENTS
        # ==================================================

        lidar_objects = get_lidar_clusters(env, ego)

        # ==================================================
        # 3) ATTACH LIDAR TO EXISTING CAMERA IDS
        # ==================================================

        camera_tracks = attach_lidar_measurements(
            camera_tracks,
            lidar_objects,
            current_time,
        )

        # ==================================================
        # 4) DISTANCE HISTORY + CLOSING SPEED
        # ==================================================

        camera_tracks = update_sensor_states(
            camera_tracks,
            current_time,
            ego.speed_km_h,
        )

        # ==================================================
        # 5) LOG SENSOR-DERIVED SPEED DATA
        # ==================================================

        log_speed_measurements(
            camera_tracks,
            ego.speed_km_h,
            current_time,
            episode_id,
        )

        # ==================================================
        # VISUALIZATION
        # ==================================================

        # result.plot() ζωγραφίζει τα BoT-SORT track IDs.
        annotated_frame = result.plot()

        for obj in camera_tracks:
            x1, y1, x2, y2 = obj["bbox"]

            if obj["lidar_distance"] is not None:
                distance_text = f'{obj["lidar_distance"]:.1f}m'
                closing_text = f'{obj["closing_speed_kmh"]:+.1f}km/h'
            else:
                distance_text = "LiDAR:N/A"
                closing_text = ""

            text = (
                f'ID {obj["track_id"]} '
                f'{obj["class"].upper()} '
                f'{distance_text} '
                f'{closing_text}'
            ).strip()

            cv2.putText(
                annotated_frame,
                text,
                (x1, max(y1 - 25, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        cv2.imshow(
            "MetaDrive Camera-First BoT-SORT + LiDAR",
            annotated_frame,
        )

        # ==================================================
        # TERMINAL OUTPUT
        # ==================================================

        if frame_counter % PRINT_EVERY == 0:
            print("\n" + "=" * 78)
            print("SIMULATION TIME:", round(simulation_time, 2), "s")
            print("EGO SPEED:", round(ego.speed_km_h, 2), "km/h")
            print("VISIBLE CAMERA TRACKS:", len(camera_tracks))
            print("LIDAR OBJECTS:", len(lidar_objects))
            print("CSV LOG:", LOG_FILE.name)

            # ----------------------------------------------
            # CAMERA-FIRST TRACKS
            # ----------------------------------------------

            print("\n--- CAMERA / BOT-SORT TRACKS ---")

            if not camera_tracks:
                print("No confirmed camera tracks")

            for obj in camera_tracks:
                print(f'\nTRACK ID: {obj["track_id"]}')
                print(" Class:", obj["class"].upper())
                print(" YOLO confidence:", round(obj["confidence"], 2))
                print(
                    " Camera center angle:",
                    round(obj["camera_angle"], 1),
                    "deg",
                )
                print(
                    " Camera bbox angular range:",
                    f'{obj["bbox_angle_min"]:.1f} .. '
                    f'{obj["bbox_angle_max"]:.1f} deg',
                )

                if obj["lidar_distance"] is not None:
                    print(
                        " LiDAR distance:",
                        round(obj["lidar_distance"], 2),
                        "m",
                    )
                    print(
                        " LiDAR angle:",
                        round(obj["lidar_angle"], 1),
                        "deg",
                    )
                    print(
                        " Camera/LiDAR center error:",
                        round(obj["lidar_angle_error"], 1),
                        "deg",
                    )
                    print(
                        " LiDAR confidence:",
                        obj["lidar_confidence"],
                    )
                    print(
                        " Closing speed (relative):",
                        round(obj["closing_speed_kmh"], 2),
                        "km/h",
                    )
                    print(
                        " Ego speed:",
                        round(ego.speed_km_h, 2),
                        "km/h",
                    )
                    print(
                        " Predicted next distance:",
                        (
                            "N/A"
                            if obj["predicted_next_distance"] is None
                            else f'{obj["predicted_next_distance"]:.2f} m'
                        ),
                    )
                    print(
                        " Estimated target speed:",
                        round(obj["estimated_target_speed_kmh"], 2),
                        "km/h",
                    )
                    if obj.get("lidar_prediction_error") is not None:
                        print(
                            " LiDAR prediction error:",
                            round(obj["lidar_prediction_error"], 2),
                            "m",
                        )
                else:
                    print(" LiDAR distance: N/A")
                    print(
                        " Last known distance:",
                        (
                            "N/A"
                            if obj["last_known_distance"] is None
                            else f'{obj["last_known_distance"]:.2f} m'
                        ),
                    )
                    print(
                        " Predicted next distance:",
                        (
                            "N/A"
                            if obj["predicted_next_distance"] is None
                            else f'{obj["predicted_next_distance"]:.2f} m'
                        ),
                    )

                print(
                    " LiDAR attachment:",
                    obj["lidar_match_status"],
                )

            # ----------------------------------------------
            # RAW LIDAR DEBUG
            # ----------------------------------------------

            print("\n--- RAW LIDAR CLUSTERS ---")

            if not lidar_objects:
                print("No LiDAR objects")

            for index, lidar_obj in enumerate(lidar_objects, start=1):
                print(
                    f'Object {index}',
                    "| Distance:",
                    round(lidar_obj["distance"], 1),
                    "m",
                    "| Angle:",
                    round(lidar_obj["angle"], 1),
                    "deg",
                    "|",
                    lidar_obj["direction"],
                    "|",
                    lidar_obj["lidar_confidence"],
                )

            print("\n" + "=" * 78)

        frame_counter += 1

        # ==================================================
        # EXIT
        # ==================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # ==================================================
        # RESET EPISODE
        # ==================================================

        if terminated or truncated:
            obs, info = env.reset()

            sensor_states.clear()
            reset_camera_tracker()

            simulation_time = 0.0
            episode_id += 1
finally:
    cv2.destroyAllWindows()
    env.close()
