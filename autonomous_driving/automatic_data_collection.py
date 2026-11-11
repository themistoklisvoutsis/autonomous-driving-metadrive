"""Automated MetaDrive + camera/YOLO/BoT-SORT + LiDAR collection.

Tested statically (syntax), not executed against a local MetaDrive installation.

Examples:
  python automatic_data_collection.py --episodes 10 --max-steps 400 --mode mixed
  python automatic_data_collection.py --episodes 60 --max-steps 500 --mode mixed
  python automatic_data_collection.py --episodes 5 --max-steps 400 --preview

Outputs are created under auto_runs/<unique-run-id>/:
  vehicle_measurements.csv: valid camera/LiDAR matches, one row per matched track per step
  ego_steps.csv: one row per sim step, including when no camera/LiDAR track is matched
  episode_events.csv: outcome for each episode (including collector-imposed cutoffs)
  run_settings.json: experiment configuration and actual collection details

No SAFE/WARNING/DANGER labels are fabricated. All collision flags are ego-level
MetaDrive outcomes; they do not identify the particular observed target track.
"""

import argparse
import csv
import json
import math
import statistics
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.policy.idm_policy import IDMPolicy

# Existing thesis settings
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV = 640, 360, 66
CONFIDENCE_THRESHOLD = 0.50
RELEVANT_CLASSES = [0, 1, 2, 3, 5, 7]
NUM_LASERS = 72
LIDAR_DISTANCE = 40
ANGLE_PER_RAY = 360 / NUM_LASERS
FUSION_HALF_ANGLE = 33
MAX_DISTANCE_GAP = 3.0
MAX_ANGLE_GAP = ANGLE_PER_RAY * 1.5
BBOX_ANGLE_MARGIN = 3.0
BOT_SORT_TRACK_BUFFER = 45
BOT_SORT_APPEARANCE_THRESHOLD = 0.85
SPEED_HISTORY_LENGTH = 5
DISTANCE_HISTORY_LENGTH = 8
MAX_LIDAR_GAP_FOR_SPEED = 0.5
MAX_PREDICTION_GAP = 0.8
MAX_PREDICTION_ERROR_M = 4.0
MAX_ABS_CLOSING_SPEED_KMH = 150.0
SENSOR_STATE_MAX_AGE = 5.0
SIMULATION_DT = 0.1

MEASUREMENT_COLUMNS = [
    "run_id", "episode_id", "scenario_seed", "simulation_time_s", "track_id",
    "class", "ego_speed_kmh", "ego_heading_deg", "lidar_distance_m",
    "closing_speed_kmh", "camera_angle_deg", "lidar_angle_deg",
    "yolo_confidence", "lidar_confidence", "lidar_match_status",
]
EGO_COLUMNS = [
    "run_id", "episode_id", "scenario_seed", "simulation_time_s",
    "ego_speed_kmh", "ego_heading_deg", "matched_objects",
    "crash", "crash_vehicle", "out_of_road", "terminated", "truncated",
]
EVENT_COLUMNS = [
    "run_id", "episode_id", "scenario_seed", "mode", "episode_length",
    "end_time_s", "crash", "crash_vehicle", "crash_object", "out_of_road",
    "arrive_dest", "terminated", "truncated", "collector_limit", "completed",
    "matched_measurements",
]


class ExperimentIDMPolicy(IDMPolicy):
    """IDM ego policy with an optional *intentional simulation-only* late-braking mode.

    This is for generating stress-test episodes, not a safe driving policy.
    When enabled, it ignores the vehicle ahead until it is very close, and
    then delegates back to the ordinary IDM controller. Crashes are not guaranteed.
    """

    late_brake = False
    brake_trigger_m = 7.0

    def acceleration(self, front_obj, dist_to_front):
        if self.late_brake and front_obj is not None and dist_to_front > self.brake_trigger_m:
            return super().acceleration(None, dist_to_front)
        return super().acceleration(front_obj, dist_to_front)


def get_arguments():
    parser = argparse.ArgumentParser(description="Automated MetaDrive sensor-data collector")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400, help="Collector cutoff per episode")
    parser.add_argument("--mode", choices=["normal", "late_brake", "mixed"], default="mixed")
    parser.add_argument("--traffic-density", type=float, default=0.20)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--preview", action="store_true", help="Show annotated camera view")
    parser.add_argument("--output", default="auto_runs", help="Directory containing per-run folders")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("--episodes and --max-steps must be positive")
    if not 0 <= args.traffic_density <= 1:
        parser.error("--traffic-density must be between 0 and 1")
    return args


def open_csv(directory, name, fieldnames):
    f = (directory / name).open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    f.flush()
    return f, writer


def save_tracker_config(directory):
    path = directory / "botsort_metadrive_reid.yaml"
    path.write_text(
        f"""tracker_type: botsort
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
""", encoding="utf-8"
    )
    return path


def reset_tracker(model):
    predictor = getattr(model, "predictor", None)
    for tracker in (getattr(predictor, "trackers", None) or []):
        if hasattr(tracker, "reset"):
            tracker.reset()


def pixel_x_to_angle(pixel_x):
    center_x = CAMERA_WIDTH / 2
    focal = center_x / math.tan(math.radians(CAMERA_FOV / 2))
    return math.degrees(math.atan((pixel_x - center_x) / focal))


def get_camera_tracks(model, frame_bgr, tracker_config):
    result = model.track(
        frame_bgr, persist=True, tracker=str(tracker_config),
        conf=CONFIDENCE_THRESHOLD, imgsz=640,
        classes=RELEVANT_CLASSES, verbose=False,
    )[0]
    camera_tracks = []
    if result.boxes is None:
        return result, camera_tracks
    for box in result.boxes:
        if box.id is None:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        a1, a2 = pixel_x_to_angle(x1), pixel_x_to_angle(x2)
        camera_tracks.append({
            "track_id": int(box.id.item()),
            "class": result.names[int(box.cls.item())],
            "confidence": float(box.conf.item()),
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "camera_angle": pixel_x_to_angle((x1+x2)/2),
            "bbox_angle_min": min(a1, a2), "bbox_angle_max": max(a1, a2),
        })
    camera_tracks.sort(key=lambda o: o["confidence"], reverse=True)
    return result, camera_tracks


def get_lidar_clusters(env, ego):
    lidar = env.engine.get_sensor("lidar")
    cloud_points, _ = lidar.perceive(
        ego, physics_world=env.engine.physics_world.dynamic_world,
        num_lasers=NUM_LASERS, distance=LIDAR_DISTANCE, show=False,
    )
    hits = []
    for i, normalized_distance in enumerate(cloud_points):
        angle = i * ANGLE_PER_RAY
        angle = angle - 360 if angle > 180 else angle
        if abs(angle) > FUSION_HALF_ANGLE or normalized_distance >= 0.999:
            continue
        hits.append({"angle": angle, "distance": normalized_distance * LIDAR_DISTANCE})
    hits.sort(key=lambda h: h["angle"])
    if not hits:
        return []
    clusters, current = [], [hits[0]]
    for hit in hits[1:]:
        previous = current[-1]
        if (hit["angle"] - previous["angle"] <= MAX_ANGLE_GAP and
                abs(hit["distance"] - previous["distance"]) <= MAX_DISTANCE_GAP):
            current.append(hit)
        else:
            clusters.append(current)
            current = [hit]
    clusters.append(current)
    return [
        {
            "angle": statistics.mean(h["angle"] for h in group),
            "distance": statistics.median(h["distance"] for h in group),
            "confidence": "STRONG" if len(group) >= 2 else "WEAK",
        }
        for group in clusters
    ]


def predicted_distance(states, track_id, now):
    state = states.get(track_id)
    if state is None or state["last_distance"] is None or state["last_lidar_seen"] is None:
        return None
    dt = now - state["last_lidar_seen"]
    if dt <= 0 or dt > MAX_PREDICTION_GAP:
        return None
    return max(0.0, state["last_distance"] - state["closing_speed_kmh"] * dt / 3.6)


def attach_lidar(camera_tracks, lidar_objects, states, now):
    pairs = []
    for ti, obj in enumerate(camera_tracks):
        obj.update({
            "lidar_distance": None, "lidar_angle": None,
            "lidar_confidence": None, "lidar_match_status": "NO_LIDAR_MATCH",
        })
        low = obj["bbox_angle_min"] - BBOX_ANGLE_MARGIN
        high = obj["bbox_angle_max"] + BBOX_ANGLE_MARGIN
        half_span = max(1.0, (obj["bbox_angle_max"] - obj["bbox_angle_min"]) / 2 + BBOX_ANGLE_MARGIN)
        predicted = predicted_distance(states, obj["track_id"], now)
        for li, lidar in enumerate(lidar_objects):
            if not low <= lidar["angle"] <= high:
                continue
            center_error = abs(obj["camera_angle"] - lidar["angle"])
            angle_score = center_error / half_span
            if predicted is None:
                score = angle_score
            else:
                err = abs(lidar["distance"] - predicted)
                if err > MAX_PREDICTION_ERROR_M:
                    continue
                score = 0.35 * angle_score + 0.65 * err / MAX_PREDICTION_ERROR_M
            pairs.append((score, ti, li))
    pairs.sort(key=lambda p: p[0])
    used_t, used_l = set(), set()
    for _, ti, li in pairs:
        if ti in used_t or li in used_l:
            continue
        obj, lidar = camera_tracks[ti], lidar_objects[li]
        obj.update({
            "lidar_distance": lidar["distance"],
            "lidar_angle": lidar["angle"],
            "lidar_confidence": lidar["confidence"],
            "lidar_match_status": "MATCHED_TO_CAMERA_ID",
        })
        used_t.add(ti)
        used_l.add(li)
    return camera_tracks


def update_speeds(camera_tracks, states, now):
    for obj in camera_tracks:
        tid = obj["track_id"]
        state = states.setdefault(tid, {
            "speed_history": [], "distance_history": [],
            "last_distance": None, "last_lidar_seen": None,
            "closing_speed_kmh": 0.0, "last_camera_seen": now,
        })
        state["last_camera_seen"] = now
        distance = obj["lidar_distance"]
        if distance is not None:
            old_dist, old_time = state["last_distance"], state["last_lidar_seen"]
            if old_dist is not None and old_time is not None:
                dt = now - old_time
                if 0 < dt <= MAX_LIDAR_GAP_FOR_SPEED:
                    instant = (old_dist - distance) / dt * 3.6
                    if abs(instant) <= MAX_ABS_CLOSING_SPEED_KMH:
                        state["speed_history"].append(instant)
                        state["speed_history"] = state["speed_history"][-SPEED_HISTORY_LENGTH:]
                        state["closing_speed_kmh"] = statistics.median(state["speed_history"])
                else:
                    state["speed_history"].clear()
                    state["closing_speed_kmh"] = 0.0
            state["last_distance"] = distance
            state["last_lidar_seen"] = now
            state["distance_history"].append((now, distance))
            state["distance_history"] = state["distance_history"][-DISTANCE_HISTORY_LENGTH:]
        obj["closing_speed_kmh"] = state["closing_speed_kmh"]
    for tid in list(states):
        if now - states[tid]["last_camera_seen"] > SENSOR_STATE_MAX_AGE:
            del states[tid]


def episode_mode(mode, episode_index):
    if mode == "mixed":
        return "normal" if episode_index % 2 == 0 else "late_brake"
    return mode


def main():
    args = get_arguments()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    directory = Path(__file__).resolve().parent / args.output / run_id
    directory.mkdir(parents=True, exist_ok=False)
    tracker_path = save_tracker_config(directory)
    settings = {
        "run_id": run_id, "script": "automatic_data_collection.py",
        "episodes": args.episodes, "max_steps": args.max_steps,
        "mode": args.mode, "traffic_density": args.traffic_density,
        "seed_range": [args.start_seed, args.start_seed + args.episodes - 1],
        "simulation_dt_s": SIMULATION_DT, "preview": args.preview,
        "policy": "IDM / custom late-braking IDM (simulation experiment)",
        "note": "Episode labels are ego-level events, not per-target labels. Collector-limit episodes are censored.",
    }
    (directory / "run_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    meas_f, meas_w = open_csv(directory, "vehicle_measurements.csv", MEASUREMENT_COLUMNS)
    ego_f, ego_w = open_csv(directory, "ego_steps.csv", EGO_COLUMNS)
    event_f, event_w = open_csv(directory, "episode_events.csv", EVENT_COLUMNS)

    config = {
        "use_render": bool(args.preview),
        "image_observation": True,
        "agent_observation": LidarStateObservation,
        "manual_control": False,
        "agent_policy": ExperimentIDMPolicy,
        "num_scenarios": args.episodes,
        "start_seed": args.start_seed,
        "horizon": args.max_steps,
        "traffic_density": args.traffic_density,
        "map": 5,
        "camera_fov": CAMERA_FOV,
        "sensors": {"rgb_camera": [RGBCamera, CAMERA_WIDTH, CAMERA_HEIGHT]},
        "vehicle_config": {
            "show_lidar": False,
            "lidar": {"num_lasers": NUM_LASERS, "distance": LIDAR_DISTANCE, "num_others": 0},
        },
    }
    env = None
    total_frames, total_meas, crashes, ended = 0, 0, 0, 0
    started_at = time.perf_counter()
    try:
        print("Loading YOLO and MetaDrive ...")
        model = YOLO("yolo26n.pt")
        env = MetaDriveEnv(config)
        camera = None
        quit_requested = False
        for episode_id in range(args.episodes):
            mode = episode_mode(args.mode, episode_id)
            ExperimentIDMPolicy.late_brake = mode == "late_brake"
            scenario_seed = args.start_seed + episode_id
            env.reset(seed=scenario_seed)
            reset_tracker(model)
            states = {}
            if camera is None:
                camera = env.engine.get_sensor("rgb_camera")
            print(f"\nEpisode {episode_id+1}/{args.episodes} | seed={scenario_seed} | mode={mode}", flush=True)
            episode_meas = 0
            last_info = {}
            terminated = truncated = False
            step_count = 0
            for step_count in range(1, args.max_steps + 1):
                _, _, terminated, truncated, info = env.step([0, 0])
                last_info = info
                now = round(step_count * SIMULATION_DT, 3)
                ego = env.agent
                heading_deg = math.degrees(float(ego.heading_theta))
                matched_count = 0

                # Process sensor output only before the ending/collision step.
                # At the terminal step we still record ego ground truth below.
                if not (terminated or truncated):
                    frame_rgb = camera.get_rgb_array_cpu()
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    yolo_result, tracks = get_camera_tracks(model, frame_bgr, tracker_path)
                    lidar_clusters = get_lidar_clusters(env, ego)
                    attach_lidar(tracks, lidar_clusters, states, now)
                    update_speeds(tracks, states, now)
                    for obj in tracks:
                        if obj["lidar_distance"] is None:
                            continue
                        meas_w.writerow({
                            "run_id": run_id, "episode_id": episode_id, "scenario_seed": scenario_seed,
                            "simulation_time_s": now, "track_id": obj["track_id"],
                            "class": obj["class"], "ego_speed_kmh": round(float(ego.speed_km_h), 3),
                            "ego_heading_deg": round(heading_deg, 3),
                            "lidar_distance_m": round(float(obj["lidar_distance"]), 3),
                            "closing_speed_kmh": round(float(obj["closing_speed_kmh"]), 3),
                            "camera_angle_deg": round(float(obj["camera_angle"]), 3),
                            "lidar_angle_deg": round(float(obj["lidar_angle"]), 3),
                            "yolo_confidence": round(float(obj["confidence"]), 4),
                            "lidar_confidence": obj["lidar_confidence"],
                            "lidar_match_status": obj["lidar_match_status"],
                        })
                        matched_count += 1
                    if args.preview:
                        annotated = yolo_result.plot()
                        cv2.imshow("MetaDrive | Automated collection | Q to stop", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            quit_requested = True

                ego_w.writerow({
                    "run_id": run_id, "episode_id": episode_id, "scenario_seed": scenario_seed,
                    "simulation_time_s": now, "ego_speed_kmh": round(float(ego.speed_km_h), 3),
                    "ego_heading_deg": round(heading_deg, 3), "matched_objects": matched_count,
                    "crash": bool(info.get("crash", False)),
                    "crash_vehicle": bool(info.get("crash_vehicle", False)),
                    "out_of_road": bool(info.get("out_of_road", False)),
                    "terminated": bool(terminated), "truncated": bool(truncated),
                })
                episode_meas += matched_count
                total_meas += matched_count
                total_frames += 1
                if step_count % 100 == 0:
                    elapsed = time.perf_counter() - started_at
                    print(f"  step {step_count:4d} | total {total_frames:6d} frames | "
                          f"avg {total_frames / max(elapsed, .001):.1f} FPS", flush=True)
                    meas_f.flush()
                    ego_f.flush()
                if terminated or truncated or quit_requested:
                    break

            # A collector cutoff is not a completed MetaDrive outcome.
            collector_limit = not (terminated or truncated)
            completed = bool(terminated or truncated)
            collision = bool(last_info.get("crash_vehicle", False))
            crashes += int(collision)
            event_w.writerow({
                "run_id": run_id, "episode_id": episode_id, "scenario_seed": scenario_seed,
                "mode": mode, "episode_length": last_info.get("episode_length", step_count),
                "end_time_s": round(step_count * SIMULATION_DT, 3),
                "crash": bool(last_info.get("crash", False)),
                "crash_vehicle": collision,
                "crash_object": bool(last_info.get("crash_object", False)),
                "out_of_road": bool(last_info.get("out_of_road", False)),
                "arrive_dest": bool(last_info.get("arrive_dest", False)),
                "terminated": bool(terminated), "truncated": bool(truncated),
                "collector_limit": collector_limit, "completed": completed,
                "matched_measurements": episode_meas,
            })
            meas_f.flush(); ego_f.flush(); event_f.flush()
            ended += 1
            print(f"  end: steps={step_count}, crash_vehicle={collision}, "
                  f"completed={completed}, matches={episode_meas}", flush=True)
            if quit_requested:
                break

    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C. Saved files contain completed rows.", flush=True)
    finally:
        elapsed = time.perf_counter() - started_at
        print(f"\nSaved to: {directory}\nEpisodes logged: {ended} | "
              f"vehicle collisions: {crashes} | matched rows: {total_meas} | "
              f"average FPS: {total_frames / max(elapsed, 0.001):.2f}", flush=True)
        for f in (meas_f, ego_f, event_f):
            f.close()
        cv2.destroyAllWindows()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
