from pathlib import Path
import pandas as pd

runs = list(Path("auto_runs").glob("*/episode_events.csv"))
latest = max(runs, key=lambda p: p.stat().st_mtime).parent

events = pd.read_csv(latest / "episode_events.csv")
ego = pd.read_csv(latest / "ego_steps.csv")
measurements = pd.read_csv(latest / "vehicle_measurements.csv")

print("RUN:", latest)
print("EPISODES:", len(events))
print("COLLISIONS:", events["crash_vehicle"].sum())
print("OUT OF ROAD:", events["out_of_road"].sum())
print("MATCHED MEASUREMENTS:", len(measurements))
print("TOTAL FRAMES:", len(ego))
print("FRAMES WITH MATCHES:", (ego["matched_objects"] > 0).sum())

print("\n--- RESULTS BY MODE ---")
print(events.groupby("mode").agg(
    episodes=("episode_id", "count"),
    collisions=("crash_vehicle", "sum"),
    out_of_road=("out_of_road", "sum"),
    total_matches=("matched_measurements", "sum")
).to_string())