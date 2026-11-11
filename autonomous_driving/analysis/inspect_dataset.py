import pandas as pd

path = "auto_runs/20261106_190345_a7eebf/"

events = pd.read_csv(path + "episode_events.csv")

print(events[[
    "episode_id",
    "mode",
    "crash_vehicle",
    "arrive_dest",
    "out_of_road",
    "terminated",
    "truncated",
    "matched_measurements"
]].to_string(index=False))


measurements = pd.read_csv(
    path + "vehicle_measurements.csv"
)

print("\n--- LAST 2 SECONDS PER EPISODE ---")

for _, event in events.iterrows():

    episode = event["episode_id"]
    end_time = event["end_time_s"]

    recent = measurements[
        (measurements["episode_id"] == episode) &
        (measurements["simulation_time_s"] >= end_time - 2) &
        (measurements["simulation_time_s"] < end_time)
    ]

    print(
        f"Episode {episode} | "
        f"Crash: {event['crash_vehicle']} | "
        f"Measurements: {len(recent)} | "
        f"Track IDs: {recent['track_id'].nunique()}"
    )