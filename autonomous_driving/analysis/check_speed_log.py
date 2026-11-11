import pandas as pd


df = pd.read_csv(
    "speed_measurements.csv"
)


TRACK_ID = 4


track = df[
    df["track_id"] == TRACK_ID
].copy()


track = track.sort_values(
    "simulation_time_s"
)


track["distance_change_m"] = (
    track["lidar_distance_m"].shift(1)
    -
    track["lidar_distance_m"]
)


track["dt"] = (
    track["simulation_time_s"]
    -
    track["simulation_time_s"].shift(1)
)


track["calculated_closing_speed_kmh"] = (
    (
        track["distance_change_m"]
        /
        track["dt"]
    )
    *
    3.6
)


print(
    track[
        [
            "simulation_time_s",
            "track_id",
            "lidar_distance_m",
            "closing_speed_kmh",
            "calculated_closing_speed_kmh"
        ]
    ].to_string(
        index=False
    )
)