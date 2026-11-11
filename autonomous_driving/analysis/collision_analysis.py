#Εδω συνδεουμε τα 2 csv μεσω του episode_id , βρισκει το επεισοδιο στο οποιο συνεβη συγκρουση 
# και επιλεγει τις προηγουμενες μετρησεις . 
import pandas as pd 

measurements=pd.read_csv("speed_measurements.csv")
events=pd.read_csv("episode_events.csv")

collision = events[events["crash_vehicle"]==True].iloc[0]

episode_id=collision["episode_id"]
collision_time=collision["episode_length"]*0.1


before_crash = measurements[
    (measurements["episode_id"] == episode_id)&
    (measurements["simulation_time_s"] < collision_time)
].copy()

print("Collision time:", collision_time)
print("Measurements before crash:", len(before_crash))

print(before_crash[
    [
        "simulation_time_s",
        "track_id",
        "class",
        "lidar_distance_m",
        "closing_speed_kmh",
        "camera_angle_deg"
    ]
].tail(15).to_string(index=False))# Αρχικά θεωρούμε άπειρο TTC
before_crash["ttc_s"] = float("inf")

# Επιλέγουμε μετρήσεις με θετική closing speed
approaching = before_crash["closing_speed_kmh"] > 0

# Υπολογίζουμε TTC
before_crash.loc[approaching, "ttc_s"] = (
    before_crash.loc[approaching, "lidar_distance_m"] /
    (before_crash.loc[approaching, "closing_speed_kmh"] / 3.6)
)
print("\n--- TTC BEFORE COLLISION ---")

print(before_crash.loc[
    before_crash["simulation_time_s"] >= collision_time - 1,
    [
        "simulation_time_s",
        "track_id",
        "lidar_distance_m",
        "closing_speed_kmh",
        "ttc_s"
    ]
].to_string(index=False))