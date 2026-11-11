import pandas as pd


#διαβαζω το csv και  χωριζω το καθε try του σιμ 
df=pd.read_csv("speed_measurements.csv")
print("Total measurements:", len(df))

df["episode_id"]=(
    df["simulation_time_s"].diff()<0
).cumsum()

print(
    df[df["track_id"]==4]
    .groupby("episode_id")
    .size()
)




#επιλεγουμε το επεισοδιο με τα περισσοτερα track_id==4 για να εξετασουμε 
episode=(
    df.loc[df["track_id"]==4,"episode_id"]
    .value_counts()
    .idxmax()

) 
#λοκαρω σε ενα oxhma 
track=df.loc[
    (df["track_id"]== 4)&
    (df["episode_id"]==episode)
    ].sort_values("simulation_time_s").copy()

track["gap_s"]= track["simulation_time_s"].diff() #xroniko keno 
track["large_gap"]= track["gap_s"] > 0.5 # ελεγχουμε αν το κενο ειναι μεγαλυτερο απο 0.5
#ελεγχουμε αν η κατηγορια του οχηματος αλλαξε 
track["class_changed"]=(
    track["class"] != track["class"].shift()
)
#αποφασιζουμε αν ξεκινα νεα τροχια *****
track["new_trajectory"]=(
    track["large_gap"]| track["class_changed"]
)
#δινω id σε καθε τροχια 
track["trajectory_id"] = track["new_trajectory"].cumsum()
track["dt"]=(
    track.groupby("trajectory_id")["simulation_time_s"].diff()
)
track["delta_distance"] = (
    track.groupby("trajectory_id")["lidar_distance_m"].diff()
)

track["closing_speed_calculated"]=(
    -track["delta_distance"]/track["dt"]
)* 3.6

print(track["dt"].describe())

print(track[
    [
        "simulation_time_s",
        "lidar_distance_m",
        "closing_speed_kmh",
        "closing_speed_calculated"
    ]
].head(20).to_string(index=False))
print("\n--- TRAJECTORIES ---")

print(
    track.groupby(["trajectory_id", "class"])
    .agg(
        measurements=("lidar_distance_m", "count"),
        start_time=("simulation_time_s", "min"),
        end_time=("simulation_time_s", "max"),
        start_distance=("lidar_distance_m", "first"),
        end_distance=("lidar_distance_m", "last")
    ).to_string()
)
car=track[track["trajectory_id"]==4].copy()

car["lidar_distance_m"]
car["closing_speed_kmh"]
car["ttc_s"] = car["lidar_distance_m"]/((car["closing_speed_kmh"])/3.6)
car.loc[car["closing_speed_kmh"]<=0,"ttc_s"]=float("inf")


min_index=car["ttc_s"].idxmin()

critical_moment=car.loc[min_index]
print("Distance:",critical_moment["lidar_distance_m"])
print("TTC:",critical_moment["ttc_s"])
print("Time:",critical_moment["simulation_time_s"])



print("\n--- VEHICLE POSITION ANALYSIS ---")

print(car[[
    "simulation_time_s",
    "lidar_distance_m",
    "closing_speed_kmh",
    "ttc_s",
    "camera_angle_deg",
    "lidar_angle_deg"
]].to_string(index=False))