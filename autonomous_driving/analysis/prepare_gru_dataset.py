from pathlib import Path
import pandas as pd 
import numpy as np

#εντοπισμος τελευταιας εκτελεσης 

runs= list(Path("auto_runs").glob("*/episode_events.csv"))
latest=max(runs,key=lambda p:p.stat().st_mtime).parent

#φορτωση των csv
ego = pd.read_csv(latest/"ego_steps.csv")
vehicles = pd.read_csv(latest / "vehicle_measurements.csv")
events=pd.read_csv(latest/"episode_events.csv")


#κοινη ακριβεια χρονου γι ατην συνδεση των αρχειων 
ego["simulation_time_s"]=ego["simulation_time_s"].round(3)
vehicles["simulation_time_s"]=vehicles["simulation_time_s"].round(3)

#τα χαρακτηριστικα που εχουμε ηδη υπολογισει 


features=[
    "lidar_distance_m",
    "closing_speed_kmh",
    "camera_angle_deg",
    "lidar_angle_deg"
]

#ομαδοποιηση ανα επεισοδιο και frame 


objects_by_frame={}

for(episode,timestamp),group in vehicles.groupby(
    ["episode_id","simulation_time_s"]
):
    objects_by_frame[(episode,timestamp)]=(
        group[features].to_dict("records")
    )

ego["objects"]=[
    objects_by_frame.get(
        (row.episode_id,row.simulation_time_s),[]
    )
    for row in ego.itertuples()
]
ego["object_count"] = ego["objects"].apply(len)

print("Episodes:", len(events))
print("Total frames:", len(ego))
print("Total matched measurements:", len(vehicles))
print("Frames with objects:", (ego["object_count"] > 0).sum())

# Εμφανίζουμε ένα πραγματικό frame με ανιχνεύσεις
sample = ego[ego["object_count"] > 0].iloc[0]

print("\nExample frame:")
print("Episode:", sample["episode_id"])
print("Time:", sample["simulation_time_s"])
print("Ego speed:", sample["ego_speed_kmh"])
print("Detected objects:", sample["objects"])

def summarize_zone(objects,min_angle,max_angle):
    
    
    selected=[
        obj for obj in objects
        if min_angle <= obj["camera_angle_deg"]<max_angle
    ]

    distances= [

        obj["lidar_distance_m"]
        for obj in selected
        if pd.notna(obj["lidar_distance_m"])

    ]

    speeds=[
        obj["closing_speed_kmh"]
        for obj in selected
        if pd.notna(obj["closing_speed_kmh"])
    ]

    return(
        len(selected),
        min(distances,default=float("nan")),
        max(speeds,default=float("nan"))
    )


zones=[
    ("negative",float("-inf"),-11),
    ("center",-11,11),
    ("positive",11,float("inf"))
]


for name, min_angle,max_angle in zones:

    summaries=ego["objects"].apply(
        lambda objects: summarize_zone(
            objects,min_angle,max_angle
        )
    )

    columns = [
        f"{name}_count",
        f"{name}_min_distance",
        f"{name}_max_closing_speed"
    ]

    ego[columns] = pd.DataFrame(
        summaries.tolist(),
        index=ego.index
    )


print("\n--- FRAME FEATURES ---")

print(
    ego[[
        "episode_id",
        "simulation_time_s",
        "ego_speed_kmh",
        "object_count",
        "negative_count",
        "center_count",
        "positive_count",
        "center_min_distance"
    ]]
    .loc[ego["object_count"] > 0]
    .head(10)
    .to_string(index=False)
)



HORIZON = 2.0

events_by_id = events.set_index("episode_id")

labels = []

for row in ego.itertuples():

    event = events_by_id.loc[row.episode_id]

    remaining = round(
        event["end_time_s"] - row.simulation_time_s, 3
    )

    crashed = bool(event["crash_vehicle"])

    if remaining <= 0:
        label = None

    elif remaining > HORIZON:
        label = 0

    elif crashed:
        label = 1

    elif remaining == HORIZON:
        label = 0

    else:
        label = None

    labels.append(label)

ego["collision_next_2s"] = pd.array(
    labels, dtype="Int64"
)

print("\n--- COLLISION LABELS ---")
print(ego["collision_next_2s"].value_counts(dropna=False))

print("\n--- TIME CHECK ---")

episode_times = ego.groupby("episode_id").agg(
    first_time=("simulation_time_s", "min"),
    last_time=("simulation_time_s", "max"),
    frames=("simulation_time_s", "count")
).reset_index()

check = episode_times.merge(
    events[[
        "episode_id",
        "end_time_s",
        "crash_vehicle"
    ]],
    on="episode_id"
)

print(check.head(10).to_string(index=False))
print("\n--- LABEL CHECK EPISODE 0 ---")

print(
    ego.loc[
        (ego["episode_id"] == 0) &
        (ego["simulation_time_s"].isin([5.0, 23.0, 24.0, 24.9])),
        ["simulation_time_s", "collision_next_2s"]
    ].to_string(index=False)
)



WINDOW =20

feature_cols =["ego_speed_kmh"]

for zone in ["negative","center","positive"]:
    feature_cols.extend([
        f"{zone}_count",
        f"{zone}_min_distance",
        f"{zone}_max_closing_speed"

    ])
X=[]
y=[]
sample_episodes=[]


for episode_id,episode_data in ego.groupby("episode_id"):
    episode_data=episode_data.sort_values(
        "simulation_time_s"
    ).reset_index(drop=True)


    for end in range(WINDOW-1,len(episode_data)):
        window = episode_data.iloc[
            end-WINDOW+1: end+1
        ]
        label=window.iloc[-1]["collision_next_2s"]

        if pd.isna(label):
            continue

        times=window["simulation_time_s"].to_numpy()

        if not np.allclose(np.diff(times),0.1,atol=0.002):
            continue

        sequence = window[feature_cols].to_numpy(
            dtype=np.float32
        )

        X.append(sequence)
        y.append(int(label))
        sample_episodes.append(episode_id)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)
sample_episodes = np.array(sample_episodes)

print("\n--- GRU SEQUENCES ---")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Episodes:", len(np.unique(sample_episodes)))
print("Missing feature values:", np.isnan(X).sum())

print("\n--- SEQUENCE LABELS ---")
print(pd.Series(y).value_counts())
