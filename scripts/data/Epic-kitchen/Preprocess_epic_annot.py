import csv
import glob
import os
from multiprocessing import Pool

import cv2
import decord
import numpy as np
import pandas as pd
from decord import VideoReader
from PIL import Image
# from https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes.git install epic-kitchens library
from epic_kitchens.hoa import load_detections, DetectionRenderer
import pickle

# Define path
annot_root_add = "../../mnt/welles/scratch/datasets/Epic-kitchen/annotation/hand-objects"
EPIC_100_hand_objects_train = (
    "../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_hand_objects_train"
)
EPIC_100_hand_objects_val = (
    "../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_hand_objects_val"
)


data_train = pd.read_csv("VideoMAE/dataset/Epic_kitchen/annotation/EPIC_100_train.csv")
data_val = pd.read_csv(
    "VideoMAE/dataset/Epic_kitchen/annotation/EPIC_100_validation.csv"
)


#######Parallel

# give a permission in terminal by: sudo chmod ugo+rwx ../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS
if not os.path.exists(EPIC_100_hand_objects_train):
    os.makedirs(EPIC_100_hand_objects_train)


start_frames_train = []
stop_frames_train = []
annot_paths_train = []
IDs_train = []
start_timestamps_train = []
stop_timestamps_train = []


for i, item in data_train.iterrows():
    IDs_train.append(i)
    participant_id = item["participant_id"]
    video_id = item["video_id"]
    start_frame = item["start_frame"]
    start_frames_train.append(start_frame)
    stop_frame = item["stop_frame"]
    stop_frames_train.append(stop_frame)
    annot_path = (
        annot_root_add
        + "/"
        + f"{participant_id}"
        + "/"
        + f"{video_id}"
        + ".pkl"
    )
    annot_paths_train.append(annot_path)
    start_timestamp = item["start_timestamp"]
    start_timestamps_train.append(start_timestamp)
    stop_timestamp = item["stop_timestamp"]
    stop_timestamps_train.append(stop_timestamp)


if not os.path.exists(EPIC_100_hand_objects_val):
    os.makedirs(EPIC_100_hand_objects_val)


start_frames_val = []
stop_frames_val = []
annot_paths_val = []
IDs_val = []
start_timestamps_val = []
stop_timestamps_val = []

for i, item in data_val.iterrows():
    IDs_val.append(i)
    participant_id = item["participant_id"]
    video_id = item["video_id"]
    start_frame = item["start_frame"]
    start_frames_val.append(start_frame)
    stop_frame = item["stop_frame"]
    stop_frames_val.append(stop_frame)
    annot_path = (
        annot_root_add
        + "/"
        + f"{participant_id}"
        + "/"
        + f"{video_id}"
        + ".pkl"
    )
    annot_paths_val.append(annot_path)
    start_timestamp = item["start_timestamp"]
    start_timestamps_val.append(start_timestamp)
    stop_timestamp = item["stop_timestamp"]
    stop_timestamps_val.append(stop_timestamp)

 
EPIC_100_hand_objects_train = [
    "../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_hand_objects_train"
] * len(annot_paths_train)
EPIC_100_hand_objects_val = [
    "../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_hand_objects_val"
] * len(annot_paths_val)


def Epic_detection_annot_creator(start_frames, stop_frames, annot_path, EPIC_100, i):
    if "train" in EPIC_100:
        train_or_val = "train"
    elif "val" in EPIC_100:
        train_or_val = "val"

    if (
        os.path.exists(
            f"{EPIC_100}/detection_{i}.pkl"
        )
        == True
    ):
        print(f"detection_{train_or_val}_{i} exists")
        return
    else:
        out_dir = os.path.join(EPIC_100, f"detection_{i}.pkl")
        detections = load_detections(annot_path)[start_frames:stop_frames]

        detected_objects_BB = []
        detected_hands_BB = []
        # each element of detected_objects_BB and detected_hands_BB is a list contains object(s)/hand(s) BB information of a frame.
        for detect in detections:
            detected_objects_BB.append([[object.bbox.left, object.bbox.top, object.bbox.right, object.bbox.bottom] for object in detect.objects])
            detected_hands_BB.append([[hand.bbox.left, hand.bbox.top, hand.bbox.right, hand.bbox.bottom] for hand in detect.hands])

        # data = pickle. load(open(annot_path, "rb"))
        # in each detection_{i}_.pkl you can find a dictionary that contains:
        pickle.dump({"objects" : detected_objects_BB, "hands":detected_hands_BB}, open(out_dir, "wb"))

    print(f"detection_{train_or_val}_{i}.pkl successfully saved")
    

# i = 98
# Epic_detection_annot_creator(start_frames_train[i],stop_frames_train[i], annot_paths_train[i],EPIC_100_hand_objects_train[i], IDs_train[i])

n_tasks = 20
print("Processing training videos...")
with Pool(n_tasks) as p:
    p.starmap(
        Epic_detection_annot_creator,
        [
            (start_frames, stop_frames, annot_paths, EPIC_100, i)
            for (start_frames, stop_frames, annot_paths, EPIC_100, i) in zip(
                start_frames_train,
                stop_frames_train,
                annot_paths_train,
                EPIC_100_hand_objects_train,
                IDs_train,
            )
        ],
    )

print("Processing validation videos...")
with Pool(n_tasks) as p:
    p.starmap(
        Epic_detection_annot_creator,
        [
            (start_frames, stop_frames, annot_paths, EPIC_100, i)
            for (start_frames, stop_frames, annot_paths, EPIC_100, i) in zip(
                start_frames_val,
                stop_frames_val,
                annot_paths_val,
                EPIC_100_hand_objects_val,
                IDs_val,
            )
        ],
    )

print("Done!")
