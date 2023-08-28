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
import ffmpeg
import tarfile
import PIL.Image


# Define path
video_root_add = "../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS"
# EPIC_100_train = (
#     "../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_train"
# )
# EPIC_100_train_rgbframes = (
#     "../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_train_rgbframes"
# )
# EPIC_100_val_rgbframes = (
#     "../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_val_rgbframes"
# )
# data_root_add = 'home/mona/VideoMAE/dataset/Epic-kitchen/raw_videos/'


# data_train = pd.read_csv("VideoMAE/dataset/Epic_kitchen/annotation/EPIC_100_train.csv")
data_train = pd.read_csv("VideoMAE/dataset/Epic_kitchen/annotation/EPIC_100_train.csv")
data_val = pd.read_csv(
    "VideoMAE/dataset/Epic_kitchen/annotation/EPIC_100_val.csv"
)


#######Parallel

# give a permission in terminal by: sudo chmod ugo+rwx ../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS
# if not os.path.exists(EPIC_100_train_rgbframes):
#     os.makedirs(EPIC_100_train_rgbframes)


start_frames_train = []
stop_frames_train = []
video_paths_train = []
IDs_train = []
fps_train = []

for i, item in data_train.iterrows():
    IDs_train.append(i)
    participant_id = item["participant_id"]
    video_id = item["video_id"]
    start_frame = item["start_frame"]
    start_frames_train.append(start_frame)
    stop_frame = item["stop_frame"]
    stop_frames_train.append(stop_frame)
    fps_train.append(item["fps"])
    video_path = (
        video_root_add
        + "/"
        + f"{participant_id}"
        + "/rgb_frames"
        + "/"
        + f"{video_id}"
    )
    video_paths_train.append(video_path)

#     if not os.path.exists(video_path):
#         os.makedirs(video_path)

#         os.system(f"tar xvf {video_path}.tar -C {video_path}")

#     # if video_id == "P01_02":
#     #     break


# if not os.path.exists(EPIC_100_val_rgbframes):
#     os.makedirs(EPIC_100_val_rgbframes)


start_frames_val = []
stop_frames_val = []
video_paths_val = []
IDs_val = []
fps_val = []
for i, item in data_val.iterrows():
    IDs_val.append(i)
    participant_id = item["participant_id"]
    video_id = item["video_id"]
    start_frame = item["start_frame"]
    start_frames_val.append(start_frame)
    stop_frame = item["stop_frame"]
    stop_frames_val.append(stop_frame)
    fps_val.append(item["fps"])
    video_path = (
        video_root_add
        + "/"
        + f"{participant_id}"
        + "/rgb_frames"
        + "/"
        + f"{video_id}"
    )
    video_paths_val.append(video_path)
    
#     if not os.path.exists(video_path):
#         os.makedirs(video_path)

#         os.system(f"tar xvf {video_path}.tar -C {video_path}")



EPIC_100_train = [EPIC_100_train_rgbframes] * len(video_paths_train)
EPIC_100_val = [EPIC_100_val_rgbframes] * len(video_paths_val)


# def trim(input_path, output_path, start=30, end=60):
#     input_stream = ffmpeg.input(input_path)

#     vid = input_stream.video.trim(start_frame=start, end_frame=end).setpts("PTS-STARTPTS")
#     # aud = input_stream.audio.filter_("atrim", start_frame=start, end_frame=end).filter_(
#     #     "asetpts", "PTS-STARTPTS"
#     # )

#     # joined = ffmpeg.concat(vid, aud, v=1, a=1).node
#     # output = ffmpeg.output(joined[0], joined[1], output_path)
#     # output.run()
#     output = ffmpeg.output(vid, output_path)

#     # set loglevel to quiet to avoid printing the whole ffmpeg command
#     ffmpeg.run(output, quiet=True)


def Epic_action_data_creator(start_frame, stop_frame, video_path, EPIC_100, fps, i):
    if "train" in EPIC_100:
        train_or_val = "train"
    elif "val" in EPIC_100:
        train_or_val = "val"

    if os.path.exists(f"{EPIC_100}/video_{i}.MP4") == True:
        print(f"video_{train_or_val}_{i} exists")
        return
    else:
        out_dir = os.path.join(EPIC_100, f"video_{i}.MP4")
        video_frames = []
        for i in range(start_frame, stop_frame+1):
            frame_template = "frame_{:010d}.jpg"
            frame = PIL.Image.open(str(video_path / frame_template.format(i + 1))).convert("RGB")
            video_frames.append(frame)
        out = cv2.VideoWriter(out_dir, cv2.VideoWriter_fourcc(*"mp4v"), fps)
        for frame in video_frames:
            out.write(np.array(frame)[:,:,::-1])
        out.release()
    
        

        # command = f"ffmpeg -i {video_path} -vcodec copy -acodec copy -ss {start_frame}00 -to {stop_frame}00 {out_dir} -loglevel quiet"
        # os.system(command)

# ....
# import cv2
# video_cap = cv2.VideoCapture("../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/P01/videos/P01_02.MP4")
# c = 0
# while video_cap.isOpened():
#     # `success` is a boolean and `frame` contains the next video frame
#     success, frame = video_cap.read()
#     if success == True:
#      c += 1
#      print (c)        trim(video_path, out_dir, start=start_frame, end=stop_frame)

        # decord_vr = decord.VideoReader(video_path, num_threads=1)
        # duration = len(decord_vr)
        # video_data = decord_vr.get_batch(list(range(duration))).asnumpy()

    print(f"video_{train_or_val}_{i}.MP4 successfully saved (train)")


# import cv2

# video_cap = cv2.VideoCapture("../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/P01/videos/P01_02.MP4")

# c = 0
# while video_cap.isOpened():
#     # `success` is a boolean and `frame` contains the next video frame
#     success, frame = video_cap.read()
#     if success == True:
#      c += 1
#      print (c)


i = 2
Epic_action_data_creator(
    start_frames_train[i],
    stop_frames_train[i],
    video_paths_train[i],
    EPIC_100_train[i],
    fps_train[i],
    IDs_train[i],
)

n_tasks = 20
print("Processing training videos...")
with Pool(n_tasks) as p:
    p.starmap(
        Epic_action_data_creator,
        [
            (start_frames, stop_frames, video_paths, EPIC_100, fps, i)
            for (start_frames, stop_frames, video_paths, EPIC_100, fps, i) in zip(
                start_frames_train,
                stop_frames_train,
                video_paths_train,
                EPIC_100_train,
                fps_train,
                IDs_train,
            )
        ],
    )

print("Processing validation videos...")
with Pool(n_tasks) as p:
    p.starmap(
        Epic_action_data_creator,
        [
            (start_frames, stop_frames, video_paths, EPIC_100, fps, i)
            for (start_frames, stop_frames, video_paths, EPIC_100, fps, i) in zip(
                start_frames_val,
                stop_frames_val,
                video_paths_val,
                EPIC_100_val,
                fps_val,
                IDs_val,
            )
        ],
    )

print("Done!")
