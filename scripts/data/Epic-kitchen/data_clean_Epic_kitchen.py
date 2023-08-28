import os
import subprocess
import sys
import time
from multiprocessing import Pool
import orjson 
import decord
import ffmpeg
from joblib import delayed, Parallel  # install psutil library to manage memory leak
# from tqdm import tqdm
import pandas as pd
# from https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes.git install epic-kitchens library
# from epic_kitchens.hoa import load_detections, DetectionRenderer
import PIL.Image
import pickle
import cv2
# import torch
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union, List
import re
# from ipywidgets import interact, IntSlider, Layout


# Define path

root_add = "path to Epic-kitchen dataset"
save_root_add = "path to save the processed videos"
# Define videos and annotations path
EPIC_100_train = root_add + "/" + "train"
EPIC_100_val =  root_add + "/" + "validation"
EPIC_100_hand_objects_train = ("path to Epic-kitchen dataset/EPIC_100_hand_objects_train")
EPIC_100_hand_objects_val = ("path to Epic-kitchen dataset/EPIC_100_hand_objects_val")


data_train = pd.read_csv("Epic_kitchen/EPIC_100_train.csv")
data_val = pd.read_csv("Epic_kitchen/EPIC_100_validation.csv")


if not os.path.exists(save_root_add):
    os.makedirs(save_root_add+ "/" + "train")
    os.makedirs(save_root_add+ "/" + "validation")

# #fuction for visualizing image with bounding box
# def visual_bbx (images, bboxes):
#     """
#     images (torch.Tensor or np.array): list of images in torch or numpy type.
#     bboxes (List[List]): list of list having bounding boxes in [x1, y1, x2, y2]
#     """
#     if isinstance(images, torch.Tensor):
#         images = images.view((16, 3) + images.size()[-2:])
#     color_list = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0, 255,255)]
#     if not os.path.exists('VideoMAE/scripts/data/Epic-kitchen/visual_bbx'):
#         os.makedirs('VideoMAE/scripts/data/Epic-kitchen/visual_bbx')
#     for i, (img, bbx) in enumerate(zip(images, bboxes)):
#         if isinstance(img, Image.Image) or isinstance(img, np.ndarray):
#             frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#             # (x1,y1,x2,y2) = (bbx[0], bbx[1], bbx[2], bbx[3])
#         elif isinstance(img, torch.Tensor):
#             frame = img.numpy().astype(np.uint8).transpose(1, 2, 0)
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#             # if len(bbx) != 0:
#             #     (x1,y1,x2,y2) = (bbx[0][0], bbx[0][1], bbx[0][2], bbx[0][3])

#         if len(bbx) != 0:
#             # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_list[0], 4)
#             ##
#             for c, b in enumerate(bbx):
#                 cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color_list[0], 4)
#                 # cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[3]), int(b[2])), color_list[1], 4)
#                 # cv2.rectangle(frame, (int(b[0]), int(b[2])), (int(b[1]), int(b[3])), color_list[2], 4)
#                 # cv2.rectangle(frame, (int(b[1]), int(b[0])), (int(b[3]), int(b[2])), color_list[3], 4)
#                 # cv2.rectangle(frame, (int(b[2]), int(b[1])), (int(b[0]), int(b[3])), color_list[4], 4)
#                 # cv2.rectangle(frame, (int(b[3]), int(b[1])), (int(b[2]), int(b[0])), color_list[0], 4)
#                 # cv2.rectangle(frame, (int(b[1]), int(b[0])), (int(b[2]), int(b[3])), color_list[1], 4)
#             ##
#         cv2.imwrite(f'VideoMAE/scripts/data/Epic-kitchen/visual_bbx/{i}.png', frame)



def data_clean(video_root_path, idx, verbose=False):

    # prepare
    video_ext = 'mp4'

    # pick train or val
    train_flag = 'train' if 'train' in video_root_path else 'validation'

    # process video & save
    start_time = time.time()

    directory = os.path.join(video_root_path, f"video_{idx}.MP4" )
    if '.' in directory.split('/')[-1]:
        video_name = directory
    else:
        video_name = '{}.{}'.format(directory, video_ext)

    try:
        # try load video

        decord_vr = decord.VideoReader(video_name, num_threads=1)

        duration = len(decord_vr)
        # if duration < 30:
        #     return [-1, -1, -1]
        video_data = decord_vr.get_batch(list(range(duration))).asnumpy()

        # get the new size (short side size 320p)
        _, img_h, img_w, _ = video_data.shape
        new_short_size = 320
        ratio = float(img_h) / float(img_w)
        if ratio >= 1.0:
            new_w = int(new_short_size)
            new_h = int(new_w * ratio / 2) * 2

        else:
            new_h = int(new_short_size)
            new_w = int(new_h / ratio / 2) * 2
        
        #scale the BBx 
        y_ratio_bb = new_h/img_h
        x_ratio_bb = new_w/img_w
        new_BB_dict = [int(idx), x_ratio_bb, y_ratio_bb, img_h, img_w]

        new_size = (new_w, new_h)
    except Exception as e:
        # skip corrupted video files
        print("Failed to load video from {} with error {}".format(
            video_name, e))
        return [-1, -1, -1]

    # visulaize video and BBs
    
    # visual_bbx([video_data[0]], [hands_bbx_norm[0]])

    # process the video
    final_save_root_add = os.path.join(save_root_add, train_flag)
    output_video_file = os.path.join(final_save_root_add, directory.replace('.MP4', '.mp4').split('/')[-1])

    # resize
    proc1 = (ffmpeg.input(directory).filter(
        'scale', new_size[0],
        new_size[1]).output(output_video_file).overwrite_output())
    p = subprocess.Popen(
        ['ffmpeg'] + proc1.get_args()+
        ['-hide_banner', '-loglevel', 'quiet', '-nostats'])


    end_time = time.time()
    dur_time = end_time - start_time
    if verbose:
        print(f'processing video {idx + 1} with total time {dur_time} & save video in {output_video_file}')

    return new_BB_dict

def scale_BB(BB_root_path, ratio_list):

    # pick train or val
    train_flag = 'train' if 'train' in BB_root_path else 'validation'

    # save scaled BB into a dict
    scaled_BB = {}

    for ratio in ratio_list:
        idx, x_ratio_bb, y_ratio_bb, img_h, img_w = ratio

        BB_path = os.path.join(BB_root_path, f"detection_{idx}.pkl")
        detection = pickle.load(open(BB_path, "rb"))

        objects =  detection["objects"]
        hands = detection["hands"]
        scaled_BB_dict = {}

        labels = []
        for objects_bbxs, hands_bbxs in zip(objects, hands):
            object_frame_bbx = {}
            hand_frame_bbx = {}
            labels_frame = []
            for object_bbx in objects_bbxs:
                object_frame_bbx['box2d'] = {'x1': object_bbx[0]*img_w*x_ratio_bb, 'y1': object_bbx[1]*img_h*y_ratio_bb,
                                        'x2': object_bbx[2]*img_w*x_ratio_bb, 'y2': object_bbx[3]*img_h*y_ratio_bb}
                object_frame_bbx['gt_annotation'] = 'object'
                labels_frame.append(object_frame_bbx)
            
            for hand_bbx in hands_bbxs:
                hand_frame_bbx['box2d'] = {'x1': hand_bbx[0]*img_w*x_ratio_bb, 'y1': hand_bbx[1]*img_h*y_ratio_bb,
                                        'x2': hand_bbx[2]*img_w*x_ratio_bb, 'y2': hand_bbx[3]*img_h*y_ratio_bb}
                hand_frame_bbx['gt_annotation'] = 'hand'
                labels_frame.append(hand_frame_bbx)
            
            labels.append({'labels':labels_frame})

        # labels = {'labels': labels}

        # scaled_BB_dict['name'] = f'video_{idx}'
        # scaled_BB_dict['labels'] = labels
        scaled_BB_dict[f'video_{idx}'] = labels

        scaled_BB.update(scaled_BB_dict)

    # save the scaled BB
    out_path = os.path.join(save_root_add, f"EPIC_100_BB_{train_flag}.json")
    with open(out_path, "w+", encoding="utf-8") as f:
        f.write(orjson.dumps(scaled_BB).decode("utf-8"))

    print(f"save scaled BB in {out_path}")


if __name__ == '__main__':
    n_tasks = 4
    # new_start_idxs = [0] * n_tasks

    # # test one case/
    # ratio = data_clean(EPIC_100_train, 10)
    # scale_BB(EPIC_100_hand_objects_train, [ratio])

    with Pool(n_tasks) as p:
        train_out = p.starmap(data_clean,[ (video_root_path, idx)
                   for (video_root_path, idx) in zip ([EPIC_100_train]*len(data_train), range(len(data_train))) ])

    scale_BB(EPIC_100_hand_objects_train, train_out)

    
    with Pool(n_tasks) as p:
        val_out = p.starmap(data_clean,[ (video_root_path, idx)
                   for (video_root_path, idx) in zip ([EPIC_100_val]*len(data_train), range(len(data_val))) ])

    scale_BB(EPIC_100_hand_objects_val, val_out)
