import os
import cv2
import argparse
import pickle
import decord
from tqdm import tqdm
import numpy as np
from glob import glob
from joblib import delayed, Parallel
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset.process_data.process_video import get_video_frames_cv, compute_TVL1
from sts.motion_sts import compute_motion_boudary, motion_mag_downsample, zero_boundary
import pandas as pd
import shutil
import orjson
from PIL import Image
import PIL.Image
from scipy.ndimage import gaussian_filter
import scipy.stats as stats

                  
# dict from video_names/ each value of the video_name is a list of dictionaries (each dictionary is a frame)
# each dictionary has 2 keys: 'labels'/ key of 'labels' is a list of labels of the a dictionary : 
# 'box2d' : {'x1': .. , 'y1': .., 'x2': .., 'y2': ..}, and 'gt_annotation': 'union'
def parse_option():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--video-add-train', type=str, default="../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_videos/train", help='path of motion map data')
    parser.add_argument('--video-add-val', type=str, default="../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_videos/validation")
    
    parser.add_argument('--motion-map-add-train', type=str, default="../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_motion_map_videos/train", help='path of motion map data')
    parser.add_argument('--motion-map-add-val', type=str, default="../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_motion_map_videos/validation")
    
    parser.add_argument('--optical-flow-add-train', type=str, default="../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_optical_flow_videos/train", help='path of motion map data')
    parser.add_argument('--optical-flow-add-val', type=str, default="../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_optical_flow_videos/validation")
  
    parser.add_argument('--dst-BB-path', type=str, default='../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition', help='path to store generated data')
    parser.add_argument('--dataset', type=str, default='Epic_Kitchens', choices=['Epic_Kitchens','SSV2'], help='dataset to training')
    parser.add_argument('--video-type', type=str, default='mp4', choices=['mp4', 'avi'], help='which data')
    parser.add_argument('--num-workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--clip-length', type=int, default=16, help='num of clip length')
    parser.add_argument('--num-steps', type=int, default=2, help='num of sampling steps')


    args = parser.parse_args()
    return args

# get the bounding box of the motion map
def BB_detector (video_add, motion_map_add, json_saved_add):
    train_flag = 'train' if 'train' in motion_map_add else 'validation'

    # read the motion map from their videos and save the bounding box in a csv file
    videos = glob(os.path.join(video_add,'*.mp4'))
    # mm_videos = glob(os.path.join(motion_map_add,'*.mp4'))
    print('begin')

    # get the video name
    video_names = [os.path.basename(video) for video in videos]

    def json_creator (video_add, video_name, motion_map_add, json_saved_add):

        out_path = os.path.join(json_saved_add, f"BB_{train_flag}", f"EPIC_100_unsupervised_BB_mm_{video_name.split('.')[0]}.json")

        if os.path.exists(out_path):
            return

        bbx_and_frame = []
        union_bbx_and_frame = []
        if 'video_'in video_name:
            # print (video_name)
            video_path = os.path.join(video_add, video_name)
            # video_name_mm = video_name.replace('_org.mp4', '.mp4')
            mm_video_path = os.path.join(motion_map_add, video_name)
            video = decord.VideoReader(video_path)
            mm_video = decord.VideoReader(mm_video_path)
            mm_video_length = len(mm_video)
            video_length = len(video)
            #get the video frames
            video_frames = video.get_batch(range(video_length)).asnumpy() #min_value = 0, max_value = 255
            motion_map_frames = mm_video.get_batch(range(mm_video_length)).asnumpy()

            BB_dict = {}
            labels = []
            # print(video_frames.shape)  # (number of the frame, 320, 570, 3)
            # print(motion_map_frames.shape)
            
            masked_frames = []
            x1s = []
            x2s = []
            y1s = []
            y2s = [] 
            bbx_midddle_x = []
            bbx_midddle_y = []
            for i in range(motion_map_frames.shape[0]):
                frame_bbx = {}
                labels_frame = []
                h,w,c = motion_map_frames[i].shape
                max_pixel = np.max(motion_map_frames[i])
                min_pixel = np.min(motion_map_frames[i])
                # print (f"frame {i}: max_pixel = {max_pixel}, min_pixel = {min_pixel}")


                #visualize the motion map
                # frame = PIL.Image.fromarray(motion_map_frames[i].astype(np.uint8))
                # frame.save(os.path.join( "../mona" , f"{video_name}_{i}_mm_before_gaussian.jpg"))

                #apply gussian filter to the motion map
                before_sigma = 1 #1,0.4,30 for having acceptable bbxs for the frames but the new setting works for union bbx
                motion_map_frames[i] = gaussian_filter(motion_map_frames[i], sigma=before_sigma)

                #visualize the motion map
                # frame = PIL.Image.fromarray(  [i].astype(np.uint8))
                # frame.save(os.path.join( "../mona" , f"{video_name}_{i}_mm_after_first_gaussian.jpg"))

                max_pixel_after_gaussian = np.max(motion_map_frames[i])


                # # put zero for the pixels which are less than 0.9 of the maximum pixel value
                remove_thrd = 0.4
                motion_map_frames[i][motion_map_frames[i] < remove_thrd * max_pixel_after_gaussian] = 0 
                
                sigma = np.std(motion_map_frames[i]) + 1e-5
                motion_map_frames[i][motion_map_frames[i] < 1.5*sigma] = 0#1.5*sigma
                
                #visualize the motion map
                # frame = PIL.Image.fromarray(motion_map_frames[i].astype(np.uint8))
                # frame.save(os.path.join( "../mona" , f"{video_name}_{i}_mm_after_std.jpg"))


 
                # apply gussian filter to the motion map
                after_sigma = 30
                motion_map_frames[i] = gaussian_filter(motion_map_frames[i], sigma=after_sigma)

                # plt.hist(gaussian_filter(a, sigma=0.5,radius=1).reshape(-1), bins=20, range=(0.1,30))
                # plt.savefig('a.png')
                # plt.close()

                # #visualize the motion map
                # frame = PIL.Image.fromarray(motion_map_frames[i].astype(np.uint8))
                # frame.save(os.path.join( "../mona" , f"{video_name}_{i}_mm_after_second_gaussian.jpg"))





                # find contours in the motion map
                #gray the image using absdiff
                gray = cv2.cvtColor(motion_map_frames[i].astype(np.uint8), cv2.COLOR_BGR2GRAY)

                contours, hierarchy = cv2.findContours(gray.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # print (f"frame {i}: number of contours = {len(contours)}")

                # find the two largest contours
                contours = sorted(contours, key=cv2.contourArea, reverse=True)


                # #visualize the contours
                # cv2.drawContours(motion_map_frames[i], contours, -1, (0,255,0), 3)
                # frame = PIL.Image.fromarray(motion_map_frames[i].astype(np.uint8))
                # frame.save(os.path.join( "../mona" , f"{video_name}_{i}_contours.jpg"))

                # pick the two largest contours
                if len(contours) >= 2:
                    con_len = 2

                    ## find the middle point of each contour
                    middle_points = []
                    for j in range(con_len):
                        middle_points.append(np.mean(contours[j], axis=0))
                    # find the distance between the middle points
                    distances = []
                    for j in range(con_len-1):
                        distances.append(np.linalg.norm(middle_points[j] - middle_points[j+1]))
                    
                    # if the distance between the middle points is larger than 100, then pick the largest contour
                    if np.max(distances) > 0.4*np.sqrt(h**2 + w**2):
                        con_len = 1
                    
                else:
                    con_len = len(contours)



                new_motion_map = np.zeros_like(motion_map_frames[i])
                # set the pixels in the two largest contours to 1
                for j in range(con_len):
                    new_motion_map = cv2.drawContours(new_motion_map, contours, j, (255,255,255), -1)
                motion_map_frames[i] = new_motion_map

                # visualize the motion map
                # frame = PIL.Image.fromarray(motion_map_frames[i].astype(np.uint8))
                # frame.save(os.path.join( "../mona" , f"{video_name}_{i}_mm_after_contour.jpg"))


                frame_video = video_frames[i].copy()

                #create a mask from motion map which convert values over ziroes to 1 and values equal to zero to 0
                thrd = 0
                mask = np.zeros(motion_map_frames[i].shape)
                mask[motion_map_frames[i] > thrd] = 1
                frame_video = frame_video * mask
                masked_frames.append(frame_video)

                
                # # visualize the masked frame
                # frame = PIL.Image.fromarray(masked_frames[i].astype(np.uint8))
                # frame.save(os.path.join( "../mona" , f"{video_name}_{i}_mask.jpg"))
                # # # print (f"save {video_name}_{i}_.jpg")


                #count the number of zero and non-zero pixels in the frame_video
                zero_pixels_f = (frame_video <= thrd).sum()
                non_zero_pixels_f = (frame_video > thrd).sum()
                # print (f"frame {i}: zero_pixels = {zero_pixels_f}, non_zero_pixels = {non_zero_pixels_f}")

                zero_pixels = (motion_map_frames[i] <= thrd).sum()
                non_zero_pixels = (motion_map_frames[i] > thrd).sum()
                # print (f"frame {i}: zero_pixels = {zero_pixels}, non_zero_pixels = {non_zero_pixels}")

                # find the nonzero pixel locations in farame_video
                nonzero_pixel_locations = np.nonzero(frame_video)

                # find the maximum and minimum of nonzero pixel locations in farame_video ###frame_video.shape=(320, 570, 3)
                if len(nonzero_pixel_locations[0]) == 0:
                    # if there is no non-zero pixel in the frame, then put previous frame's bounding box
                    if i == 0:
                        x_1 = int (w/4)
                        y_1 = int (h/4)
                        x_2 = int (3*w/4)
                        y_2 = int (3*h/4)
                else:
                    y_1 = np.min(nonzero_pixel_locations[0])
                    x_1 = np.min(nonzero_pixel_locations[1])
                    y_2 = np.max(nonzero_pixel_locations[0])
                    x_2 = np.max(nonzero_pixel_locations[1])
                        # print (f"frame {i}: x_1 = {x_1}, y_1 = {y_1}, x_2 = {x_2}, y_2 = {y_2}")    

                # visualize the video frame with bounding box
                # frame_video_bb = video_frames[i].copy()
                # cv2.rectangle(frame_video_bb, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)
                # frame = PIL.Image.fromarray(frame_video_bb.astype(np.uint8))
                # frame.save(os.path.join( "../mona" , f"{video_name}_{i}_with_bb.jpg"))

                    
                # dict from video_names/ each value of the video_name is a list of dictionaries (each dictionary is a frame)
                # each dictionary has 2 keys: 'labels'/ key of 'labels' is a list of labels of the a dictionary : 
                # 'box2d' : {'x1': .. , 'y1': .., 'x2': .., 'y2': ..}, and 'gt_annotation': 'union'
                

                frame_bbx['box2d'] = {'x1': x_1, 'y1': y_1, 'x2': x_2, 'y2': y_2}
                frame_bbx['gt_annotation'] = 'union'
                labels_frame.append(frame_bbx.copy())
                labels.append({'labels':labels_frame})

            # remove the bbxs with ((x_2-x_1) > 0.6*w) or ((y_2-y_1) > 0.6*h) and also replace bounding boxes less than 0.01 of h and w with the bbx with the bbx of the next frame
            all_vars = []
            for i in range(len(labels)):
                x1i = labels[i]['labels'][0]['box2d']['x1']
                x2i = labels[i]['labels'][0]['box2d']['x2']
                y1i = labels[i]['labels'][0]['box2d']['y1']
                y2i = labels[i]['labels'][0]['box2d']['y2']
                x1k = labels[i]['labels'][0]['box2d']['x1']
                x2k = labels[i]['labels'][0]['box2d']['x2']
                y1k = labels[i]['labels'][0]['box2d']['y1']
                y2k = labels[i]['labels'][0]['box2d']['y2']
                j = i
                # while (((x2i-x1i) * (y2i-y1i) > 0.6*w*h) or ((x2i-x1i) * (y2i-y1i) < 0.01*w*h)) and j < len(labels)-1:
                while (((x2i-x1i)> 0.7*w) or ((y2i-y1i) > 0.7*h) or ((x2i-x1i) * (y2i-y1i) < 0.01*w*h)) and j < len(labels)-1:
                    labels[i]['labels'][0]['box2d']['x1'] = labels[j+1]['labels'][0]['box2d']['x1']
                    labels[i]['labels'][0]['box2d']['y1'] = labels[j+1]['labels'][0]['box2d']['y1']
                    labels[i]['labels'][0]['box2d']['x2'] = labels[j+1]['labels'][0]['box2d']['x2']
                    labels[i]['labels'][0]['box2d']['y2'] = labels[j+1]['labels'][0]['box2d']['y2']
                    # print (f"frame {i}: remove bbx and replace it with the bbx of the next frame")
                    x1i = labels[j]['labels'][0]['box2d']['x1']
                    x2i = labels[j]['labels'][0]['box2d']['x2']
                    y1i = labels[j]['labels'][0]['box2d']['y1']
                    y2i = labels[j]['labels'][0]['box2d']['y2']
                    j += 1
                
                    if j == len(labels)-1:
                        k = i
                        if ((x2i-x1i) > 0.7*w) or ((y2i-y1i) > 0.7*h):
                            labels[k]['labels'][0]['box2d']['x1'] = int(x1k/2) 
                            labels[k]['labels'][0]['box2d']['y1'] = int(y1k/2)
                            labels[k]['labels'][0]['box2d']['x2'] = int(x2k/2)
                            labels[k]['labels'][0]['box2d']['y2'] = int(y2k/2)
                        elif ((x2i-x1i) < 0.01*w) or ((y2i-y1i) < 0.01*h):
                            labels[k]['labels'][0]['box2d']['x1'] = int(0.25*w) 
                            labels[k]['labels'][0]['box2d']['y1'] = int(0.25*h)
                            labels[k]['labels'][0]['box2d']['x2'] = int(0.75*w)
                            labels[k]['labels'][0]['box2d']['y2'] = int(0.75*h)


                # #create a rectangle around the bbx
                # x1 = labels[i]['labels'][0]['box2d']['x1']
                # y1 = labels[i]['labels'][0]['box2d']['y1']
                # x2 = labels[i]['labels'][0]['box2d']['x2']
                # y2 = labels[i]['labels'][0]['box2d']['y2']
                # cv2.rectangle(masked_frames[i], (x1, y1), (x2, y2), (0, 255, 0), 2)
                # frame = PIL.Image.fromarray(masked_frames[i].astype(np.uint8))
                # frame.save(os.path.join( "../mona" , f"{video_name}_{i}_mask.jpg"))
                # print (f"save {video_name}_{i}_.jpg") 

                if i == 0:
                    mean_x1_union = labels[i]['labels'][0]['box2d']['x1']
                    mean_y1_union = labels[i]['labels'][0]['box2d']['y1']
                    mean_x2_union = labels[i]['labels'][0]['box2d']['x2']
                    mean_y2_union = labels[i]['labels'][0]['box2d']['y2']
                else:
                    mean_x1_union = int(np.mean(x1s))
                    mean_y1_union = int(np.mean(y1s))
                    mean_x2_union = int(np.mean(x2s))
                    mean_y2_union = int(np.mean(y2s))

                x1s.append(labels[i]['labels'][0]['box2d']['x1'])
                y1s.append(labels[i]['labels'][0]['box2d']['y1'])
                x2s.append(labels[i]['labels'][0]['box2d']['x2'])
                y2s.append(labels[i]['labels'][0]['box2d']['y2'])

                #find the middle point of the bbx
                x_mid = int((x1s[i] + x2s[i])/2)
                y_mid = int((y1s[i] + y2s[i])/2)
                bbx_midddle_x.append(x_mid)
                bbx_midddle_y.append(y_mid)


            # smooth the bbxs based on the previous and next bbxs
                #compute variance of union bbx
                var_x1_union =int(np.mean(abs( x1s[i]- mean_x1_union)**2))
                var_y1_union = int(np.mean(abs( y1s[i]- mean_y1_union)**2))
                var_x2_union = int(np.mean(abs( x2s[i] - mean_x2_union)**2))
                var_y2_union = int(np.mean(abs( y2s[i]- mean_y2_union)**2))
                var_union = (var_x1_union + var_x2_union + var_y1_union + var_y2_union)/4

                all_vars.append(var_union)


            for i in range(len(labels)):
                    if all_vars[i] > (labels[i]['labels'][0]['box2d']['x2']-labels[i]['labels'][0]['box2d']['x1'])*(labels[i]['labels'][0]['box2d']['y2']-labels[i]['labels'][0]['box2d']['y1'])*0.1:
                        # if i < len(labels)-7:
                        #     if not ((bbx_midddle_x[i+1]-bbx_midddle_x[i+2] < 2 or bbx_midddle_y[i+1]-bbx_midddle_y[i+2] < 1) and \
                        # (bbx_midddle_x[i+2]-bbx_midddle_x[i+3] < 2 or bbx_midddle_y[i+2]-bbx_midddle_y[i+3] < 1) and \
                        # (bbx_midddle_x[i+3]-bbx_midddle_x[i+4] < 2 or bbx_midddle_y[i+3]-bbx_midddle_y[i+4] < 1) and \
                        # (bbx_midddle_x[i+4]-bbx_midddle_x[i+5] < 2 or bbx_midddle_y[i+4]-bbx_midddle_y[i+5] < 1)):#.25
                        labels[i]['labels'][0]['box2d']['x1'] = labels[i-1]['labels'][0]['box2d']['x1']
                        labels[i]['labels'][0]['box2d']['y1'] = labels[i-1]['labels'][0]['box2d']['y1']
                        labels[i]['labels'][0]['box2d']['x2'] = labels[i-1]['labels'][0]['box2d']['x2']
                        labels[i]['labels'][0]['box2d']['y2'] = labels[i-1]['labels'][0]['box2d']['y2']
            

                    # add padding to the bbx
                    x1 = labels[i]['labels'][0]['box2d']['x1']
                    y1 = labels[i]['labels'][0]['box2d']['y1']
                    x2 = labels[i]['labels'][0]['box2d']['x2']
                    y2 = labels[i]['labels'][0]['box2d']['y2']


                    # padding = 0.1
                    if (x2-x1) <= 0.4*w:
                        x1 = x1 - 0.05*(x2-x1) 
                        x2 = x2 + 0.05*(x2-x1)

                    if (y2-y1) <= 0.4*h:
                        y1 = y1 - 0.05*(y2-y1)
                        y2 = y2 + 0.05*(y2-y1)

                    # # make bbx square
                    # if (x2-x1) > (y2-y1):
                    #     y1 = y1 - 0.5*((x2-x1)-(y2-y1))
                    #     y2 = y2 + 0.5*((x2-x1)-(y2-y1))
                    # else:
                    #     x1 = x1 - 0.5*((y2-y1)-(x2-x1))
                    #     x2 = x2 + 0.5*((y2-y1)-(x2-x1))


                    x1 = int(max(0, x1))
                    y1 = int(max(0, y1))    
                    x2 = int(min(w, x2))
                    y2 = int(min(h, y2))
                    labels[i]['labels'][0]['box2d']['x1'] = x1
                    labels[i]['labels'][0]['box2d']['y1'] = y1
                    labels[i]['labels'][0]['box2d']['x2'] = x2
                    labels[i]['labels'][0]['box2d']['y2'] = y2
                
                    # #create a rectangle around the bbx
                    # cv2.rectangle(video_frames[i], (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # frame = PIL.Image.fromarray(video_frames[i].astype(np.uint8))
                    # frame.save(os.path.join( "../mona" , f"{video_name}_{i}_bbx.jpg"))###############
                    # bbx_and_frame.append(frame)
                    # # print (f"save {video_name}_{i}.jpg")
                    # 
            # # comoute the an unic union over the frames using min and max of the bbx
            x1s = [labels[i]['labels'][0]['box2d']['x1'] for i in range(len(labels))]
            y1s = [labels[i]['labels'][0]['box2d']['y1'] for i in range(len(labels))]
            x2s = [labels[i]['labels'][0]['box2d']['x2'] for i in range(len(labels))]
            y2s = [labels[i]['labels'][0]['box2d']['y2'] for i in range(len(labels))]
            x1 = min(x1s)
            y1 = min(y1s)
            x2 = max(x2s)
            y2 = max(y2s)

            for i in range(len(labels)):
                labels[i]['labels'][0]['box2d']['x1'] = x1
                labels[i]['labels'][0]['box2d']['y1'] = y1
                labels[i]['labels'][0]['box2d']['x2'] = x2
                labels[i]['labels'][0]['box2d']['y2'] = y2

            
                #create a rectangle around the bbx
                # cv2.rectangle(video_frames[i], (x1, y1), (x2, y2), (0, 255, 0), 2)
                # frame = PIL.Image.fromarray(video_frames[i].astype(np.uint8))
                # # frame.save(os.path.join( "../mona" , f"{video_name}_{i}_bbx.jpg"))###############
                # union_bbx_and_frame.append(frame)        

            BB_dict[f'{video_name}'] = labels
                                
            # out_dir = os.path.join("../mona" , f"{video_name}_mm_unsupervised_bbxs.mp4")
            # out = cv2.VideoWriter(filename=out_dir, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=30 , frameSize=(570,320), isColor=True)
            # for f in union_bbx_and_frame:
            #     out.write(np.array(f)[:,:,::-1])
            # out.release()
            # # print(f'visualizing video and bbx & save it in {out_dir}')

   
        
        # # save the bbxs in a json file
        
        # if the parent directory does not exist, create it
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        with open(out_path, "w+", encoding="utf-8") as f:
            f.write(orjson.dumps(BB_dict).decode("utf-8"))
        # print (f"save the bbxs in {out_path}")

        # # save the video of new frames
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(os.path.join( "../mona" , f"{video_name}_detection.mp4"), fourcc, 50.0, (570, 320))
        # for frame in masked_frames:
        #     out.write(np.array(frame)[:,:,::-1])
        # out.release()
        # print (f"save {video_name}_detection.mp4")
    
    Parallel(n_jobs=8)(delayed(json_creator)(vid_add, vid_name, mm_add, save_add) for (vid_add, vid_name, mm_add, save_add) in tqdm(zip([video_add]*len(videos), video_names, [motion_map_add]*len(videos), [json_saved_add]*len(videos)), total=len(videos)))
    # test one case
    # json_creator(video_add, video_names[1], motion_map_add, json_saved_add)



### main function
args = parse_option()
BB_detector(args.video_add_train, args.motion_map_add_train, args.dst_BB_path)
BB_detector(args.video_add_val, args.motion_map_add_val, args.dst_BB_path)

# read all json files in BB_train
print("read all json files in BB_train")
json_files = glob(os.path.join(args.dst_BB_path, "BB_train", "*.json"))
json_list = {}
for file in json_files:
    with open(file, "r", encoding="utf-8") as f:
        json_file = orjson.loads(f.read())
        #remove .mp4 from the key
        json_file = {k.replace(".mp4", ""): v for k, v in json_file.items()}
        json_list.update(json_file)

# join all json files in one json file
print("join all json files in one json file")
with open(os.path.join(args.dst_BB_path, "Unsupervised_BB_EPIC_100_train.json"), "w+", encoding="utf-8") as f:
    f.write(orjson.dumps(json_list).decode("utf-8"))

print("train is done")


# read all json files
print("read all json files in BB_validation")
json_files = glob(os.path.join(args.dst_BB_path, "BB_validation", "*.json"))
json_list = {}
for file in json_files:
    with open(file, "r", encoding="utf-8") as f:
        json_file = orjson.loads(f.read())
        #remove .mp4 from the key
        json_file = {k.replace(".mp4", ""): v for k, v in json_file.items()}
        json_list.update(json_file)

# join all json files in one json file
with open(os.path.join(args.dst_BB_path, "Unsupervised_BB_EPIC_100_validation.json"), "w+", encoding="utf-8") as f:
    f.write(orjson.dumps(json_list).decode("utf-8"))

print("validation is done")

# # # load the bbxs for comparison
# with open('../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/EPIC_100_BB_smooth_train.json', "r", encoding="utf-8") as f:
#     Total_BB_train = orjson.loads(f.read())

# video_list = ["video_10","video_11","video_77","video_139","video_152","video_239","video_456","video_502","video_723","video_1024","video_4583", "video_9304", "video_10387","video_11736", "video_23442", "video_30377", "video_48204", "video_50673", "video_66245", "video_67000"]


# video_path = "../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_videos/train"
# for i in range (len(video_list)):
#     BBs = Total_BB_train[video_list[i]]
#     video_name = video_list[i] + ".mp4"

#     vid_path = os.path.join(video_path, video_name)
#     video = decord.VideoReader(vid_path)
#     video_length = len(video)
#     video_frames = video.get_batch(range(video_length)).asnumpy()

    # # load the bbxs
    # bbx_and_frame = []
    # for j in range(video_length):
    #     w , h = video_frames[j].shape[1], video_frames[j].shape[0]
    #     x1 = BBs[j]['labels'][0]['box2d']['x1']
    #     y1 = BBs[j]['labels'][0]['box2d']['y1']
    #     x2 = BBs[j]['labels'][0]['box2d']['x2']
    #     y2 = BBs[j]['labels'][0]['box2d']['y2']
    #     cv2.rectangle(video_frames[j], (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)        
    #     frame = PIL.Image.fromarray(video_frames[j].astype(np.uint8))
    #     bbx_and_frame.append(frame)
    #     frame.save(os.path.join( "../mona" , f"{video_name}_{j}_supervised.jpg"))
    #     # print (f"save {video_name}_{i}.jpg")
    
    # out_dir = os.path.join("../mona" , f"{video_name}_supervised_bbxs.mp4")
    # out = cv2.VideoWriter(filename=out_dir, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=30 , frameSize=(w,h), isColor=True)
    # for f in bbx_and_frame:
    #     out.write(np.array(f)[:,:,::-1])
    # out.release()
    # print(f'visualizing video {i} and bbx & save it in {out_dir}')
