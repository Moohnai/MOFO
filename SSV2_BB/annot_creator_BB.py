import pandas as pd
import json
import os, cv2
import gdown
import orjson

# read json bbox files
Total_video_BB={}
with open('/home/mona/VideoMAE/SSV2_BB/bounding_box_smthsmth_scaled.json', "r", encoding="utf-8") as f:
    Total_video_BB = orjson.loads(f.read())


# create dataframe
train_df = {'path':[], 'label_name':[], 'label_num':[]}
val_df = {'path':[], 'label_name':[], 'label_num':[]}

# root addresses
root_add = "/home/mona/VideoMAE/dataset/somethingsomething/"
video_mp4_root_add = "/home/mona/VideoMAE/dataset/somethingsomething/mp4_videos_BB"

f = open(os.path.join(root_add, 'labels','labels.json'))
labels = json.load(f)


f = open(os.path.join(root_add, 'labels','train.json'))
train_label = json.load(f)
for i in train_label:
    id = i['id']
    path = os.path.join(video_mp4_root_add, id+'.mp4')
    
    # check if the mp4 file exists
    if not os.path.exists(path):
        continue

    # check if bbox information exists in Total_video_BB
    if not id in Total_video_BB:
        continue
    else:
        num_frames_bbox = len(Total_video_BB[id])
        vid = cv2.VideoCapture(path)
        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count != num_frames_bbox:
            continue

    label_name = i['template'].lower()
    label_name = label_name.replace('[something]','something')
    label_name = label_name.replace('[some substance]','some substance')
    label_name = label_name.replace('[something in it]','something in it')
    label_name = label_name.replace('[number of]','number of')
    label_name = label_name.replace('[something soft]','something soft')
    label_name = label_name.replace('[something unbendable]','something unbendable')
    label_name = label_name.replace('[something that cannot actually stand upright]','something that cannot actually stand upright')
    label_name = label_name.replace('[something similar to other things that are already on the table]','something similar to other things that are already on the table')
    label_name = label_name.replace('[one of many similar things on the table]','one of many similar things on the table')
    label_name = label_name.replace('[something else that cannot support it]','something else that cannot support it')
    label_name = label_name.replace('[something that is not tearable]','something that is not tearable')
    label_name = label_name.replace('[somewhere]','somewhere')
    label_name = label_name.replace('[part]','part')
    label_num = labels[label_name.capitalize()]
    train_df['path'].append(path)
    train_df['label_name'].append(label_name)
    train_df['label_num'].append(label_num)

f = open(os.path.join(root_add, 'labels','validation.json'))
val_label = json.load(f)
for i in val_label:
    id = i['id']
    path = os.path.join(video_mp4_root_add, id+'.mp4')

    # check if the mp4 file exists
    if not os.path.exists(path):
        continue

    # check if bbox information exists in Total_video_BB
    if not id in Total_video_BB:
        continue
    else:
        num_frames_bbox = len(Total_video_BB[id])
        vid = cv2.VideoCapture(path)
        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count != num_frames_bbox:
            continue

    label_name = i['template'].lower()
    label_name = label_name.replace('[something]','something')
    label_name = label_name.replace('[some substance]','some substance')
    label_name = label_name.replace('[something in it]','something in it')
    label_name = label_name.replace('[number of]','number of')
    label_name = label_name.replace('[something soft]','something soft')
    label_name = label_name.replace('[something unbendable]','something unbendable')
    label_name = label_name.replace('[something that cannot actually stand upright]','something that cannot actually stand upright')
    label_name = label_name.replace('[something similar to other things that are already on the table]','something similar to other things that are already on the table')
    label_name = label_name.replace('[one of many similar things on the table]','one of many similar things on the table')
    label_name = label_name.replace('[something else that cannot support it]','something else that cannot support it')
    label_name = label_name.replace('[something that is not tearable]','something that is not tearable')
    label_name = label_name.replace('[somewhere]','somewhere')
    label_name = label_name.replace('[part]','part')
    label_num = labels[label_name.capitalize()]
    val_df['path'].append(path)
    val_df['label_name'].append(label_name)
    val_df['label_num'].append(label_num)

train_df = pd.DataFrame(train_df)
val_df = pd.DataFrame(val_df)


# #  downlload bounding boxes/ check the existing path in BB/ save the correspondence videos/ create a csv file
# url_1 = 'https://drive.google.com/file/d/1OlggVsZt8eLOI33C3GKAp9EHajMqxQ_Y/view?usp=sharing'
# url_2 = 'https://drive.google.com/file/d/10GQ3RINLAwnw7C2c91Lo17TuD2HnoRgr/view?usp=sharing'
# url_3 = 'https://drive.google.com/file/d/1-kebQmdN4lE6NI3CxGeKLKNJqwaT8uRG/view?usp=sharing'
# url_4 = 'https://drive.google.com/file/d/1oVkc4o8LaWZhF7DLDtqAEVYKOcG0dNjQ/view?usp=sharing'


# gdown.download(url_1, output='/home/mona/SSV2_BB/', quiet=False, fuzzy=True) 
# gdown.download(url_2, output='/home/mona/SSV2_BB/', quiet=False, fuzzy=True) 
# gdown.download(url_3, output='/home/mona/SSV2_BB/', quiet=False, fuzzy=True) 
# gdown.download(url_4, output='/home/mona/SSV2_BB/', quiet=False, fuzzy=True) 






# to_csv() 
csv_annotation_root = "/home/mona/VideoMAE/dataset/somethingsomething/annotation"
if not os.path.exists(csv_annotation_root):
    os.makedirs(csv_annotation_root)
train_df.to_csv(path_or_buf=os.path.join(csv_annotation_root, "train_BB.csv"), sep=' ', na_rep='', float_format=None, 
columns=None, header=False, index=False, index_label=None, mode='w', encoding=None, 
compression='infer', quoting=None, quotechar='"', line_terminator=None, 
chunksize=None, date_format=None, doublequote=True, escapechar=None, 
decimal='.', errors='strict', storage_options=None)

val_df.to_csv(path_or_buf=os.path.join(csv_annotation_root, "val_BB.csv"), sep=' ', na_rep='', float_format=None, 
columns=None, header=False, index=False, index_label=None, mode='w', encoding=None, 
compression='infer', quoting=None, quotechar='"', line_terminator=None, 
chunksize=None, date_format=None, doublequote=True, escapechar=None, 
decimal='.', errors='strict', storage_options=None)