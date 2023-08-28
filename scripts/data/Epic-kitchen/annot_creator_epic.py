import pandas as pd
import csv
import os 
import json
import itertools


# root addresses

root_add_train = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_train.csv"
root_add_val = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_validation.csv" 
# video_mp4_root_add_train = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_videos/train"
# video_mp4_root_add_val = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_videos/validation"
# cropped_video_mp4_root_add_train = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_cropped_BBframe_videos/train"
# cropped_video_mp4_root_add_val = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_cropped_BBframe_videos/validation"
random_fixed_cropped_video_mp4_root_add_train = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_random_fixed_BBframe_cropped_videos/train"
random_fixed_cropped_video_mp4_root_add_val = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_random_fixed_BBframe_cropped_videos/validation"





# create dataframe
train_df = {'path':[], 'label_name':[], 'label_num':[]}
val_df = {'path':[], 'label_name':[], 'label_num':[]}


# ###### crete json file for labels
# ###verb
# train_df = pd.read_csv(root_add_train)
# num_uniq_verb_labels = len(train_df['verb_class'].unique())
# verb_dict = {label:[] for label in range(num_uniq_verb_labels)}
# for i, item in train_df.iterrows():
#     if not any(item['verb'] in verb for verb in verb_dict[item['verb_class']]):
#         verb_dict[item['verb_class']].append(item['verb'])
# #find verbs with same class


# verb_dict = dict(sorted(verb_dict.items(), key=lambda item: item[0]))
# verb_dict = {str(k): v for k, v in verb_dict.items()}

# root_add = "/home/mona/VideoMAE/dataset/Epic_kitchen/annotation/verb"

# with open(os.path.join(root_add, 'labels','labels.json'), 'w') as fp:
#     json.dump(verb_dict, fp, indent=4)


class_list = [2,3,4,7,8,9,10,14,21,26,32,47,55,63,77]

class_dic = {2:0, 3:1, 4:2, 7:3, 8:4, 9:5, 10:6, 14:7, 21:8, 26:9, 32:10, 47:11, 55:12, 63:13, 77:14}
#high accuracy classes: 2,3,4,7,10

train_label = pd.read_csv(root_add_train)
for i, item in train_label.iterrows():
    if item['verb_class'] not in class_list:
        continue
    else:
        path = os.path.join(random_fixed_cropped_video_mp4_root_add_train, f"video_{i}.mp4")
        if not os.path.exists(path):
            continue
        label_name = item ['verb']
        label_num = item ['verb_class']
        # label_name = item ['noun']
        # label_num = item ['noun_class']
        train_df['path'].append(path)
        train_df['label_name'].append(label_name)
        train_df['label_num'].append(class_dic[label_num])
        

val_label = pd.read_csv(root_add_val)
for i, item in val_label.iterrows():
    if item['verb_class'] not in class_list:
        continue
    else:
        path = os.path.join(random_fixed_cropped_video_mp4_root_add_val, f"video_{i}.mp4")
        if not os.path.exists(path):
            continue
        label_name = item ['verb']
        label_num = item ['verb_class']
        # label_name = item ['noun']
        # label_num = item ['noun_class']
        val_df['path'].append(path)
        val_df['label_name'].append(label_name)
        val_df['label_num'].append(class_dic[label_num])


train_df = pd.DataFrame(train_df)
val_df = pd.DataFrame(val_df)


# to_csv() 
csv_annotation_root = "/home/mona/VideoMAE/dataset/Epic_kitchen/annotation/verb/randomcropped15class"
if not os.path.exists(csv_annotation_root):
    os.makedirs(csv_annotation_root)
train_df.to_csv(path_or_buf=os.path.join(csv_annotation_root, "train.csv"), sep=' ', na_rep='', float_format=None, 
columns=None, header=False, index=False, index_label=None, mode='w', encoding=None, 
compression='infer', quoting=None, quotechar='"', line_terminator=None, 
chunksize=None, date_format=None, doublequote=True, escapechar=None, 
decimal='.', errors='strict', storage_options=None)

val_df.to_csv(path_or_buf=os.path.join(csv_annotation_root, "val.csv"), sep=' ', na_rep='', float_format=None, 
columns=None, header=False, index=False, index_label=None, mode='w', encoding=None, 
compression='infer', quoting=None, quotechar='"', line_terminator=None, 
chunksize=None, date_format=None, doublequote=True, escapechar=None, 
decimal='.', errors='strict', storage_options=None)

# number of train and val samples
print(f"number of train samples: {len(train_df)}")
print(f"number of val samples: {len(val_df)}")

# number of train and val samples per class
print(f"number of train samples per class: {train_df['label_num'].value_counts()}")
print(f"number of val samples per class: {val_df['label_num'].value_counts()}")


# # #######
# number of train samples: 23751
# number of val samples: 3681
# number of train samples per class:
# 2     6927
# 3     4870
# 4     3483
# 8     1861
# 7     1742
# 9     1595
# 10    1574
# 14     737
# 21     346
# 26     232
# 32     147
# 47      87
# 55      73
# 63      52
# 77      25

# number of val samples per class: 
# 2     1141
# 3      810
# 4      514
# 7      292
# 10     287
# 9      242
# 8      211
# 14      97
# 26      31
# 21      21
# 47      20
# 32       7
# 63       5
# 55       3
