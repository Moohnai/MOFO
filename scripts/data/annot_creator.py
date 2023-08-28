import pandas as pd
import json
import os 

# create dataframe
train_df = {'path':[], 'label_name':[], 'label_num':[]}
val_df = {'path':[], 'label_name':[], 'label_num':[]}
test_df = {'path':[], 'label_name':[], 'label_num':[]}

# root addresses
root_add = "path to dataset root"
video_mp4_root_add = "path to dataset videos"

f = open(os.path.join(root_add, 'labels','labels.json'))
labels = json.load(f)


f = open(os.path.join(root_add, 'labels','train.json'))
train_label = json.load(f)
for i in train_label:
    id = i['id']
    path = os.path.join(video_mp4_root_add, id+'.mp4')
    if not os.path.exists(path):
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
    if not os.path.exists(path):
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

test_label = pd.read_csv(os.path.join(root_add, 'labels','test-answers.csv'), sep=';', header=None)
for _, (id, label_name) in test_label.iterrows():
    # id, label_name = i.split(';')
    path = os.path.join(video_mp4_root_add, str(id)+'.mp4')
    if not os.path.exists(path):
        continue
    label_name = label_name.lower()
    label_num = labels[label_name.capitalize()]
    test_df['path'].append(path)
    test_df['label_name'].append(label_name)
    test_df['label_num'].append(label_num)

train_df = pd.DataFrame(train_df)
val_df = pd.DataFrame(val_df)
test_df = pd.DataFrame(test_df)

# to_csv() 
csv_annotation_root = "path to save csv file"
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

test_df.to_csv(path_or_buf=os.path.join(csv_annotation_root, "test.csv"), sep=' ', na_rep='', float_format=None,
columns=None, header=False, index=False, index_label=None, mode='w', encoding=None,
compression='infer', quoting=None, quotechar='"', line_terminator=None,
chunksize=None, date_format=None, doublequote=True, escapechar=None,
decimal='.', errors='strict', storage_options=None)