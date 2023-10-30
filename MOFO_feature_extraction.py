from decord import VideoReader, cpu
import os
import numpy as np
import warnings

import torch
import video_transforms as video_transforms
import volume_transforms as volume_transforms
from timm.models import create_model
import utils
from collections import OrderedDict
import modeling_finetune

model_name='vit_base_patch16_224_feature_ext'
nb_classes=97
num_frames=16
num_segments=1
num_segment=16
tubelet_size=2
drop=0.0
drop_path=0.1
attn_drop_rate=0.0
drop_block_rate=None
use_mean_pooling ='store_true'
init_scale =0.001
keep_aspect_ratio=True
new_height=256
new_width=320
model_prefix=''
test_num_segment=2
model_key='model|module'
crop_size=224
device = 'cpu'


def loadvideo_decord(sample, sample_rate_scale=1):
    """Load video content using Decord"""
    fname = sample

    if not (os.path.exists(fname)):
        return []

    # avoid hanging issue
    if os.path.getsize(fname) < 1 * 1024:
        print('SKIP: ', fname, " - ", os.path.getsize(fname))
        return []
    try:
        if keep_aspect_ratio:
            vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
        else:
            vr = VideoReader(fname, width=new_width, height=new_height,
                                num_threads=1, ctx=cpu(0))
    except:
        print("video cannot be loaded by decord: ", fname)
        return []

    # all_index = []
    # tick = len(vr) / float(num_segment)
    # all_index = list(np.array([int(tick / 2.0 + tick * x) for x in range(num_segment)] +
    #                     [int(tick * x) for x in range(num_segment)]))
    # while len(all_index) < (num_segment * test_num_segment):
    #     all_index.append(all_index[-1])
    # all_index = list(np.sort(np.array(all_index))) 
    # vr.seek(0)
    # buffer = vr.get_batch(all_index).asnumpy()

    average_duration = len(vr) // num_segment#???????????
    all_index = []
    if average_duration > 0:
        all_index += list(np.multiply(list(range(num_segment)), average_duration) + np.random.randint(average_duration,
                                                                                                    size=num_segment))
    elif len(vr) > num_segment:
        all_index += list(np.sort(np.random.randint(len(vr), size=num_segment)))
    else:
        all_index += list(np.zeros((num_segment,)))
    all_index = list(np.array(all_index)) 
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()
    return buffer

def load_data(video_path, test_num_segment = 2, test_num_crop = 3, short_side_size=224):

    chunk_nb, split_nb = test_num_segment, test_num_crop
    buffer = loadvideo_decord(video_path)

    # define the data preprocessing
    data_transform = video_transforms.Compose([
                video_transforms.Resize(short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(crop_size, crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])

    buffer = data_transform(buffer)
    return buffer


 
# data loading
# video_path = "path/to/video"
video_path = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_videos/train/video_10.mp4"
data = load_data(video_path)

# load the model
model = create_model(
    model_name,
    pretrained=False,
    num_classes=nb_classes,
    all_frames=num_frames * num_segments,
    tubelet_size=tubelet_size,
    drop_rate=drop,
    drop_path_rate=drop_path,
    attn_drop_rate=attn_drop_rate,
    drop_block_rate=None,
    use_mean_pooling=use_mean_pooling,
    init_scale=init_scale,
)

# load the checkpoint
checkpoint_path = "/home/mona/MOFO/checkpoint-799.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("Load ckpt from %s" % checkpoint_path)
checkpoint_model = None
for model_key in model_key.split('|'):
    if model_key in checkpoint:
        checkpoint_model = checkpoint[model_key]
        print("Load state_dict by model_key = %s" % model_key)
        break
if checkpoint_model is None:
    checkpoint_model = checkpoint
state_dict = model.state_dict()
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

all_keys = list(checkpoint_model.keys())
new_dict = OrderedDict()
for key in all_keys:
    if key.startswith('backbone.'):
        new_dict[key[9:]] = checkpoint_model[key]
    elif key.startswith('encoder.'):
        new_dict[key[8:]] = checkpoint_model[key]
    else:
        new_dict[key] = checkpoint_model[key]
checkpoint_model = new_dict

utils.load_state_dict(model, checkpoint_model, prefix=model_prefix)

# inference the model
data = data.unsqueeze(0)
data = data.to(device, non_blocking=True)
model = model.to(device)
with torch.cuda.amp.autocast():
    output = model(data)    