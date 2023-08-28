import torch
import numpy as np
import random
from datasets import build_pretraining_dataset, build_pretraining_dataset_BB, build_pretraining_dataset_BB_no_global_union
import argparse
import utils
from tqdm import tqdm

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_args():
    parser = argparse.ArgumentParser('VideoMAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--epochs', default=800 , type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')

    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')

    parser.add_argument('--mask_ratio', default=0.9, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
                        
    parser.add_argument('--normlize_target', default=True, type=bool,
                        help='normalized the target patch pixels')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=(0.9, 0.95), type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/mona/VideoMAE/dataset/Epic_kitchen/annotation/verb/15class/train.csv', type=str,
                        help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 2)
    parser.add_argument('--output_dir', default='/home/mona/VideoMAE/results/pretrain_BB',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/home/mona/VideoMAE/results/pretrain_BB',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='/home/mona/VideoMAE/results/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # dataset name
    parser.add_argument('--data_set', default='SSV2', type=str, help='Other dataset options is: Kinetics-400, UCF101, HMDB51')
    parser.add_argument('--reprob', default=0.5, type=float, help='Probability that the Random Erasing operation will be performed.')
    parser.add_argument('--remode', default='const', type=str, help="mode: pixel color mode, one of 'const', 'rand', or 'pixel. \
        'const' - erase block is constant color of 0 for all channels \
        'rand'  - erase block is same per-channel random (normal) color \
        'pixel' - erase block is per-pixel random (normal) color")
    parser.add_argument('--recount', default=1, type=int, help='max_count: maximum number of erasing blocks per image, area per box is scaled by count. \
        per-image count is randomly chosen between 1 and this value.')
    parser.add_argument('--test_num_segment', default=10, type=int)
    parser.add_argument('--test_num_crop', default=3, type=int)
    parser.add_argument('--short_side_size', default=256, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


args = get_args()
args.patch_size = (14, 14, 8)
args.window_size = (args.num_frames // 2, args.input_size //args.patch_size[0], args.input_size // args.patch_size[1])
dataset_train = build_pretraining_dataset_BB(args)

num_tasks = utils.get_world_size()
global_rank = utils.get_rank()
sampler_rank = global_rank

sampler_train = torch.utils.data.DistributedSampler(
    dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
)

    
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, sampler=sampler_train,
    batch_size=64,
    num_workers=4,
    pin_memory=False,
    drop_last=True,
    worker_init_fn=seed_worker
)


BB_Percentage=[]
for _, BB, _ in tqdm(data_loader_train, total=len(data_loader_train)):
    BB_batch = torch.tensor([(bb[2]-bb[0])*(bb[3]-bb[1]) for bb in BB])
    BB_Percentage.extend((BB_batch/(224*224))*100)
print(f'average percentage of bounding box coverage' , torch.tensor(BB_Percentage).mean())