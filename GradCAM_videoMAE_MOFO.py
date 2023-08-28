import argparse
import cv2, os
import numpy as np
import torch
import pandas as pd

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from decord import VideoReader, cpu
import orjson


# val_path = '../../home/mona/VideoMAE/dataset/Epic_kitchen/annotation/verb/train.csv'
# vid_path = '../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_videos/train'
# bb_path = '../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/Unsupervised_BB_EPIC_100_train.json'



val_path = '../../home/mona/VideoMAE/dataset/somethingsomething/annotation/train.csv'
vid_path = '../../mnt/welles/scratch/datasets/SSV2/mp4_videos'
bb_path = '../../mnt/welles/scratch/datasets/SSV2/Unsupervised_BB_SSV2_train.json'


# read bb file with orjson
with open(bb_path, 'rb') as f:
    bb_data = orjson.loads(f.read())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./1.jpg',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true', default=False,
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        default=True,
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam++',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_2(tensor, height=27, width=27):
    result = tensor[:, -729:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # result = tensor[:, :729, :].reshape(tensor.size(0),
    #                                   height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def load_VideoMAE(model, pret_path):
    # if args.eval:
    #     args.resume = os.path.join(pret_path, 'checkpoint-best.pth') 
    #     print("Load the best model checkpoint: %s" % args.resume)
    #     tag = 'checkpoint-best'
    # else:
    #     args.resume = os.path.join(pret_path, 'checkpoint-%d' % latest_ckpt)
    #     print("Auto resume checkpoint: %d" % latest_ckpt)
    #     tag = 'checkpoint-%d' % latest_ckpt
    checkpoint = torch.load(pret_path, map_location='cpu')
    model.load_state_dict(checkpoint['module'])

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
            # if self.keep_aspect_ratio:
            if True:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=320, height=256,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        # if self.mode == 'test':
        all_index = []#///////////////////////////////////////////
        tick = len(vr) / float(16)#//////////////////////////////////
        all_index = list(np.array([int(tick / 2.0 + tick * x) for x in range(16)] +
                            [int(tick * x) for x in range(16)]))#//////////////////////////////////
        while len(all_index) < (16 * 2):#//////////////////////////////////
            all_index.append(all_index[-1])
        all_index = list(np.sort(np.array(all_index))) 
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    #####
    # create a folder to save results
    save_dir = os.path.join('CAM_results', "SSV2", args.method)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    from timm.models import create_model
    import modeling_finetune
    
    model_org = create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=174,
        all_frames=16 * 1,
        tubelet_size=2,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=0.001,
    )
    model_BB = create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=174,
        all_frames=16 * 1,
        tubelet_size=2,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=0.001,
    )
    # load weights
    load_VideoMAE(model_BB, './VideoMAE/results/Finetune_all_SSV2_BB_masked_0.75_200SSV2.pt')
    load_VideoMAE(model_org, './VideoMAE/results/original_finetune_SSV2.pt')
    model_BB.eval()
    model_org.eval()


    # read videos from validation set
    videos = pd.read_csv(val_path, header=None, delimiter=' ')
    #####

    if args.use_cuda:
        model_org = model_org.cuda()
        model_BB = model_BB.cuda()
    
    #####
    import video_transforms as video_transforms
    import volume_transforms as volume_transforms

    data_resize = video_transforms.Compose([
        video_transforms.Resize(size=(224), interpolation='bilinear')
    ])
    data_transform = video_transforms.Compose([
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])

    for row in videos.iterrows():
        sample, label_name, label = row[1][0], row[1][1], row[1][2]

        vid_name = sample.split('/')[-1].split('.')[0]

        # read bbox
        bbox = bb_data[vid_name][0]['labels'][0]['box2d']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

        # load video
        chunk_nb, split_nb = (0, 1)
        buffer = loadvideo_decord(sample)

        # plot bbox on buffer
        buffer_md = buffer.copy()
        buffer_md = [cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) for img in buffer_md]
        buffer_md = np.stack(buffer_md, 0)

        buffer = data_resize(buffer)
        buffer_md = data_resize(buffer_md)
        if isinstance(buffer, list):
            buffer = np.stack(buffer, 0)
        if isinstance(buffer_md, list):
            buffer_md = np.stack(buffer_md, 0)

        spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - 224) \
                            / (3 - 1)
        temporal_start = chunk_nb # 0/1 ////////////////////////////////////////////////
        spatial_start = int(split_nb * spatial_step)
        if buffer.shape[1] >= buffer.shape[2]:
            buffer = buffer[temporal_start::2, \
                    spatial_start:spatial_start + 224, :, :]#/////////////////////////////////////////
            buffer_md = buffer_md[temporal_start::2, \
                    spatial_start:spatial_start + 224, :, :]#/////////////////////////////////////////
        else:
            buffer = buffer[temporal_start::2, \
                    :, spatial_start:spatial_start + 224, :]#//////////////////////////////////
            buffer_md = buffer_md[temporal_start::2, \
                    :, spatial_start:spatial_start + 224, :]#//////////////////////////////////

        input_tensor = data_transform(buffer).unsqueeze(0)
        rgb_img_org = buffer_md[7]
        #covnert to float and 0-1 range
        rgb_img = np.float32(buffer[7]) / 255
        #####

        # if the model has the correct prediction, then plot its cam
        with torch.no_grad():
            if args.use_cuda:
                input_tensor = input_tensor.cuda()
            # original model
            output_org = model_org(input_tensor)
            prediction_org = torch.argmax(output_org, dim=1).item()
            # BB model
            output_BB = model_BB(input_tensor)
            prediction_BB = torch.argmax(output_BB, dim=1).item()

        if prediction_BB == label and prediction_org != label:

            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested category.
            targets = None
            # targets = [ClassifierOutputTarget(label)]

            for layer in range(0, 12):
                # layer = 5
                target_layers_org = [
                    # model_org.blocks[0].norm1,
                    # model_org.blocks[1].norm1,
                    # model_org.blocks[2].norm1,
                    # model_org.blocks[3].norm1,
                    model_org.blocks[layer].norm1,
                    # model_org.blocks[5].norm1,
                ]
                target_layers_BB = [
                    # model_BB.blocks[0].norm1,
                    # model_BB.blocks[1].norm1,
                    # model_BB.blocks[2].norm1,
                    # model_BB.blocks[3].norm1,
                    model_BB.blocks[layer].norm1,
                    # model_BB.blocks[5].norm1,
                ]

                if args.method not in methods:
                    raise Exception(f"Method {args.method} not implemented")

                if args.method == "ablationcam":
                    cam_BB = methods[args.method](model=model_BB,
                                            target_layers=target_layers_BB,
                                            use_cuda=args.use_cuda,
                                            reshape_transform=reshape_transform_2,
                                            ablation_layer=AblationLayerVit())
                    
                    cam_org = methods[args.method](model=model_org,
                                            target_layers=target_layers_org,
                                            use_cuda=args.use_cuda,
                                            reshape_transform=reshape_transform_2,
                                            ablation_layer=AblationLayerVit())
                else:
                    cam_BB = methods[args.method](model=model_BB,
                                            target_layers=target_layers_BB,
                                            use_cuda=args.use_cuda,
                                            reshape_transform=reshape_transform_2)
                    
                    cam_org = methods[args.method](model=model_org,
                                            target_layers=target_layers_org,
                                            use_cuda=args.use_cuda,
                                            reshape_transform=reshape_transform_2)

                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam_BB.batch_size = 32
                cam_org.batch_size = 32

                # grayscale_cam_BB = cam_BB(input_tensor=input_tensor,
                #                     targets=targets,
                #                     eigen_smooth=args.eigen_smooth,
                #                     aug_smooth=args.aug_smooth)
                # grayscale_cam_org = cam_org(input_tensor=input_tensor,
                #                     targets=targets,
                #                     eigen_smooth=args.eigen_smooth,
                #                     aug_smooth=args.aug_smooth)

                # # Here grayscale_cam has only one image in the batch
                # grayscale_cam_BB = grayscale_cam_BB[0, :]
                # grayscale_cam_org = grayscale_cam_org[0, :]

                # cam_image_BB = show_cam_on_image(rgb_img, grayscale_cam_BB)
                # cam_image_org = show_cam_on_image(rgb_img, grayscale_cam_org)
                # if args.aug_smooth and args.eigen_smooth:
                #     img_name_org = f'{vid_name}_{args.method}_eigen_aug_cam_{layer}_org.jpg'
                #     img_name_BB = f'{vid_name}_{args.method}_eigen_aug_cam_{layer}_BB.jpg'
                # elif args.aug_smooth:
                #     img_name_org = f'{vid_name}_{args.method}_aug_cam_{layer}_org.jpg'
                #     img_name_BB = f'{vid_name}_{args.method}_aug_cam_{layer}_BB.jpg'
                # elif args.eigen_smooth:
                #     img_name_org = f'{vid_name}_{args.method}_eigen_cam_{layer}_org.jpg'
                #     img_name_BB = f'{vid_name}_{args.method}_eigen_cam_{layer}_BB.jpg'
                # else:
                #     img_name_org = f'{vid_name}_{args.method}_cam_{layer}_org.jpg'
                #     img_name_BB = f'{vid_name}_{args.method}_cam_{layer}_BB.jpg'

                # img_name_org = os.path.join(save_dir, img_name_org)
                # img_name_BB = os.path.join(save_dir, img_name_BB)
                # frame_name = os.path.join(save_dir, f'{vid_name}_{label_name}.jpg')
                # cv2.imwrite(img_name_org, cam_image_org)
                # cv2.imwrite(img_name_BB, cam_image_BB)
                # cv2.imwrite(frame_name, rgb_img_org[:,:,::-1])
                # print(f"Generated {img_name_org} for the input video")

                ## Added
                for eigen in [True, False]:
                    for aug in [True, False]:

                        grayscale_cam_BB = cam_BB(input_tensor=input_tensor,
                                            targets=targets,
                                            eigen_smooth=eigen,
                                            aug_smooth=aug)
                        grayscale_cam_org = cam_org(input_tensor=input_tensor,
                                            targets=targets,
                                            eigen_smooth=eigen,
                                            aug_smooth=aug)

                        # Here grayscale_cam has only one image in the batch
                        grayscale_cam_BB = grayscale_cam_BB[0, :]
                        grayscale_cam_org = grayscale_cam_org[0, :]

                        cam_image_BB = show_cam_on_image(rgb_img, grayscale_cam_BB)
                        cam_image_org = show_cam_on_image(rgb_img, grayscale_cam_org)
                        if aug and eigen:
                            img_name_org = f'{vid_name}_{args.method}_eigen_aug_cam_{layer}_org.jpg'
                            img_name_BB = f'{vid_name}_{args.method}_eigen_aug_cam_{layer}_BB.jpg'
                        elif aug:
                            img_name_org = f'{vid_name}_{args.method}_aug_cam_{layer}_org.jpg'
                            img_name_BB = f'{vid_name}_{args.method}_aug_cam_{layer}_BB.jpg'
                        elif eigen:
                            img_name_org = f'{vid_name}_{args.method}_eigen_cam_{layer}_org.jpg'
                            img_name_BB = f'{vid_name}_{args.method}_eigen_cam_{layer}_BB.jpg'
                        else:
                            img_name_org = f'{vid_name}_{args.method}_cam_{layer}_org.jpg'
                            img_name_BB = f'{vid_name}_{args.method}_cam_{layer}_BB.jpg'

                        img_name_org = os.path.join(save_dir, img_name_org)
                        img_name_BB = os.path.join(save_dir, img_name_BB)
                        frame_name = os.path.join(save_dir, f'{vid_name}_{label_name}.jpg')
                        cv2.imwrite(img_name_org, cam_image_org)
                        cv2.imwrite(img_name_BB, cam_image_BB)
                        cv2.imwrite(frame_name, rgb_img_org[:,:,::-1])
                        print(f"Generated {img_name_org} for the input video")