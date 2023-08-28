import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import wandb
import matplotlib.pyplot as plt
import os
import textwrap
import cv2
import numpy as np

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                ###########
                tube_mean = videos_squeeze.mean(dim=-2, keepdim=True)
                tube_std = videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
                ###########
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs = model(videos, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()

        #######################################################
               #######################################################
        #visualize the reconstruction
        out = videos_patch.clone()
        out[bool_masked_pos] = outputs.clone().float().reshape(-1, C)
        out = rearrange(out, 'b n (p c) -> b n p c', c = 3)
        out_denorm = out.reshape(B,1568,512,3) * tube_std + tube_mean
        reconstructed_out= rearrange(out_denorm, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size, p2=patch_size,
            t=8, h=14, w=14)
            
        masked_videos = torch.ones_like(videos_patch)
        masked_videos[bool_masked_pos] = torch.zeros_like(outputs.clone().float().reshape(-1, C))
        masked_videos = rearrange(masked_videos, 'b n (p c) -> b n p c', c = 3)
        masked_videos_denorm = masked_videos.reshape(B,1568,512,3) 
        masked_videos_out= rearrange(masked_videos_denorm, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size, p2=patch_size,
            t=8, h=14, w=14)
        masked_videos_out = masked_videos_out * unnorm_videos
        


        # save the video of the reconstruction and the original video
                    
        # for i in range (unnorm_videos.shape[0]):
        #     out_vis = cv2.VideoWriter(filename=f'./input_video_{i}_original.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=4, frameSize=(224, 224), isColor=True)
        #     for j in range (unnorm_videos.shape[2]):
        #         frame = unnorm_videos[i, :, j, :, :].detach().cpu().numpy()
        #         frame = frame.transpose(1, 2, 0)
        #         # clip values to [0, 1]
        #         frame = np.clip(frame, 0, 1)
        #         frame = (frame * 255).astype(np.uint8)
        #         out_vis.write(np.array(frame)[:,:,::-1])
        #     out_vis.release()
        
                    
        for i in range (reconstructed_out.shape[0]):
            out_vis = cv2.VideoWriter(filename=f'./reconstructed_video_{i}.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=4, frameSize=(224, 224), isColor=True)
            for j in range (reconstructed_out.shape[2]):
                frame = reconstructed_out[i, :, j, :, :].detach().cpu().numpy()
                frame = frame.transpose(1, 2, 0)
                # clip values to [0, 1]
                frame = np.clip(frame, 0, 1)
                frame = (frame * 255).astype(np.uint8)
            #     out_vis.write(np.array(frame)[:,:,::-1].astype(np.uint8))
            # out_vis.release()
 



                img_1 = unnorm_videos[i, :, j, :, :].detach().cpu().numpy()
                # convert channel first to channel last
                img_1 = img_1.transpose(1, 2, 0)
                img_2 = reconstructed_out[i, :, j, :, :].detach().cpu().numpy()
                img_2 = img_2.transpose(1, 2, 0)
                img_3 = masked_videos_out[i, :, j, :, :].detach().cpu().numpy()
                img_3 = img_3.transpose(1, 2, 0)

                # save the reconstructed image
                model_name = model.__class__.__name__
                save_path = f'./reconstruction/VideoMAE/video_{i}'#os.path.join('VideoMAE', 'reconstruction')
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)

                # fig, ax = plt.subplots(1, 2, figsize=(10, 8))
                # # set title
                # fig.suptitle(
                #     "\n".join(
                #         textwrap.wrap(
                #             "Reconstruction of the video, Model: " + model_name, 60
                #         )
                #     )
                # )
                # plt.subplot(1, 2, 1)
                # plt.title("Original video")
                # plt.imshow(img_1)
                # plt.subplot(1, 2, 2)
                # plt.title("Reconstructed video")
                # plt.imshow(img_2)
                # plt.savefig(os.path.join(save_path, model_name + f'_both_{i}-{j}.png'))
                # plt.close()

                # plot each image separately
                plt.imshow(img_1)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, model_name + f'_original{i}-{j}.png'), transparent = True, bbox_inches = 'tight', pad_inches = 0)
                plt.close()
                plt.imshow(img_2)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, model_name + f'_reconstructed{i}-{j}.png'), transparent = True, bbox_inches = 'tight', pad_inches = 0)
                plt.close()
                plt.imshow(img_3)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, model_name + f'_masked{i}-{j}.png'), transparent = True, bbox_inches = 'tight', pad_inches = 0)
                plt.close()
        ########################################################
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_BB(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, loss_weight= None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss(reduction='none')

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bbox, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        #create mask on video based on the bbox
        video_masks= torch.zeros_like(videos)
        for v in range(videos.shape[0]):
            video_bbox_region = bbox[v]
            for frame_index in range(videos.shape[1]):
                video_masks[v][:,frame_index, int(video_bbox_region[frame_index,1]):int(video_bbox_region[frame_index,3]), int(video_bbox_region[frame_index,0]):int(video_bbox_region[frame_index,2])] = 1 # y , x
        
        mask_for_input = video_masks.clone()  
        # import cv2
        # import numpy as np
        # frame = videos[1][:,-1].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f'{i}.png', frame)

                

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]



            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                ###########
                tube_mean = videos_squeeze.mean(dim=-2, keepdim=True)
                tube_std = videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
                ###########
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
                # update video masks 
                video_masks = rearrange(video_masks, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                video_masks = rearrange(video_masks, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)
                # update video masks 
                video_masks = rearrange(video_masks, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)
            # create label mask for applying bbox
            mask_labels = video_masks[bool_masked_pos].reshape(B, -1, C)

            
            ### mask the input video (put 0 for the pixels outside of the BB)
            # videos = videos * mask_for_input

            # # find zero elements   in labels
            # labels_mask_loc = torch.where(mask_labels==0)
            # labels_mask = torch.ones_like(labels)
            # labels_mask[labels_mask_loc[0], labels_mask_loc[1], labels_mask_loc[2]] = loss_weight

        with torch.cuda.amp.autocast():
            outputs = model(videos, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)
            # apply label mask to loss and average
            # loss = loss * labels_mask
            loss = loss.mean()

        loss_value = loss.item()

    
        # #######################################################
        # #visualize the reconstruction
        # out = videos_patch.clone()
        # out[bool_masked_pos] = outputs.clone().float().reshape(-1, C)
        # out = rearrange(out, 'b n (p c) -> b n p c', c = 3)
        # out_denorm = out.reshape(B,1568,512,3) * tube_std + tube_mean
        # reconstructed_out= rearrange(out_denorm, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size, p2=patch_size,
        #     t=8, h=14, w=14)
            
        # masked_videos = torch.ones_like(videos_patch)
        # masked_videos[bool_masked_pos] = torch.zeros_like(outputs.clone().float().reshape(-1, C))
        # masked_videos = rearrange(masked_videos, 'b n (p c) -> b n p c', c = 3)
        # masked_videos_denorm = masked_videos.reshape(B,1568,512,3) 
        # masked_videos_out= rearrange(masked_videos_denorm, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size, p2=patch_size,
        #     t=8, h=14, w=14)
        # masked_videos_out = masked_videos_out * unnorm_videos
        


        # # save the video of the reconstruction and the original video
                    
        # # for i in range (unnorm_videos.shape[0]):
        # #     out_vis = cv2.VideoWriter(filename=f'./input_video_{i}_original.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=4, frameSize=(224, 224), isColor=True)
        # #     for j in range (unnorm_videos.shape[2]):
        # #         frame = unnorm_videos[i, :, j, :, :].detach().cpu().numpy()
        # #         frame = frame.transpose(1, 2, 0)
        # #         # clip values to [0, 1]
        # #         frame = np.clip(frame, 0, 1)
        # #         frame = (frame * 255).astype(np.uint8)
        # #         out_vis.write(np.array(frame)[:,:,::-1])
        # #     out_vis.release()
        
                    
        # for i in range (reconstructed_out.shape[0]):
        #     out_vis = cv2.VideoWriter(filename=f'./reconstructed_video_{i}.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=4, frameSize=(224, 224), isColor=True)
        #     for j in range (reconstructed_out.shape[2]):
        #         frame = reconstructed_out[i, :, j, :, :].detach().cpu().numpy()
        #         frame = frame.transpose(1, 2, 0)
        #         # clip values to [0, 1]
        #         frame = np.clip(frame, 0, 1)
        #         frame = (frame * 255).astype(np.uint8)
        #     #     out_vis.write(np.array(frame)[:,:,::-1].astype(np.uint8))
        #     # out_vis.release()
 
        #         # extract bbox
        #         bb = bbox[i, 0, :].detach().cpu().numpy()


        #         img_1 = unnorm_videos[i, :, j, :, :].detach().cpu().numpy()
        #         # convert channel first to channel last
        #         img_1 = img_1.transpose(1, 2, 0)
        #         img_2 = reconstructed_out[i, :, j, :, :].detach().cpu().numpy()
        #         img_2 = img_2.transpose(1, 2, 0)
        #         img_3 = masked_videos_out[i, :, j, :, :].detach().cpu().numpy()
        #         img_3 = (img_3.transpose((1, 2, 0))*255).astype(np.uint8).copy()
        #         # # plot the bbox
        #         # img_3 = cv2.rectangle(
        #         #     img_3,
        #         #     (int(bb[0]), int(bb[1])),
        #         #     (int(bb[2]), int(bb[3])),
        #         #     (0, 255, 0),
        #         #     2,
        #         # )
                
                

        #         # save the reconstructed image
        #         model_name = model.__class__.__name__
        #         save_path = f'./reconstruction/MOFO/video_{i}'#os.path.join('VideoMAE', 'reconstruction')
        #         if not os.path.exists(save_path):
        #             os.makedirs(save_path, exist_ok=True)

        #         # fig, ax = plt.subplots(1, 2, figsize=(10, 8))
        #         # # set title
        #         # fig.suptitle(
        #         #     "\n".join(
        #         #         textwrap.wrap(
        #         #             "Reconstruction of the video, Model: " + model_name, 60
        #         #         )
        #         #     )
        #         # )
        #         # plt.subplot(1, 2, 1)
        #         # plt.title("Original video")
        #         # plt.imshow(img_1)
        #         # plt.subplot(1, 2, 2)
        #         # plt.title("Reconstructed video")
        #         # plt.imshow(img_2)
        #         # plt.savefig(os.path.join(save_path, model_name + f'_both_{i}-{j}.png'))
        #         # plt.close()

        #         # plot each image separately
        #         plt.imshow(img_1)
        #         plt.axis('off')
        #         plt.tight_layout()
        #         plt.savefig(os.path.join(save_path, model_name + f'_original{i}-{j}.png'), transparent = True, bbox_inches = 'tight', pad_inches = 0)
        #         plt.close()
        #         plt.imshow(img_2)
        #         plt.axis('off')
        #         plt.tight_layout()
        #         plt.savefig(os.path.join(save_path, model_name + f'_reconstructed{i}-{j}.png'), transparent = True, bbox_inches = 'tight', pad_inches = 0)
        #         plt.close()
        #         plt.imshow(img_3)
        #         plt.axis('off')
        #         plt.tight_layout()
        #         plt.savefig(os.path.join(save_path, model_name + f'_masked{i}-{j}.png'), transparent = True, bbox_inches = 'tight', pad_inches = 0)
        #         plt.close()

        ########################################################

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        # # log to weights & biases
        # wandb_dict = {}
        # for key, value in metric_logger.meters.items():
        #     wandb_dict["train_iter_"+key] = value.global_avg
        # wandb.log(wandb_dict, step=it)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}






