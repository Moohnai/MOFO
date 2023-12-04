import csv
import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import video_transforms as video_transforms
import video_transforms_BB_focused as video_transforms_BB_focused
import volume_transforms as volume_transforms
import orjson


class EpicVideoClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, classtype, anno_path, data_path, mode='train', clip_len=8,
                crop_size=224, short_side_size=256, new_height=256,
                new_width=340, keep_aspect_ratio=True, num_segment=1,
                num_crop=1, test_num_segment=10, test_num_crop=3, args=None):
        self.classtype = classtype
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        self.data_root = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_videos/"
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        self.samples = []
        with open(self.anno_path) as f:
            csv_reader = csv.reader(f)
            _ = next(csv_reader)  # skip the header
            for i, row in enumerate(csv_reader):
                pid, vid = row[1:3]
                narration = row[8]
                verb, noun = int(row[10]), int(row[12])
                vid_path = 'video_{}.mp4'.format(i)
                m = "validation" if mode == "test" else mode
                vid_path = os.path.join(self.data_root, m, vid_path)
                # start_frame = int(np.round(fps * start_timestamp))
                # end_frame = int(np.ceil(fps * end_timestamp))
                self.samples.append((vid_path, narration, verb, noun))

        # add fps to samples
        for idx, sample in enumerate(self.samples):
            vid_path = sample[0]
            vr = VideoReader(vid_path, num_threads=1, ctx=cpu(0))
            fps = vr.get_avg_fps()
            self.samples[idx] = (vid_path, narration, verb, noun, fps)

        self.label_mapping = args.mapping_vn2act

        if (mode == 'train'):
            pass
        
        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.samples)):
                        verb, noun = self.samples[idx][2], self.samples[idx][3]
                        sample_label = self.label_mapping['{}:{}'.format(verb, noun)]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.samples[idx][0])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args 
            scale_t = 1

            verb, noun = self.samples[index][2], self.samples[index][3]
            sample = self.samples[index][0]
            verb, noun = self.samples[index][2], self.samples[index][3]
            buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t) # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    verb, noun = self.samples[index][2], self.samples[index][3]
                    sample = self.samples[index][0]
                    buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    verb, noun = self.samples[index][2], self.samples[index][3]
                    label = self.label_mapping['{}:{}'.format(verb, noun)]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)
            
            return buffer, self.label_mapping['{}:{}'.format(verb, noun)], index, {}

        elif self.mode == 'validation':
            verb, noun = self.samples[index][2], self.samples[index][3]
            sample = self.samples[index][0]
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    verb, noun = self.samples[index][2], self.samples[index][3]
                    sample = self.samples[index][0]
                    verb, noun = self.samples[index][2], self.samples[index][3]
                    buffer = self.loadvideo_decord(sample)
            buffer = self.data_transform(buffer)
            return buffer, self.label_mapping['{}:{}'.format(verb, noun)], sample.split("/")[-1].split(".")[0]

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample)
    
            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.loadvideo_decord(sample)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                / (self.test_num_crop - 1)
            temporal_start = chunk_nb # 0/1 ////////////////////////////////////////////////
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start::2, \
                       spatial_start:spatial_start + self.short_side_size, :, :]#/////////////////////////////////////////
            else:
                buffer = buffer[temporal_start::2, \
                       :, spatial_start:spatial_start + self.short_side_size, :]#//////////////////////////////////

            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                   chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
        self,
        buffer,
        args,
    ):

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C 
        
        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'Epic-Kitchens' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer


    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        if self.mode == 'test':
            all_index = []#///////////////////////////////////////////
            tick = len(vr) / float(self.num_segment)#//////////////////////////////////
            all_index = list(np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segment)] +
                               [int(tick * x) for x in range(self.num_segment)]))#//////////////////////////////////
            while len(all_index) < (self.num_segment * self.test_num_segment):#//////////////////////////////////
                all_index.append(all_index[-1])
            all_index = list(np.sort(np.array(all_index))) 
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # handle temporal segments????????????????
        average_duration = len(vr) // self.num_segment#???????????
        all_index = []
        if average_duration > 0:
            all_index += list(np.multiply(list(range(self.num_segment)), average_duration) + np.random.randint(average_duration,
                                                                                                        size=self.num_segment))
        elif len(vr) > self.num_segment:
            all_index += list(np.sort(np.random.randint(len(vr), size=self.num_segment)))
        else:
            all_index += list(np.zeros((self.num_segment,)))
        all_index = list(np.array(all_index)) 
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer#???????????????????

    def __len__(self):
        if self.mode != 'test':
            return len(self.samples)
        else:
            return len(self.test_dataset)


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor




#####################################################


class EpicVideoClsDataset_BB_focused(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, classtype, anno_path, data_path, mode='train', clip_len=8,
                crop_size=224, short_side_size=256, new_height=256,
                new_width=340, keep_aspect_ratio=True, num_segment=1,
                num_crop=1, test_num_segment=10, test_num_crop=3, args=None):
        self.classtype = classtype
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        self.data_root = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_videos/"
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        if mode == 'train':
            print(f"Loading {mode} bbox json file...")
            with open('../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/Unsupervised_BB_EPIC_100_train.json', "r", encoding="utf-8") as f:
                Total_video_BB = orjson.loads(f.read())
        elif mode == 'validation':
            print(f"Loading {mode} bbox json file...")
            with open('../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/Unsupervised_BB_EPIC_100_validation.json', "r", encoding="utf-8") as f:
                Total_video_BB = orjson.loads(f.read())
        elif mode == 'test':
            print(f"Loading {mode} bbox json file...")
            with open('../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/Unsupervised_BB_EPIC_100_validation.json', "r", encoding="utf-8") as f:
                Total_video_BB = orjson.loads(f.read())
        self.bb_data = Total_video_BB

        self.samples = []
        with open(self.anno_path) as f:
            csv_reader = csv.reader(f)
            _ = next(csv_reader)  # skip the header
            for i, row in enumerate(csv_reader):
                pid, vid = row[1:3]
                narration = row[8]
                verb, noun = int(row[10]), int(row[12])
                vid_path = 'video_{}.mp4'.format(i)
                m = "validation" if mode == "test" else mode
                vid_path = os.path.join(self.data_root, m, vid_path)
                # start_frame = int(np.round(fps * start_timestamp))
                # end_frame = int(np.ceil(fps * end_timestamp))
                self.samples.append((vid_path, narration, verb, noun))

        # add fps to samples
        for idx, sample in enumerate(self.samples):
            vid_path = sample[0]
            vr = VideoReader(vid_path, num_threads=1, ctx=cpu(0))
            fps = vr.get_avg_fps()
            self.samples[idx] = (vid_path, narration, verb, noun, fps)
                
        self.label_mapping = args.mapping_vn2act

        if (mode == 'train'):
            pass
        
        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize_BB_focused(self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop_BB_focused(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor_BB_focused(),
                video_transforms.Normalize_BB_focused(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize_BB_focused(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor_BB_focused(),
                video_transforms.Normalize_BB_focused(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.samples)):
                        verb, noun = self.samples[idx][2], self.samples[idx][3]
                        sample_label = self.label_mapping['{}:{}'.format(verb, noun)]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.samples[idx][0])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, index):


        if self.mode == 'train':
            args = self.args 
            scale_t = 1

            sample = self.samples[index][0]
            verb, noun = self.samples[index][2], self.samples[index][3]
            buffer, frame_id_list = self.loadvideo_decord(sample, sample_rate_scale=scale_t) # T H W C

            frames_bbox = []
            ### for EPIC-KITCHENS
            video_name = sample
            for idx, c in enumerate(frame_id_list):
                union_frame_bboxs = np.array([[x['box2d']["x1"], x['box2d']["y1"], x['box2d']["x2"], x['box2d']["y2"]] for x in self.bb_data[video_name.split('/')[-1].split('.')[0]][c]['labels']]).reshape(-1) # x1, y1, x2, y2
                frames_bbox.append(union_frame_bboxs)

            frames_bbox = np.array(frames_bbox)

            # # to aviod error in the model, we will replace changing bounding box behavior with the average bounding box but with the same size
            # frames_bbox = np.array([np.mean(frames_bbox, axis=0)]*len(frames_bbox))

        
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    verb, noun = self.samples[index][2], self.samples[index][3]
                    sample = self.samples[index][0]
                    buffer, frame_id_list = self.loadvideo_decord(sample, sample_rate_scale=scale_t) # T H W C

                    frames_bbox = []
                    ### for EPIC-KITCHENS
                    video_name = sample
                    for idx, c in enumerate(frame_id_list):
                        union_frame_bboxs = np.array([[x['box2d']["x1"], x['box2d']["y1"], x['box2d']["x2"], x['box2d']["y2"]] for x in self.bb_data[video_name.split('/')[-1].split('.')[0]][c]['labels']]).reshape(-1) # x1, y1, x2, y2
                        frames_bbox.append(union_frame_bboxs)

                    frames_bbox = np.array(frames_bbox)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    verb, noun = self.samples[index][2], self.samples[index][3]
                    label = self.label_mapping['{}:{}'.format(verb, noun)]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer, bbox = self._aug_frame(buffer, frames_bbox, args)
            
            return buffer, bbox, self.label_mapping['{}:{}'.format(verb, noun)], index, {}

        elif self.mode == 'validation':
            verb, noun = self.samples[index][2], self.samples[index][3]
            sample = self.samples[index][0]
            buffer, frame_id_list = self.loadvideo_decord(sample)

            frames_bbox = []
            ### for EPIC-KITCHENS
            video_name = sample
            for idx, c in enumerate(frame_id_list):
                union_frame_bboxs = np.array([[x['box2d']["x1"], x['box2d']["y1"], x['box2d']["x2"], x['box2d']["y2"]] for x in self.bb_data[video_name.split('/')[-1].split('.')[0]][c]['labels']]).reshape(-1) # x1, y1, x2, y2
                frames_bbox.append(union_frame_bboxs)

            frames_bbox = np.array(frames_bbox)

            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    verb, noun = self.samples[index][2], self.samples[index][3]
                    sample = self.samples[index][0]
                    buffer, frame_id_list = self.loadvideo_decord(sample)

                    frames_bbox = []
                    ### for EPIC-KITCHENS
                    video_name = sample
                    for idx, c in enumerate(frame_id_list):
                        union_frame_bboxs = np.array([[x['box2d']["x1"], x['box2d']["y1"], x['box2d']["x2"], x['box2d']["y2"]] for x in self.bb_data[video_name.split('/')[-1].split('.')[0]][c]['labels']]).reshape(-1) # x1, y1, x2, y2
                        frames_bbox.append(union_frame_bboxs)

                    frames_bbox = np.array(frames_bbox)

            (buffer, frames_bbox) = self.data_transform((buffer, frames_bbox))

            # convert to tensor
            frames_bbox = torch.from_numpy(frames_bbox)

            return buffer, frames_bbox, self.label_mapping['{}:{}'.format(verb, noun)], sample.split("/")[-1].split(".")[0]

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer, frame_id_list = self.loadvideo_decord(sample)

            frames_bbox = []
            ### for EPIC-KITCHENS
            video_name = sample
            for idx, c in enumerate(frame_id_list):
                union_frame_bboxs = np.array([[x['box2d']["x1"], x['box2d']["y1"], x['box2d']["x2"], x['box2d']["y2"]] for x in self.bb_data[video_name.split('/')[-1].split('.')[0]][c]['labels']]).reshape(-1) # x1, y1, x2, y2
                frames_bbox.append(union_frame_bboxs)

            frames_bbox = np.array(frames_bbox)
    
            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer, frame_id_list = self.loadvideo_decord(sample)

                frames_bbox = []
                ### for EPIC-KITCHENS
                video_name = sample
                for idx, c in enumerate(frame_id_list):
                    union_frame_bboxs = np.array([[x['box2d']["x1"], x['box2d']["y1"], x['box2d']["x2"], x['box2d']["y2"]] for x in self.bb_data[video_name.split('/')[-1].split('.')[0]][c]['labels']]).reshape(-1) # x1, y1, x2, y2
                    frames_bbox.append(union_frame_bboxs)

                frames_bbox = np.array(frames_bbox)

            (buffer, frames_bbox) = self.data_resize((buffer, frames_bbox))
            frames_bbox = np.array(frames_bbox)

            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                / (self.test_num_crop - 1)
            temporal_start = chunk_nb # 0/1 ////////////////////////////////////////////////
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start::2, \
                       spatial_start:spatial_start + self.short_side_size, :, :]#/////////////////////////////////////////
                frames_bbox = frames_bbox[temporal_start::2]

                #########################################
                x1_crop, y1_crop, x2_crop, y2_crop = spatial_start, 0, spatial_start + self.short_side_size, 224 

                # now apply the same crop to bbox
                bbox_crop = []
                for bb in frames_bbox:

                    # bbox is in format (x1, y1, x2, y2)
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]

                    # check if the bbox and crop intersect
                    if x1_crop > x2 or x2_crop < x1 or y1_crop > y2 or y2_crop < y1:
                        # no intersection, return the coordinates of the crop
                        x1, y1, x2, y2 = x1_crop, y1_crop, x2_crop, y2_crop
                    else:
                        x1 = max(x1, x1_crop)
                        y1 = max(y1, y1_crop)
                        x2 = min(x2, x2_crop)
                        y2 = min(y2, y2_crop)

                    bbox_crop.append([x1, y1, x2, y2])

                # bbox_crop = torch.tensor(np.array(bbox_crop))
                bbox_crop = np.array(bbox_crop)
                #########################################
            else:
                buffer = buffer[temporal_start::2, \
                       :, spatial_start:spatial_start + self.short_side_size, :]#//////////////////////////////////
                frames_bbox = frames_bbox[temporal_start::2]

                #########################################
                x1_crop, y1_crop, x2_crop, y2_crop = 0, spatial_start, 224, spatial_start + self.short_side_size

                # now apply the same crop to bbox
                bbox_crop = []
                for bb in frames_bbox:

                    # bbox is in format (x1, y1, x2, y2)
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]

                    # check if the bbox and crop intersect
                    if x1_crop > x2 or x2_crop < x1 or y1_crop > y2 or y2_crop < y1:
                        # no intersection, return the coordinates of the crop
                        x1, y1, x2, y2 = x1_crop, y1_crop, x2_crop, y2_crop
                    else:
                        x1 = max(x1, x1_crop)
                        y1 = max(y1, y1_crop)
                        x2 = min(x2, x2_crop)
                        y2 = min(y2, y2_crop)

                    bbox_crop.append([x1, y1, x2, y2])

                # bbox_crop = torch.tensor(np.array(bbox_crop))
                bbox_crop = np.array(bbox_crop)
                #########################################

            (buffer, frames_bbox) = self.data_transform((buffer, frames_bbox))

            # convert to tensor
            frames_bbox = torch.from_numpy(frames_bbox)

            return buffer, frames_bbox, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                   chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
        self,
        buffer,
        bbox,
        args,
    ):

        aug_transform = video_transforms_BB_focused.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        # transform tensors to list of bounding boxes
        bbox = [bbx.tolist() for bbx in bbox]

        # buffer, bbox = aug_transform(buffer, bbox)
        buffer, bbox = aug_transform((buffer, bbox))

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C 
        
        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer, bbox = spatial_sampling_BB_focused(
            buffer,
            bbox,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'Epic-Kitchens' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer, bbox


    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        if self.mode == 'test':
            all_index = []#///////////////////////////////////////////
            tick = len(vr) / float(self.num_segment)#//////////////////////////////////
            all_index = list(np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segment)] +
                               [int(tick * x) for x in range(self.num_segment)]))#//////////////////////////////////
            while len(all_index) < (self.num_segment * self.test_num_segment):#//////////////////////////////////
                all_index.append(all_index[-1])
            all_index = list(np.sort(np.array(all_index))) 
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer, all_index

        # handle temporal segments????????????????
        average_duration = len(vr) // self.num_segment#???????????
        all_index = []
        if average_duration > 0:
            all_index += list(np.multiply(list(range(self.num_segment)), average_duration) + np.random.randint(average_duration,
                                                                                                        size=self.num_segment))
        elif len(vr) > self.num_segment:
            all_index += list(np.sort(np.random.randint(len(vr), size=self.num_segment)))
        else:
            all_index += list(np.zeros((self.num_segment,)).astype(int))
        all_index = list(np.array(all_index)) 
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer, all_index

    def __len__(self):
        if self.mode != 'test':
            return len(self.samples)
        else:
            return len(self.test_dataset)

def spatial_sampling_BB_focused(
    frames,
    bbox,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms_BB_focused.random_resized_crop
            )
            frames, bbox = transform_func(
                images=frames,
                bbox=bbox,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames, bbox




############################################################
import decord
import os.path as osp
import glob
import torchvision
from torchvision import tv_tensors
import torchvision.transforms._transforms_video as transforms_video
import torch.nn as nn
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F, Transform


class Permute_BB(nn.Module):
    """
    Permutation as an op
    """

    def __init__(self, ordering):
        super().__init__()
        self.ordering = ordering

    def forward(self, frames, bbox=None):
        """
        Args:
            frames in some ordering, by default (C, T, H, W)
        Returns:
            frames in the ordering that was specified
        """
        return frames.permute(self.ordering), bbox
    
class Permute(nn.Module):
    """
    Permutation as an op
    """

    def __init__(self, ordering):
        super().__init__()
        self.ordering = ordering

    def forward(self, frames):
        """
        Args:
            frames in some ordering, by default (C, T, H, W)
        Returns:
            frames in the ordering that was specified
        """
        return frames.permute(self.ordering)

def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    frame_ids = np.convolve(np.linspace(start_frame, end_frame, num_segments + 1), [0.5, 0.5], mode='valid')
    if jitter:
        seg_size = float(end_frame - start_frame - 1) / num_segments
        shift = (np.random.rand(num_segments) - 0.5) * seg_size
        frame_ids += shift
    return frame_ids.astype(int).tolist()

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

def get_video_reader(videoname, num_threads, fast_rrc, rrc_params, fast_rcc, rcc_params):
    video_reader = None
    if fast_rrc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rrc_params[0], height=rrc_params[0],
            use_rrc=True, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
        )
    elif fast_rcc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rcc_params[0], height=rcc_params[0],
            use_rcc=True,
        )
    else:
        video_reader = decord.VideoReader(videoname, num_threads=num_threads)
    return video_reader

def video_loader(root, vid, ext, second, end_second,
                 chunk_len=300, fps=30, clip_length=32,
                 threads=1,
                 fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False, rcc_params=(224, ),
                 jitter=False):
    assert fps > 0, 'fps should be greater than 0'

    vr = get_video_reader(
        osp.join(root, '{}'.format(vid)),
        num_threads=threads,
        fast_rrc=fast_rrc, rrc_params=rrc_params,
        fast_rcc=fast_rcc, rcc_params=rcc_params,
    )
    end_second = min(end_second, len(vr) / fps)

    # calculate frame_ids
    frame_offset = int(np.round(second * fps))
    total_duration = max(int((end_second - second) * fps), clip_length)
    frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)

    # load frames
    assert max(frame_ids) < len(vr)
    try:
        frames = vr.get_batch(frame_ids).asnumpy()
    except decord.DECORDError as error:
        print(error)
        frames = vr.get_batch([0] * len(frame_ids)).asnumpy()

    return torch.from_numpy(frames.astype(np.float32)), frame_ids

class VideoCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, data_root, anno_path, mode, metadata=None, is_trimmed=True):
        self.dataset = dataset
        self.data_root = data_root
        self.metadata = metadata
        self.is_trimmed = is_trimmed
        self.mode = mode
        self.anno_path = anno_path
        m = "validation" if mode == "test" else mode
        self.data_root = os.path.join(self.data_root, m)
        
        self.samples = []
        with open(self.anno_path) as f:
            csv_reader = csv.reader(f)
            _ = next(csv_reader)  # skip the header
            for i, row in enumerate(csv_reader):
                pid, vid = row[1:3]
                narration = row[8]
                start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
                verb, noun = int(row[10]), int(row[12])
                vid_path = 'video_{}.mp4'.format(i)
                
                if len(row) == 16:
                    fps = float(row[-1])
                else:
                    vid_path = os.path.join(self.data_root, vid_path)
                    vr = VideoReader(vid_path, num_threads=1, ctx=cpu(0))
                    fps = vr.get_avg_fps()
                # start_frame = int(np.round(fps * start_timestamp))
                # end_frame = int(np.ceil(fps * end_timestamp))
                self.samples.append((vid_path, start_timestamp, end_timestamp, fps, narration, verb, noun))

        # add fps to the annotation file
        if len(row) == 15:
            df = pd.read_csv(self.anno_path)
            df['fps'] = [self.samples[i][3] for i in range(len(self.samples))]
            df.to_csv(self.anno_path, index=False)

    def get_raw_item(
        self, i, is_training=True, num_clips=1,
        chunk_len=300, clip_length=32, clip_stride=2,
        sparse_sample=False,
        narration_selection='random',
        threads=1,
        fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False, rcc_params=(224,),
    ):

        vid_path, start_second, end_second, fps, narration, verb, noun = self.samples[i]
        end_second = end_second - start_second
        start_second = 0
        frames, frame_ids = video_loader(self.data_root, vid_path, 'MP4',
                                start_second, end_second,
                                chunk_len=chunk_len, fps=fps,
                                clip_length=clip_length,
                                threads=threads,
                                fast_rrc=fast_rrc,
                                rrc_params=rrc_params,
                                fast_rcc=fast_rcc,
                                rcc_params=rcc_params,
                                jitter=is_training)
        return frames, '{}:{}'.format(verb, noun), frame_ids

    def __len__(self):
        return len(self.samples)

class VideoClassyDataset(VideoCaptionDatasetBase):
    def __init__(
        self, dataset, anno_path, metadata=None, transform=None,
        is_training=True,
        num_clips=1,
        chunk_len=-1,
        clip_length=32, clip_stride=2,
        threads=1,
        fast_rrc=False,
        rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False,
        rcc_params=(224,),
        sparse_sample=False,
        is_trimmed=True,
        mode='train',
        args=None,
        root="/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_videos/",
        ):
        super().__init__(dataset, root, anno_path, mode, metadata, is_trimmed=is_trimmed)

        self.transform = transform
        self.is_training = True if mode == 'train' else False
        self.num_clips = num_clips
        self.chunk_len = chunk_len
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_rcc = fast_rcc
        self.rcc_params = rcc_params
        self.sparse_sample = sparse_sample
        self.mode = mode
        self.label_mapping = args.mapping_vn2act

        mean, std = [0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]

        if self.mode == "train":
            base_train_transform_ls = [
                Permute([3, 0, 1, 2]),
                torchvision.transforms.RandomResizedCrop(rcc_params[0], scale=(0.5, 1.0)),
                transforms_video.NormalizeVideo(mean=mean, std=std),
            ]
            self.transform = torchvision.transforms.Compose(base_train_transform_ls)
        else:
            base_val_transform_ls = [
                Permute([3, 0, 1, 2]),
                torchvision.transforms.Resize(rcc_params[0]),
                torchvision.transforms.CenterCrop(rcc_params[0]),
                transforms_video.NormalizeVideo(mean=mean, std=std),
            ]
            self.transform = torchvision.transforms.Compose(base_val_transform_ls)

    def __getitem__(self, i):
        frames, label, _ = self.get_raw_item(
            i, is_training=self.is_training,
            chunk_len=self.chunk_len,
            num_clips=self.num_clips,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            threads=self.threads,
            fast_rrc=self.fast_rrc,
            rrc_params=self.rrc_params,
            fast_rcc=self.fast_rcc,
            rcc_params=self.rcc_params,
            sparse_sample=self.sparse_sample,
        )

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)

        if self.label_mapping is not None:
            if isinstance(label, list):
                # multi-label case
                res_array = np.zeros(len(self.label_mapping))
                for lbl in label:
                    res_array[self.label_mapping[lbl]] = 1.
                label = res_array
            else:
                label = self.label_mapping[label]

        return frames, label, self.samples[i][0].split("/")[-1].split(".")[0], {}


class VideoClassyDataset_BB(VideoCaptionDatasetBase):
    def __init__(
        self, dataset, anno_path, metadata=None, transform=None,
        is_training=True,
        num_clips=1,
        chunk_len=-1,
        clip_length=32, clip_stride=2,
        threads=1,
        fast_rrc=False,
        rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False,
        rcc_params=(224,),
        sparse_sample=False,
        is_trimmed=True,
        mode='train',
        args=None,
        root="/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_videos/",
        ):
        super().__init__(dataset, root, anno_path, mode, metadata, is_trimmed=is_trimmed)

        self.transform = transform
        self.is_training = True if mode == 'train' else False
        self.num_clips = num_clips
        self.chunk_len = chunk_len
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_rcc = fast_rcc
        self.rcc_params = rcc_params
        self.sparse_sample = sparse_sample
        self.mode = mode
        self.label_mapping = args.mapping_vn2act

        mean, std = [0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]

        if self.mode == "train":
            base_train_transform_ls = [
                Permute_BB([0, 3, 1, 2]),
                v2.RandomResizedCrop(rcc_params[0], scale=(0.5, 1.0)),
                v2.Normalize(mean, std),
                Permute_BB([1, 0, 2, 3]),
            ]
            self.transform = v2.Compose(base_train_transform_ls)
        else:
            base_val_transform_ls = [
                Permute_BB([0, 3, 1, 2]),
                v2.Resize(rcc_params[0]),
                v2.CenterCrop(rcc_params[0]),
                v2.Normalize(mean, std),
                Permute_BB([1, 0, 2, 3]),
            ]
            self.transform = v2.Compose(base_val_transform_ls)

        if mode == 'train':
            print(f"Loading {mode} bbox json file...")
            with open('../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/Unsupervised_BB_EPIC_100_train.json', "r", encoding="utf-8") as f:
                Total_video_BB = orjson.loads(f.read())
        elif mode == 'validation':
            print(f"Loading {mode} bbox json file...")
            with open('../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/Unsupervised_BB_EPIC_100_validation.json', "r", encoding="utf-8") as f:
                Total_video_BB = orjson.loads(f.read())
        elif mode == 'test':
            print(f"Loading {mode} bbox json file...")
            with open('../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/Unsupervised_BB_EPIC_100_validation.json', "r", encoding="utf-8") as f:
                Total_video_BB = orjson.loads(f.read())
        self.bb_data = Total_video_BB

    def __getitem__(self, i):
        frames, label, frame_ids = self.get_raw_item(
            i, is_training=self.is_training,
            chunk_len=self.chunk_len,
            num_clips=self.num_clips,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            threads=self.threads,
            fast_rrc=self.fast_rrc,
            rrc_params=self.rrc_params,
            fast_rcc=self.fast_rcc,
            rcc_params=self.rcc_params,
            sparse_sample=self.sparse_sample,
        )

        frames_bbox = []
        ### for EPIC-KITCHENS
        video_name = self.samples[i][0]
        for idx, c in enumerate(frame_ids):
            union_frame_bboxs = np.array([[x['box2d']["x1"], x['box2d']["y1"], x['box2d']["x2"], x['box2d']["y2"]] for x in self.bb_data[video_name.split('/')[-1].split('.')[0]][c]['labels']]).reshape(-1) # x1, y1, x2, y2
            frames_bbox.append(union_frame_bboxs)

        frames_bbox = np.array(frames_bbox)
        union_frame_bb = np.array([np.mean(frames_bbox, axis=0)])
        union_frame_bb = tv_tensors.BoundingBoxes(union_frame_bb, format="XYXY", canvas_size=(frames.shape[1], frames.shape[2]))

        # apply transformation
        if self.transform is not None:
            frames, bbox = self.transform(frames, union_frame_bb)

        # repeat bbox in the 1st dimension to match the number of frames
        bbox = np.repeat(bbox, frames.shape[1], axis=0).to(torch.int32)

        if self.label_mapping is not None:
            if isinstance(label, list):
                # multi-label case
                res_array = np.zeros(len(self.label_mapping))
                for lbl in label:
                    res_array[self.label_mapping[lbl]] = 1.
                label = res_array
            else:
                label = self.label_mapping[label]

        return frames, bbox, label, self.samples[i][0].split("/")[-1].split(".")[0], {}

