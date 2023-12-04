import os
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator, TubeMaskingGenerator_BB
from kinetics import VideoClsDataset, VideoMAE, VideoMAE_BB, VideoMAE_BB_no_global_union
from ssv2 import SSVideoClsDataset
from epic_kitchens import EpicVideoClsDataset, EpicVideoClsDataset_BB_focused, VideoClassyDataset, VideoClassyDataset_BB


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

class DataAugmentationForVideoMAE_BB(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop_BB_no_global_union(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            # ToTorchFormatTensor(div=False),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator_BB(
                args.window_size, args.mask_ratio, args.mask_ratio_BB
            )

    def __call__(self, images):
        process_data, process_bbx = self.transform(images)
        return process_data, process_bbx, self.masked_position_generator(process_bbx)

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr





def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
    # dataset = build_dataset(is_train=True, test_mode=False, args=args)
    print("Data Aug = %s" % str(transform))
    return dataset



def build_pretraining_dataset_BB(args):
    transform = DataAugmentationForVideoMAE_BB(args)
    dataset = VideoMAE_BB_no_global_union(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
    # dataset = build_dataset(is_train=True, test_mode=False, args=args)
    print("Data Aug = %s" % str(transform))
    return dataset



def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400
    
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
            # anno_path = args.data_path
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
            
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 
            # anno_path = args.eval_data_path 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = args.nb_classes

    
    elif args.data_set == 'Epic-Kitchens':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'EPIC_100_train.csv')
            num_clips = args.num_segments
            # anno_path = args.data_path
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'EPIC_100_validation.csv') 
            num_clips = args.test_num_segment
            
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'EPIC_100_validation.csv')
            # anno_path = args.eval_data_path 
            num_clips = args.test_num_segment


        # dataset = EpicVideoClsDataset(
        #     classtype=args.classtype,
        #     anno_path=anno_path,
        #     data_path='/',
        #     mode=mode,
        #     clip_len=args.num_frames,
        #     num_segment=args.num_frames,
        #     test_num_segment=args.test_num_segment,
        #     test_num_crop=args.test_num_crop,
        #     num_crop=1 if not test_mode else 3,
        #     keep_aspect_ratio=True,
        #     crop_size=args.input_size,
        #     short_side_size=args.short_side_size,
        #     new_height=256,
        #     new_width=320,
        #     args=args,
        #     )

        dataset = VideoClassyDataset(
            args.data_set, 
            anno_path=anno_path,
            num_clips=num_clips,
            clip_length=args.num_frames, 
            clip_stride=args.sampling_rate,
            threads=1,
            mode=mode,
            args=args,
            )
        
        nb_classes = args.nb_classes        

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_dataset_BB_focused(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400
    
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
            # anno_path = args.data_path
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
            
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 
            # anno_path = args.eval_data_path 

        # dataset = SSVideoClsDataset(
        #     anno_path=anno_path,
        #     data_path='/',
        #     mode=mode,
        #     clip_len=1,
        #     num_segment=args.num_frames,
        #     test_num_segment=args.test_num_segment,
        #     test_num_crop=args.test_num_crop,
        #     num_crop=1 if not test_mode else 3,
        #     keep_aspect_ratio=True,
        #     crop_size=args.input_size,
        #     short_side_size=args.short_side_size,
        #     new_height=256,
        #     new_width=320,
        #     args=args)
        dataset = EpicVideoClsDataset_BB_focused(
        classtype=args.classtype,
        anno_path=anno_path,
        data_path='/',
        mode=mode,
        clip_len=args.num_frames,
        num_segment=args.num_frames,
        test_num_segment=args.test_num_segment,
        test_num_crop=args.test_num_crop,
        num_crop=1 if not test_mode else 3,
        keep_aspect_ratio=True,
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        new_height=256,
        new_width=320,
        args=args,
        )
        nb_classes = args.nb_classes

    
    elif args.data_set == 'Epic-Kitchens':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'EPIC_100_train.csv')
            num_clips = args.num_segments
            # anno_path = args.data_path
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'EPIC_100_validation.csv') 
            num_clips = args.test_num_segment
            
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'EPIC_100_validation.csv')
            # anno_path = args.eval_data_path 
            num_clips = args.test_num_segment


        # dataset = EpicVideoClsDataset_BB_focused(
        #     classtype=args.classtype,
        #     anno_path=anno_path,
        #     data_path='/',
        #     mode=mode,
        #     clip_len=args.num_frames,
        #     num_segment=args.num_frames,
        #     test_num_segment=args.test_num_segment,
        #     test_num_crop=args.test_num_crop,
        #     num_crop=1 if not test_mode else 3,
        #     keep_aspect_ratio=True,
        #     crop_size=args.input_size,
        #     short_side_size=args.short_side_size,
        #     new_height=256,
        #     new_width=320,
        #     args=args,
        #     )
        
        dataset = VideoClassyDataset_BB(
            args.data_set, 
            anno_path=anno_path,
            num_clips=num_clips,
            clip_length=args.num_frames, 
            clip_stride=args.sampling_rate,
            threads=1,
            mode=mode,
            args=args,
            )
        nb_classes = args.nb_classes        

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
