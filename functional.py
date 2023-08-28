import numbers
import cv2
import numpy as np
import PIL
import torch


def _is_tensor_clip(clip):
    return torch.is_tensor(clip) and clip.ndimension() == 4


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped

def crop_clip_BB_focused(clip, bbox, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))

    x1_crop, y1_crop, x2_crop, y2_crop = min_w, min_h, min_w + w, min_h + h

    # now apply the same crop to bbox
    bbox_crop = []
    for bb in bbox:

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

    return cropped, bbox_crop


def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[0], size[1]
        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = [
            cv2.resize(img, size, interpolation=np_inter) for img in clip
        ]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.BILINEAR
        else:
            pil_inter = PIL.Image.NEAREST
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled

def resize_clip_BB_focused(clip, bbox, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip, bbox
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
            x_ratio = new_w / im_w
            y_ratio = new_h / im_h
        else:
            size = size[0], size[1]
            x_ratio = size[0] / im_w
            y_ratio = size[1] / im_h
        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = [
            cv2.resize(img, size, interpolation=np_inter) for img in clip
        ]
        scaled_bbox = [[int(bb[0] * x_ratio), int(bb[1] * y_ratio), int(bb[2] * x_ratio), int(bb[3] * y_ratio)] for bb in bbox]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip, bbox
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
            x_ratio = new_w / im_w
            y_ratio = new_h / im_h
        else:
            size = size[1], size[0]
            x_ratio = size[0] / im_w
            y_ratio = size[1] / im_h
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.BILINEAR
        else:
            pil_inter = PIL.Image.NEAREST
        scaled = [img.resize(size, pil_inter) for img in clip]
        scaled_bbox = [[int(bb[0] * x_ratio), int(bb[1] * y_ratio), int(bb[2] * x_ratio), int(bb[3] * y_ratio)] for bb in bbox]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled, scaled_bbox


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def normalize(clip, mean, std, inplace=False):
    if not _is_tensor_clip(clip):
        raise TypeError('tensor is not a torch clip.')

    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

    return clip
