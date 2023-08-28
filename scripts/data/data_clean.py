import os
import subprocess
import sys
import time
from multiprocessing import Pool

import decord
import ffmpeg



root_add = "path to dataset root"
save_root_add = "path to save the processed videos"

if not os.path.exists(save_root_add):
    os.makedirs(save_root_add)

def data_clean(list_file, idx):

    # prepare
    video_ext = 'mp4'

    # load video file list

    clips = list_file

    # process video & save
    start_time = time.time()

    directory = clips[idx]
    if '.' in directory.split('/')[-1]:
        video_name = directory
    else:
        video_name = '{}.{}'.format(directory, video_ext)

    try:
        # try load video

        decord_vr = decord.VideoReader(video_name, num_threads=1)

        duration = len(decord_vr)
        if duration < 30:
            return
        video_data = decord_vr.get_batch(list(range(duration))).asnumpy()

        # get the new size (short side size 320p)
        _, img_h, img_w, _ = video_data.shape
        new_short_size = 320
        ratio = float(img_h) / float(img_w)
        if ratio >= 1.0:
            new_w = int(new_short_size)
            new_h = int(new_w * ratio / 2) * 2
        else:
            new_h = int(new_short_size)
            new_w = int(new_h / ratio / 2) * 2
        new_size = (new_w, new_h)

    except Exception as e:
        # skip corrupted video files
        print("Failed to load video from {} with error {}".format(
            video_name, e))
        return

    # process the video
    output_video_file = os.path.join(save_root_add, directory.replace('.webm', '.mp4').split('/')[-1])

    # resize
    proc1 = (ffmpeg.input(directory).filter(
        'scale', new_size[0],
        new_size[1]).output(output_video_file).overwrite_output())
    p = subprocess.Popen(
        ['ffmpeg'] + proc1.get_args()+
        ['-hide_banner', '-loglevel', 'quiet', '-nostats'])


    end_time = time.time()
    dur_time = end_time - start_time
    print(f'processing video {idx + 1} with total time {dur_time} & save video in {output_video_file}')


if __name__ == '__main__':
    list_file = sys.argv[-1]
    n_tasks = 64
    new_start_idxs = [0] * n_tasks
    

    dir_path = rf'{root_add}20bn-something-something-v2'
    # print("Total number of videos: ", len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))]))
    list_file = [os.path.join(dir_path, entry) for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))]
    list_file = sorted(list_file, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    num_videos = len(list_file)
    print(f'load list file with {num_videos} videos successfully.')

    # data_clean(list_file, 0)

    with Pool(n_tasks) as p:
        p.starmap(data_clean,
                  [ (list_file, idx)
                   for idx in range(num_videos)])
