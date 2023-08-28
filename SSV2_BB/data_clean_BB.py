import os
import subprocess
import sys
import time
from multiprocessing import Pool
import orjson 
import decord
import ffmpeg
from joblib import delayed, Parallel  # install psutil library to manage memory leak
from tqdm import tqdm




root_add = "/home/mona/VideoMAE/dataset/somethingsomething/"
save_root_add = "/home/mona/VideoMAE/dataset/somethingsomething/mp4_videos_BB"
# read json bbox files
Total_video_BB={}
for i in range(1,5):
    with open(os.path.join('/home/mona/VideoMAE/SSV2_BB/',f'bounding_box_smthsmth_part{str(i)}.json'), "r", encoding="utf-8") as f:
            video_BB = orjson.loads(f.read())
    Total_video_BB.update(video_BB)


if not os.path.exists(save_root_add):
    os.makedirs(save_root_add)

def data_clean(list_file, idx, verbose=True):

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
            return [-1, -1, -1]
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
        
        #scale the BBx 
        y_ratio_bb = new_h/img_h
        x_ratio_bb = new_w/img_w
        new_BB_dict = [-1, -1, -1]
        i = video_name.split('/')[-1].split('.')[0]
        if str(i) in Total_video_BB:
            new_BB_dict = [int(i), x_ratio_bb, y_ratio_bb]    
        new_size = (new_w, new_h)

    except Exception as e:
        # skip corrupted video files
        print("Failed to load video from {} with error {}".format(
            video_name, e))
        return [-1, -1, -1]

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
    if verbose:
        print(f'processing video {idx + 1} with total time {dur_time} & save video in {output_video_file}')

    return new_BB_dict


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

    # print(Total_video_BB ['4'][0]['labels'][0]['box2d'])
    # print(Total_video_BB ['5'][0]['labels'][0]['box2d'])

    # data_clean(list_file, 4)

    with Pool(n_tasks) as p:
        outputs=p.starmap(data_clean,
                  [ (list_file, idx)
                   for idx in range(len(list_file))])

    # modify the x and y coordinates
    for out in outputs:
        if out[0] != -1:
            i, x_ratio, y_ratio = out
            i = str(i)
            for j in range(len(Total_video_BB[i])):
                for id, _ in enumerate(Total_video_BB[i][j]['labels']):
                    x1 = Total_video_BB [i][j]['labels'][id]['box2d']['x1']*x_ratio
                    x2 = Total_video_BB [i][j]['labels'][id]['box2d']['x2']*x_ratio
                    y1 = Total_video_BB [i][j]['labels'][id]['box2d']['y1']*y_ratio
                    y2 = Total_video_BB [i][j]['labels'][id]['box2d']['y2']*y_ratio
                    Total_video_BB[i][j]['labels'][id]['box2d']={'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}

    # print(Total_video_BB ['4'][0]['labels'][0]['box2d'])
    # print(Total_video_BB ['5'][0]['labels'][0]['box2d'])


    sacaledbbx_add = os.path.join('/home/mona/VideoMAE/SSV2_BB/','bounding_box_smthsmth_scaled.json')
    with open(sacaledbbx_add, "w+", encoding="utf-8") as f:
        f.write(orjson.dumps(Total_video_BB).decode("utf-8"))