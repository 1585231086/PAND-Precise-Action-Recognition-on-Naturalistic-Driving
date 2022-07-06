import os
import multiprocessing as mp
import argparse
import cv2
import numpy as np
import shutil
import csv
from tqdm import tqdm, trange
import time
import sys
sys.path.insert(0,'../')
from AFSD.common.videotransforms import imresize

def rmdir(pdir):
    if os.path.isdir(pdir):
        shutil.rmtree(pdir)
        print("{} target has been deleted。".format(pdir))
    else:
        print('{} target dose not exist.'.format(pdir))


def mkdir(pdir):
    if os.path.exists(pdir):
        print("{} target has existed。".format(pdir))
    else:
        os.makedirs(pdir)
        print('{} dir is built。'.format(pdir))


parser = argparse.ArgumentParser()
parser.add_argument('--thread_num', default=1, type=int)
parser.add_argument('--video_root_dir', type=str, default='/home/xyc/AICity/data/')
parser.add_argument('--output_root_dir', type=str, default='/home/xyc/AICity/dataB')
parser.add_argument('--video_info_path', type=str, default='/home/xyc/data/AICity/info/test_video_info.csv')
parser.add_argument('--max_frame_num', type=int, default=768)
args = parser.parse_args()

thread_num = args.thread_num
video_root_dir = args.video_root_dir
output_root_dir = args.output_root_dir

mkdir(output_root_dir)
mkdir(os.path.split(args.video_info_path)[0])
videos_path = []

with open(args.video_info_path, "w") as f:
    csv_writer = csv.writer(f, dialect="excel")
    csv_writer.writerow(['video', 'fps', 'sample_fps', 'count', 'sample_count'])

for user in os.listdir(video_root_dir):
    for file in os.listdir(os.path.join(video_root_dir, user)):
        if file.endswith(('.MP4', '.mp4')):
            videos_path.append(os.path.join(video_root_dir, user, file))
            capV = cv2.VideoCapture(os.path.join(video_root_dir, user, file))
            fpsV = int(capV.get(cv2.CAP_PROP_FPS))
            count_total = int(capV.get(cv2.CAP_PROP_FRAME_COUNT))
            with open(args.video_info_path, "a") as f:
                csv_writer = csv.writer(f, dialect="excel")
                csv_writer.writerow([user + '/' + file.split('.')[0], fpsV, fpsV, count_total, count_total])


def sub_processor(pid, files,sample_fps=10,resolution=112):
    for file in files[:]:
        user = os.path.split(os.path.split(file)[0])[1]
        mkdir(os.path.join(output_root_dir, user))
        file_name = os.path.splitext(os.path.split(file)[1])[0]

        target_file = os.path.join(output_root_dir, user, file_name + '.npy')
        print("{:s} is processing... \tTarget file is {:s}.".format(file, target_file))
        cap = cv2.VideoCapture(file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        imgs = []
        t1=time.time()
        # for i in trange(int(count)):
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
        #     imgs.append(frame[:, :, ::-1])
        # if count != len(imgs):
        #     print('{} frame num is less'.format(file_name))
        # imgs = np.stack(imgs)
        # print(imgs.shape)
        step = fps / sample_fps
        cur_step = .0
        cur_count = 0
        save_count = 0
        for i in trange(int(count)):
            ret, frame = cap.read()
            if ret is False:
                break
            frame = np.array(frame)[:, :, ::-1]
            cur_count += 1
            cur_step += 1
            if cur_step >= step:
                cur_step -= step
                # save the frame
                target_img = imresize(frame, [resolution, resolution], 'bicubic')
                imgs.append(target_img)
                save_count += 1
        imgs = np.stack(imgs)
        print(imgs.shape)
        # if max_frame_num is not None:
        #     imgs = imgs[:max_frame_num]
        print("Saving.....")
        np.save(target_file, imgs)
        print("{:s} is done. Cost {:.2f}s".format(target_file,time.time()-t1))


processes = []
video_num = len(videos_path)
per_process_video_num = video_num // thread_num

for i in range(thread_num):
    if i == thread_num - 1:
        sub_files = videos_path[i * per_process_video_num:]
    else:
        sub_files = videos_path[i * per_process_video_num: (i + 1) * per_process_video_num]
    p = mp.Process(target=sub_processor, args=(i, sub_files))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
