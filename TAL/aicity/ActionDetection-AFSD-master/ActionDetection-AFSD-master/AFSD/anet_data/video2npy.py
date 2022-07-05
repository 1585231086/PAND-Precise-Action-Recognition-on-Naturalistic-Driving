import os
import multiprocessing as mp
import argparse
import cv2
import numpy as np
import shutil

def rmdir(pdir):
    if os.path.isdir(pdir):
        shutil.rmtree(pdir)
        print("{}目标已删除。".format(pdir))
    else:
        print('{}目录不存在。'.format(pdir))

def mkdir(pdir):
    if os.path.isdir(pdir):
        print("{}目标已存在。".format(pdir))
    else:
        os.mkdir(pdir)
        print('{}目录已创建。'.format(pdir))

parser = argparse.ArgumentParser()
parser.add_argument('--thread_num', default=1, type=int)
parser.add_argument('--video_root_dir', type=str, default='/home/xyc/AICity/video_data')
parser.add_argument('--output_root_dir', type=str, default='/home/xyc/AICity/dataB')
parser.add_argument('--max_frame_num', type=int, default=768)
args = parser.parse_args()

thread_num = args.thread_num
video_root_dir = args.video_root_dir
output_root_dir = args.output_root_dir
max_frame_num = args.max_frame_num

mkdir(output_root_dir)

videos_path = []
for user in os.listdir(video_root_dir):
    for file in os.listdir(os.path.join(video_root_dir,user)):
        if file.endswith(('.MP4','.mp4')):
            videos_path.append(os.path.join(video_root_dir,user,file))

def sub_processor(pid, files):
    for file in files[:]:
        user = os.path.split(os.path.split(file)[0])[1]
        mkdir(os.path.join(output_root_dir, user))
        file_name = os.path.splitext(os.path.split(file)[1])[0]

        target_file = os.path.join(output_root_dir, user, file_name + '.npy')
        print("{:s} is processing... \tTarget file is {:s}.".format(file, target_file))
        cap = cv2.VideoCapture(file)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        imgs = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            imgs.append(frame[:, :, ::-1])
        if count != len(imgs):
            print('{} frame num is less'.format(file_name))
        imgs = np.stack(imgs)
        print(imgs.shape)
        # if max_frame_num is not None:
        #     imgs = imgs[:max_frame_num]
        np.save(target_file, imgs)

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
