# -*- coding: utf-8 -*-
# @Time    : 2022/3/22 9:10
# @Author  : ycxia
# @File    : pose_pkl_build.py
# @Brief   : build the pose detection pkl file.
# Copyright (c) OpenMMLab. All rights reserved.
import abc
import argparse
import os
import os.path as osp
import random as rd
import shutil
import string
import warnings
from collections import defaultdict
from tqdm import tqdm,trange
import cv2
import mmcv
import numpy as np
from copy import deepcopy
import time
try:
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except ImportError:
    warnings.warn(
        'Please install MMDet and MMPose for NTURGB+D pose extraction.'
    )  # noqa: E501

mmdet_root = './mmdetection'
mmpose_root = './mmpose-master'

args = abc.abstractproperty()
args.det_config = './demo/faster_rcnn_r50_fpn_2x_coco.py'#f'{mmdet_root}/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'  # noqa: E501
args.det_checkpoint = './models/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'#f''  #https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth noqa: E501
args.det_score_thr = 0.5
args.pose_config = './demo/hrnet_w32_coco_256x192.py'#f'{mmpose_root}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'  # noqa: E501
args.pose_checkpoint = './models/hrnet_w32_coco_256x192-c78dce93_20200708.pth'#'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'  # noqa: E501


def gen_id(size=8):
    chars = string.ascii_uppercase + string.digits
    return ''.join(rd.choice(chars) for _ in range(size))


def extract_frame(video_path):
    dname = gen_id()
    os.makedirs(dname, exist_ok=True)
    frame_tmpl = osp.join(dname, 'img_{:05d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    while flag:
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths


def detection_inference(args, frame_paths):
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def intersection(b0, b1):
    l, r = max(b0[0], b1[0]), min(b0[2], b1[2])
    u, d = max(b0[1], b1[1]), min(b0[3], b1[3])
    return max(0, r - l) * max(0, d - u)


def iou(b0, b1):
    i = intersection(b0, b1)
    u = area(b0) + area(b1) - i
    return i / u


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def removedup(bbox):

    def inside(box0, box1, thre=0.8):
        return intersection(box0, box1) / area(box0) > thre

    num_bboxes = bbox.shape[0]
    if num_bboxes == 1 or num_bboxes == 0:
        return bbox
    valid = []
    for i in range(num_bboxes):
        flag = True
        for j in range(num_bboxes):
            if i != j and inside(bbox[i],
                                 bbox[j]) and bbox[i][4] <= bbox[j][4]:
                flag = False
                break
        if flag:
            valid.append(i)
    return bbox[valid]


def is_easy_example(det_results, num_person):
    threshold = 0.95

    def thre_bbox(bboxes, thre=threshold):
        shape = [sum(bbox[:, -1] > thre) for bbox in bboxes]
        ret = np.all(np.array(shape) == shape[0])
        return shape[0] if ret else -1

    if thre_bbox(det_results) == num_person:
        det_results = [x[x[..., -1] > 0.95] for x in det_results]
        return True, np.stack(det_results)
    return False, thre_bbox(det_results)


def bbox2tracklet(bbox):
    iou_thre = 0.6
    tracklet_id = -1
    tracklet_st_frame = {}
    tracklets = defaultdict(list)
    for t, box in enumerate(bbox):
        for idx in range(box.shape[0]):
            matched = False
            for tlet_id in range(tracklet_id, -1, -1):
                cond1 = iou(tracklets[tlet_id][-1][-1], box[idx]) >= iou_thre
                cond2 = (
                    t - tracklet_st_frame[tlet_id] - len(tracklets[tlet_id]) <
                    10)
                cond3 = tracklets[tlet_id][-1][0] != t
                if cond1 and cond2 and cond3:
                    matched = True
                    tracklets[tlet_id].append((t, box[idx]))
                    break
            if not matched:
                tracklet_id += 1
                tracklet_st_frame[tracklet_id] = t
                tracklets[tracklet_id].append((t, box[idx]))
    return tracklets


def drop_tracklet(tracklet):
    tracklet = {k: v for k, v in tracklet.items() if len(v) > 5}

    def meanarea(track):
        boxes = np.stack([x[1] for x in track]).astype(np.float32)
        areas = (boxes[..., 2] - boxes[..., 0]) * (
            boxes[..., 3] - boxes[..., 1])
        return np.mean(areas)

    tracklet = {k: v for k, v in tracklet.items() if meanarea(v) > 5000}
    return tracklet


def distance_tracklet(tracklet):
    dists = {}
    for k, v in tracklet.items():
        bboxes = np.stack([x[1] for x in v])
        c_x = (bboxes[..., 2] + bboxes[..., 0]) / 2.
        c_y = (bboxes[..., 3] + bboxes[..., 1]) / 2.
        c_x -= 480
        c_y -= 270
        c = np.concatenate([c_x[..., None], c_y[..., None]], axis=1)
        dist = np.linalg.norm(c, axis=1)
        dists[k] = np.mean(dist)
    return dists


def tracklet2bbox(track, num_frame):
    # assign_prev
    bbox = np.zeros((num_frame, 5))
    trackd = {}
    for k, v in track:
        bbox[k] = v
        trackd[k] = v
    for i in range(num_frame):
        if bbox[i][-1] <= 0.5:
            mind = np.Inf
            for k in trackd:
                if np.abs(k - i) < mind:
                    mind = np.abs(k - i)
            bbox[i] = bbox[k]
    return bbox


def tracklets2bbox(tracklet, num_frame):
    dists = distance_tracklet(tracklet)
    sorted_inds = sorted(dists, key=lambda x: dists[x])
    dist_thre = np.Inf
    for i in sorted_inds:
        if len(tracklet[i]) >= num_frame / 2:
            dist_thre = 2 * dists[i]
            break

    dist_thre = max(50, dist_thre)

    bbox = np.zeros((num_frame, 5))
    bboxd = {}
    for idx in sorted_inds:
        if dists[idx] < dist_thre:
            for k, v in tracklet[idx]:
                if bbox[k][-1] < 0.01:
                    bbox[k] = v
                    bboxd[k] = v
    bad = 0
    for idx in range(num_frame):
        if bbox[idx][-1] < 0.01:
            bad += 1
            mind = np.Inf
            mink = None
            for k in bboxd:
                if np.abs(k - idx) < mind:
                    mind = np.abs(k - idx)
                    mink = k
            bbox[idx] = bboxd[mink]
    return bad, bbox


def bboxes2bbox(bbox, num_frame):
    ret = np.zeros((num_frame, 2, 5))
    for t, item in enumerate(bbox):
        if item.shape[0] <= 2:
            ret[t, :item.shape[0]] = item
        else:
            inds = sorted(
                list(range(item.shape[0])), key=lambda x: -item[x, -1])
            ret[t] = item[inds[:2]]
    for t in range(num_frame):
        if ret[t, 0, -1] <= 0.01:
            ret[t] = ret[t - 1]
        elif ret[t, 1, -1] <= 0.01:
            if t:
                if ret[t - 1, 0, -1] > 0.01 and ret[t - 1, 1, -1] > 0.01:
                    if iou(ret[t, 0], ret[t - 1, 0]) > iou(
                            ret[t, 0], ret[t - 1, 1]):
                        ret[t, 1] = ret[t - 1, 1]
                    else:
                        ret[t, 1] = ret[t - 1, 0]
    return ret


def ntu_det_postproc(vid, det_results):
    det_results = [removedup(x) for x in det_results]
    label = int(vid.split('/')[-1].split('A')[1][:3])
    mpaction = list(range(50, 61)) + list(range(106, 121))
    n_person = 2 if label in mpaction else 1
    is_easy, bboxes = is_easy_example(det_results, n_person)
    if is_easy:
        print('\nEasy Example')
        return bboxes

    tracklets = bbox2tracklet(det_results)
    tracklets = drop_tracklet(tracklets)

    print(f'\nHard {n_person}-person Example, found {len(tracklets)} tracklet')
    if n_person == 1:
        if len(tracklets) == 1:
            tracklet = list(tracklets.values())[0]
            det_results = tracklet2bbox(tracklet, len(det_results))
            return np.stack(det_results)
        else:
            bad, det_results = tracklets2bbox(tracklets, len(det_results))
            return det_results
    # n_person is 2
    if len(tracklets) <= 2:
        tracklets = list(tracklets.values())
        bboxes = []
        for tracklet in tracklets:
            bboxes.append(tracklet2bbox(tracklet, len(det_results))[:, None])
        bbox = np.concatenate(bboxes, axis=1)
        return bbox
    else:
        return bboxes2bbox(det_results, len(det_results))


def pose_inference(args, frame_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))

    num_frame = len(det_results)
    num_person = max([len(x) for x in det_results])
    kp = np.zeros((num_person, num_frame, 17, 3), dtype=np.float32)

    for i, (f, d) in enumerate(zip(frame_paths, det_results)):
        # Align input format
        d = [dict(bbox=x) for x in list(d) if x[-1] > 0.5]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        for j, item in enumerate(pose):
            kp[j, i] = item['keypoints']
        prog_bar.update()
    return kp

def load_person_bbox(det_root):#'date_detect'
    det_result_dict=dict()
    for video_frames_dir in tqdm(os.listdir(det_root)):

        video_frames_path = os.path.join(det_root, video_frames_dir)
        with open(os.path.join(video_frames_path,'bbox_reuslts.txt'),'r') as f:
            frames_results = f.readlines()
        frame_num = len(os.listdir(video_frames_path))-2#2个txt文件
        t = time.time()
        n=0
        # print('\r', "[" + ">" * int(n / frame_num * 25) + "." * (
        #         25 - int(n / frame_num * 25)) + "]{:.2f}s : {:.2f}% has been done.".format(
        #     time.time() - t, n / frame_num * 100), end='', flush=False)
        det_results = []
        for i, line in enumerate(frames_results):
            line = line.split('\n')[0]
            if line.startswith('/home'):
                frame_id = int(os.path.split(line)[1][5:-4])
                if n != 0:
                    if len(results) == 0:
                        det_results.append([[n-1,0,0,0,0,0,0]])
                    else:
                        det_results.append(deepcopy(results))
                results=[]

                # print('\r', "{:s}[".format(os.path.split(video_frames_path)[1]) + ">" * int(n / frame_num * 25) + "." * (
                #             25 - int(n / frame_num * 25)) + "]{:.2f}s : {:.2f}% has been done.".format(
                #     time.time() - t, n / frame_num * 100), end='', flush=True)
                n+=1
                # det_results.append(results)
            else:
                if line.endswith(',0'):# person类别的index是0
                    result = line.split(',')
                    result = [int(result[0]), int(result[1]), int(result[2]), int(result[3]), float(result[4])]
                    results.append([frame_id]+result+[(result[2]-result[0])*(result[3]-result[1])])
        det_results.append(deepcopy(results))
        out_results=[]
        for results in det_results:
            npresults=np.array(results)
            try:
                max_idx=np.argmax(npresults[:,-1])
            except:
                print(npresults)
            out_results.append(npresults[max_idx])
        det_result_dict[video_frames_dir] = deepcopy(out_results)
    return det_result_dict

# def load_labels_from_json(data_root):
#     out_labels=dict()
#     dict = load_json_file(os.path.join(data_root, 'labels.json'))
#     for video_frames_dir in tqdm(os.listdir(det_root)):
#         true_labels = np.zeros(frames_num, dtype='int') + 18
#         for key in dict[video.lower()].keys():
#             begin_frame = int(timecvt(dict[video.lower()][key]['begin']) * 8)
#             end_frame = int(timecvt(dict[video.lower()][key]['end']) * 8)
#             true_labels[begin_frame:end_frame] = int(key)
#         out_labels[video_frames_dir]=true_labels
#     return out_labels

def load_labels_from_txt(label_file,pre='dash'):
    out_labels=dict()
    with open(label_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        line = line.split('\n')[0]
        line = line.split(' ')
        video = line[0]
        begin_frame = int(line[1])
        clip_size = 16  # line[-2]
        label = int(line[-1])
        if label == 18:###TODO label != 18 and video.lower().startswith(pre)
            if video not in out_labels.keys():
                out_labels[video]=[[begin_frame, label]]
            out_labels[video].append([begin_frame, label])
    return out_labels

def split_by_clip(det_root,det_result_dict,total_labels):
    frame_pathss, det_resultss, labels = [],[],[]
    for video in tqdm(total_labels.keys()):
        for label in total_labels[video]:#[begin_frame, label]
            frame_pathss.append([os.path.join(det_root,video, "frame{:>06d}.jpg".format(label[0]+i)) for i in range(16)])
            det_resultss.append([x[1:6].reshape(1,5) for x in det_result_dict[video][label[0]:label[0]+16]])
            labels.append(label)
    return frame_pathss,det_resultss,labels



def ntu_pose_extraction(det_root, label_file, skip_postproc=False, pre='dash'):
    # frame_paths = extract_frame(vid)
    det_result_dict=load_person_bbox(det_root)
    total_labels=load_labels_from_txt(label_file, pre)
    frame_pathss,det_resultss,labels=split_by_clip(det_root,det_result_dict,total_labels)
    # det_results = detection_inference(args, frame_paths)
    # if not skip_postproc:
    #     det_results = ntu_det_postproc(vid, det_results)
    import random

    idxs=np.arange(len(labels))
    # random.shuffle(idxs)
    annos=[]
    # for frame_paths,det_results,label in tqdm(zip(frame_pathss,det_resultss,labels)):
    for i in trange(len(idxs)):
        frame_paths, det_results, label = frame_pathss[i],det_resultss[i],labels[i]
        pose_results = pose_inference(args, frame_paths, det_results)
        anno = dict()
        anno['keypoint'] = pose_results[..., :2]
        anno['keypoint_score'] = pose_results[..., 2]
        anno['frame_dir'] = osp.basename(osp.split(frame_paths[0])[0])+'__'+osp.splitext(osp.basename(frame_paths[0]))[0]
        anno['img_shape'] = (1080, 1920)
        anno['original_shape'] = (1080, 1920)
        anno['total_frames'] = pose_results.shape[1]
        anno['label'] = label[1]
        annos.append(anno)
    # shutil.rmtree(osp.dirname(frame_paths[0]))

    return annos

def ntu_pose_extraction_all(det_root, label_file, skip_postproc=False, pre='dash'):
    # frame_paths = extract_frame(vid)
    det_result_dict=load_person_bbox(det_root)
    # frame_pathss,det_resultss,labels=split_by_clip(det_root,det_result_dict,total_labels)
    # det_results = detection_inference(args, frame_paths)
    # if not skip_postproc:
    #     det_results = ntu_det_postproc(vid, det_results)
    # import random
    #
    # idxs=np.arange(len(labels))
    # random.shuffle(idxs)
    annos=[]
    # for frame_paths,det_results,label in tqdm(zip(frame_pathss,det_resultss,labels)):
    for video in tqdm(det_result_dict.keys()):
        for i in range(0,len(det_result_dict[video])-16,16):
            det_results = [x[1:6].reshape(1,5) for x in det_result_dict[video][i:i+16]]
            frame_paths = [os.path.join(det_root,video, "frame{:>06d}.jpg".format(i+idx)) for idx in range(16)]
            pose_results = pose_inference(args, frame_paths, det_results)
            anno = dict()
            anno['keypoint'] = pose_results[..., :2]
            anno['keypoint_score'] = pose_results[..., 2]
            anno['frame_dir'] = osp.basename(osp.split(frame_paths[0])[0])+'__'+osp.splitext(osp.basename(frame_paths[0]))[0]
            anno['img_shape'] = (1080, 1920)
            anno['original_shape'] = (1080, 1920)
            anno['total_frames'] = pose_results.shape[1]
            anno['label'] = 0
            annos.append(anno)
        if len(det_result_dict[video])%16!=0:
            print(len(det_result_dict[video]))
            det_results = [x[1:6].reshape(1, 5) for x in det_result_dict[video][i+16:]]
            frame_paths = [os.path.join(det_root, video, "frame{:>06d}.jpg".format(i+16 + idx)) for idx in range(len(det_result_dict[video])-i-16)]
            pose_results = pose_inference(args, frame_paths, det_results)
            anno = dict()
            anno['keypoint'] = pose_results[..., :2]
            anno['keypoint_score'] = pose_results[..., 2]
            anno['frame_dir'] = osp.basename(osp.split(frame_paths[0])[0]) + '__' + \
                                osp.splitext(osp.basename(frame_paths[0]))[0]
            anno['img_shape'] = (1080, 1920)
            anno['original_shape'] = (1080, 1920)
            anno['total_frames'] = pose_results.shape[1]
            anno['label'] = 0
            annos.append(anno)
    # shutil.rmtree(osp.dirname(frame_paths[0]))

    return annos

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Pose Annotation for a single NTURGB-D video')
    parser.add_argument('--det_root',default='/home/spring/aicity/data/data_detect2', type=str, help='detection images save dir')
    parser.add_argument('--output',default='/home/spring/aicity/data/pose/A2_pose_val.pkl', type=str, help='output pickle name')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--label_file',default='/home/spring/aicity/data/labels/train_35133_0_5sGap_add24_8_background_labels_with_bboxes.txt', type=str, help=' txt')
    parser.add_argument('--pre',default='dash', type=str, help=' dash, rear or right')

    args = parser.parse_args()
    return args

def divide_view(ann_file,view):
    import pickle
    with open(ann_file, 'rb') as f:
        data = pickle.load(f)
    out_data = []
    for ann_info in data:
        if ann_info['frame_dir'].lower().startswith(view):
            out_data.append(ann_info)
    out_name = ann_file.split('.')[0]+'_'+view
    mmcv.dump(out_data, out_name+'.pkl')
    return

if __name__ == '__main__':
    global_args = parse_args()
    args.device = global_args.device
    args.det_root = global_args.det_root
    args.output = global_args.output
    args.label_file = global_args.label_file
    args.pre = global_args.pre
    anno = ntu_pose_extraction_all(args.det_root, args.label_file, pre = args.pre)
    mmcv.dump(anno, args.output)
    # divide_view(args.output, 'rear')