# -*- coding: utf-8 -*-
# @Time    : 2022/3/26 19:41
# @Author  : ycxia
# @File    : batch_inference.py
# @Brief   :
import argparse
import os
import os.path as osp
import sys
working_path=os.getcwd()
if working_path not in sys.path:
    sys.path.insert(1,working_path)
import cv2
import numpy as np
import torch
from mmcv import Config, DictAction

from mmaction.apis import init_recognizer
import time
import re
from operator import itemgetter
from mmcv.parallel import collate, scatter
from mmaction.core import OutputHook
from mmaction.datasets.pipelines import Compose
from tqdm import tqdm,trange
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


def inference_recognizer(model,
                         video_path,
                         label_path,
                         result_path,
                         step,
                         use_frames=False,
                         outputs=None,
                         as_tensor=True):
    """Inference a video with the detector.

    Args:
        model (nn.Module): The loaded recognizer.
        video_path (str): The video file path/url or the rawframes directory
            path. If ``use_frames`` is set to True, it should be rawframes
            directory path. Otherwise, it should be video file path.
        label_path (str): The label file path.
        use_frames (bool): Whether to use rawframes as input. Default:False.
        outputs (list(str) | tuple(str) | str | None) : Names of layers whose
            outputs need to be returned, default: None.
        as_tensor (bool): Same as that in ``OutputHook``. Default: True.

    Returns:
        dict[tuple(str, float)]: Top-5 recognition result dict.
        dict[torch.tensor | np.ndarray]:
            Output feature maps from layers specified in `outputs`.
    """
    if not (osp.exists(video_path) or video_path.startswith('http')):
        raise RuntimeError(f"'{video_path}' is missing")

    if osp.isfile(video_path) and use_frames:
        raise RuntimeError(
            f"'{video_path}' is a video file, not a rawframe directory")
    if osp.isdir(video_path) and not use_frames:
        raise RuntimeError(
            f"'{video_path}' is a rawframe directory, not a video file")

    if isinstance(outputs, str):
        outputs = (outputs, )
    assert outputs is None or isinstance(outputs, (tuple, list))

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # construct label map
    with open(label_path, 'r') as f:
        label = [line.strip() for line in f]
    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline[1:]
    # test_pipeline[0]['frame_uniform'] = True
    test_pipeline = Compose(test_pipeline)
    # start_index = cfg.data.test.get('start_index', 1)

    # count the number of frames that match the format of `filename_tmpl`
    # RGB pattern example: img_{:05}.jpg -> ^img_\d+.jpg$
    # Flow patteren example: {}_{:05d}.jpg -> ^x_\d+.jpg$
    filename_tmpl = 'frame{:06}.jpg'
    pattern = f'^{filename_tmpl}$'
    pattern = pattern.replace(
        pattern[pattern.find('{'):pattern.find('}') + 1], '\\d+')
    total_frames = len(
        list(
            filter(lambda x: re.match(pattern, x) is not None,
                   os.listdir(video_path))))
    outs = []

    with open(result_path, 'w') as f:
        print(time.asctime(time.localtime(time.time()))+'\n')
    for start_idx in trange(0,total_frames-(16-step),step):
        data=dict(frame_dir=video_path,
                  total_frames=start_idx+16,
                  label= -1,
                  start_index= start_idx,
                  filename_tmpl=filename_tmpl,
                  modality='RGB',
                  frame_inds=np.arange(start_idx,start_idx+16),
                  clip_len=16,
                  frame_interval= 1,
                  num_clips=1)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]

        # forward the model
        with OutputHook(model, outputs=outputs, as_tensor=as_tensor) as h:
            with torch.no_grad():
                scores = model(return_loss=False, **data)[0]
            returned_features = h.layer_outputs if outputs else None

        score_tuples = tuple(zip(label, scores))
        score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)

        top5_label = score_sorted[:5]
        outs.append(scores)
        with open(result_path, 'a') as f:
            f.writelines('{:d}'.format(start_idx))
            for score in scores:
                f.writelines(" {:.4f}".format(score))
            f.writelines('\n')
    return label, outs

def show_result(label_path,rslt_file):
    with open(label_path, 'r') as f:
        label = [line.strip() for line in f]
    results = np.loadtxt(rslt_file, delimiter=None)
    for result in results:
        clip_id = result[0]
        result = result[1:]
        print('{:>02d}:{:>02.2f}--{:s}, \t{:.4f}'.format(int(clip_id / 8 // 60), (clip_id / 8 % 60),
                                                       label[result.argmax()], result.max()))
    return

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('video_data_root', help='data_root directory')
    parser.add_argument('label', help='label file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--use-frames',
        default=True,
        action='store_true',
        help='whether to use rawframes as input')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--fps',
        default=30,
        type=int,
        help='specify fps value of the output video when using rawframes to '
        'generate file')
    parser.add_argument(
        '--font-scale',
        default=0.5,
        type=float,
        help='font scale of the label in output video')
    parser.add_argument(
        '--font-color',
        default='white',
        help='font color of the label in output video')
    parser.add_argument(
        '--step',
        default=8,
        type=int,
        help='window sliding step, e.g. 4, 8, 16')
    parser.add_argument(
        '--target-resolution',
        nargs=2,
        default=None,
        type=int,
        help='Target resolution (w, h) for resizing the frames when using a '
        'video as input. If either dimension is set to -1, the frames are '
        'resized by keeping the existing aspect ratio')
    parser.add_argument(
        '--resize-algorithm',
        default='bicubic',
        help='resize algorithm applied to generate video')
    parser.add_argument('--out_fname', default=None, help='output filename')
    parser.add_argument('--start_end', nargs=2, default=[0,None], type =int ,help='start end idx')
    parser.add_argument('--view', default='rear', type=str, help='view of dash, rear, right')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if os.path.split(args.video_data_root)[1].lower().startswith(('dash','rear','right')):
        # assign the desired device.
        video = os.path.split(args.video_data_root)[1]
        device = torch.device(args.device)
        cfg = Config.fromfile(args.config)
        cfg.merge_from_dict(args.cfg_options)
        # build the recognizer from a config file and checkpoint file/url
        model = init_recognizer(
            cfg, args.checkpoint, device=device, use_frames=args.use_frames)
        t = time.localtime(time.time())
        mkdir('./output')
        result_root = os.path.join('./output',
                                   "{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}".format(t.tm_year, t.tm_mon, t.tm_mday,
                                                                                 t.tm_hour, t.tm_min, t.tm_sec))
        mkdir(result_root)
        filen = result_root+'/'+video + '_' + args.checkpoint.split('/')[-1].split('.')[0] + f'_results_{args.step}.txt'
        results = inference_recognizer(
            model, args.video_data_root, args.label, filen, args.step,
            use_frames=args.use_frames)
        print('The top-5 labels with corresponding scores are:')
        show_result(args.label, filen)
        return
    else:
        videos = os.listdir(args.video_data_root)
        videos = list(filter(lambda x: x.lower().startswith(args.view), videos))
        # assign the desired device.
        device = torch.device(args.device)
        cfg = Config.fromfile(args.config)
        cfg.merge_from_dict(args.cfg_options)
        #build the recognizer from a config file and checkpoint file/url
        model = init_recognizer(
            cfg, args.checkpoint, device=device, use_frames=args.use_frames)
        t=time.localtime(time.time())
        result_root = os.path.join('./submit_results_source')#,"{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}".format(t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec))
        mkdir(result_root)
        for video in tqdm(videos[args.start_end[0]:args.start_end[1]]):
            filen = result_root+'/'+video +'_'+ args.checkpoint.split('/')[-1].split('.')[0] + f'_results_{args.step}.txt'
            results = inference_recognizer(
                model, os.path.join(args.video_data_root, video), args.label, filen, args.step, use_frames=args.use_frames)
            print('The top-5 labels with corresponding scores are:')
            show_result(args.label, filen)

        return


if __name__ == '__main__':
    main()
