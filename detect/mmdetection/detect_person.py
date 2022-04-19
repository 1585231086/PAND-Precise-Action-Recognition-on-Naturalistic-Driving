# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 14:32
# @Author  : ycxia
# @File    : detect_person.py
# @Brief   : create the original object detection txt files for every video.

from mmdet.apis import init_detector#, inference_detector
import mmcv
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from config import config
from core.visualization.image import imshow_det_bboxes
import numpy as np
import torch
import time
import cv2
import os
import shutil
import argparse
from utils import load_json_file,load_json_lines,save_json_lines,mkdir,rmdir
from pathlib import *
from tqdm import tqdm

def filter_cls(result, class_names):
    out=([np.array([],dtype=np.float32).reshape(0,5)]*80,[[]]*80)
    for cls in class_names:
        ind=config.classes.index(cls)
        out[0][ind] = result[0][ind]
        out[1][ind] = result[1][ind]
    return out

def show_result(
                img,
                result,
                Clss,
                out_file,
                score_thr = 0.3):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor or tuple): The results to draw over `img`
            bbox_result or (bbox_result, segm_result).
        score_thr (float, optional): Minimum score of bboxes to be shown.
            Default: 0.3.
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (None or str or tuple(int) or :obj:`Color`):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """

    bbox_color = (72, 101, 241)
    text_color = (72, 101, 241)
    mask_color = None
    thickness = 2
    font_size = 13
    win_name = ''
    show = False
    wait_time = 0
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = True
    # draw bounding boxes
    img = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms,
        class_names=Clss,
        score_thr=score_thr,
        bbox_color=bbox_color,
        text_color=text_color,
        mask_color=mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    return img



def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    cfg.data.test.pipeline[1]['img_scale'] = (480, 270)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=False, **data)

    if not is_batch:
        return results[0]
    else:
        return results


def batch_detect_by_frame(imgs_dir, result_dir, model):
    t = time.time()
    imgfiles = os.listdir(imgs_dir)
    imgfiles.sort(key=lambda x: int(x[5:-4]))
    imgs = [os.path.join(imgs_dir, x) for x in imgfiles]
    mkdir(result_dir)
    save_dir = os.path.join(result_dir, os.path.split(imgs_dir)[1])
    mkdir(save_dir)
    batch_size=4
    for i in range(len(imgfiles)):
        img=imgfiles[i]
        if not os.path.isfile(os.path.join(save_dir, img)):
            result = inference_detector(model, imgs[i])
            score_thr = 0.4
            bbox_result, segm_result = result
            bboxes = np.vstack(bbox_result)
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = [
                np.full(bbox.shape[0], ii, dtype=np.int32)
                for ii, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            labels = labels[inds]
            with open(os.path.join(save_dir, 'bbox_reuslts.txt'), 'a') as f:
                f.writelines(["{:s}\n".format(os.path.join(imgs_dir,img))])
                for idxs, (bbox, label) in enumerate(zip(bboxes, labels)):
                    f.writelines(["{:d},{:d},{:d},{:d},{:.5f},{:d}\n".format(int(bbox[0]), int(bbox[1]),
                                                                             int(bbox[2]), int(bbox[3]), bbox[4],
                                                                             label)])
            show_result(os.path.join(imgs_dir, img), result, Clss=model.CLASSES,
                        out_file=os.path.join(save_dir, img))
        print('\r', "["+">"*int(i/(len(imgfiles))*25)+"."*(25-int(i/(len(imgfiles))*25))+"]{:.2f}s : {:.2f}% has been done.".format(time.time() - t, i/(len(imgfiles))*100), end='', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('-dir',default=r'/home/xyc/data/aicity/data2', help='laptop_videoinfo.json./2018-03-05.13-10-00.13-15-00.hospital.G341.avitrain config file path')
    parser.add_argument('-result-dir', default=r'/home/xyc/data/aicity/data_detect2',
                        help='laptop_videoinfo.json./2018-03-05.13-10-00.13-15-00.hospital.G341.avitrain config file path')
    parser.add_argument('-dev', default='cuda:0',
                        help='device')

    parser.add_argument('--indx',nargs=2, default=[0,None], type=int,
                        help='device')

    args = parser.parse_args()

    config_file = './configs_detect/detectors/detectors_htc_r50_1x_coco.py'
    checkpoint_file = './models/detectors_htc_r50_1x_coco-329b1453.pth'

    # model initial
    model = init_detector(config_file, checkpoint_file, device=args.dev)

    for frames_dir in tqdm(os.listdir(args.dir)[args.indx[0]:args.indx[1]]):
        batch_detect_by_frame(os.path.join(args.dir, frames_dir), args.result_dir, model)
    # singleclass_json_detect(args.dir, model,config)


if __name__ == '__main__':
    main()
