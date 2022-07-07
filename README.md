# PAND-Precise-Action-Recognition-on-Naturalistic-Driving
A solution for the 6th Aicity challenge workshop track 3.  Temporal action localization is an important problem in computer vision. It is challenge to infer the start and end of action instances on small-scale datasets covering multi-view information accurately. In this paper, we propose an effective action temporal localization method to localize the temporal boundaries. Our approach in-cludes (i) a method integrating an action recognition network and a temporal action localization network, (ii) a post-processing method for selecting and correcting temporal boxes to ensure that the model finds accurate boundaries. In addition, the frame-level object detection information is also utilized. Extensive experiments prove the effectiveness of our method and we rank the 6th on the Test-A2 of the 6th AI City Challenge Track 3. 
# Getting Started

This page provides basic tutorials about the usage of PAND.

<!-- TOC -->

- [Datasets](#datasets)
- [Inference with Pre-Trained Models](#inference-with-pre-trained-models)
  - [Test a dataset](#test-a-dataset)
- [Train a Model](#train-a-model)
  - [Train with a single GPU](#train-with-a-single-gpu)
  - [Train with multiple GPUs](#train-with-multiple-gpus)

<!-- TOC -->

## Datasets

The folder structure:

```
/xxxx
  ├──Aicity
    ├── annotations
    ├── bottle_detect
    ├── cellphone_detect
    ├── data
    │   ├── VIDEO1
    │   │   ├── frame000000.jpg
    │   │   ├── frame000001.jpg
    │   │   ├── ...
    │   ├── VIDEO2
    │   │   ├── frame000000.jpg
    │   │   ├── frame000001.jpg
    │   │   ├── ...
    │   ├── ...
    ├── data_detect
    │   ├── VIDEO1
    │   │   ├── bbox_reuslts.txt
    │   │   ├── frame000000.jpg
    │   │   ├── frame000001.jpg
    │   │   ├── ...
    │   ├── VIDEO2
    │   │   ├── bbox_reuslts.txt
    │   │   ├── frame000000.jpg
    │   │   ├── frame000001.jpg
    │   │   ├── ...
    │   ├── ...
    ├── labels
    ├── pose
    ├── videos
    │   ├── user_id_xxx
    │   │   ├── VIDEO1.MP4
    │   │   ├── VIDEO2.MP4
    │   │   ├── ...
    │   ├── ...
    ├── video_ids.csv
```

## Inference with Pre-Trained Models

We provide testing scripts to evaluate a whole dataset (Kinetics-400, Something-Something V1&V2, (Multi-)Moments in Time, etc.),
and provide some high-level apis for easier integration to other projects.

### Test a dataset

You can use the following commands to test a dataset.

```shell
# divide frame from videos
python data_process.py --video_dir /xxxx/Aicity/videos --data_root /xxxx/Aicity/data

# detect bottle and cellphone
cd detect/mmdetection
python detect_person.py -dir /xxxx/Aicity/data -result-dir /xxxx/Aicity/data_detect -dev cuda:0

# keypoint detect
cd ..
python pose_pkl_build.py --det_root /xxxx/Aicity/data_detect --output /xxxx/Aicity/pose/B_pose.pkl --device cuda:0

# action recognition
cd ..
cd Swin-transformer/Video-Swin-Transformer-master
python tools/batch_inference.py ./work_dirs/exp4/swin_base_patch244_window877_kinetics400_1k.py ./work_dirs/exp4/latest.pth /xxxx/Aicity/data ./label_map.txt --step 4 --device cuda:0 --view dash
python tools/batch_inference.py ./work_dirs/exp5/swin_base_patch244_window877_kinetics400_1k.py ./work_dirs/exp5/latest.pth /xxxx/Aicity/data ./label_map.txt --step 4 --device cuda:0 --view right
python tools/batch_inference.py ./work_dirs/exp6/swin_base_patch244_window877_kinetics400_1k.py ./work_dirs/exp6/latest.pth /xxxx/Aicity/data ./label_map.txt --step 4 --device cuda:0 --view rear

# TAL
cd ..
cd TAL/aicity/ActionDetection-AFSD-master/ActionDetection-AFSD-master/AFSD
# Convert video to npy file
python anet_data/video2npy.py --video_root_dir /xxxx/Aicity/videos --output_root_dir /xxxx/Aicity/dataNPY
# inference
python thumos14/test.py ../configs/thumos14.yaml --checkpoint_path=models/thumos14/checkpoint-150.ckpt --output_json=../../../thumos14_rgb.json
# NMS process
cd ../../../
python time_postp.py --input_dir thumos14_rgb.json --output_dir ./output --recog_result_dir ../../Swin-transformer/Video-Swin-Transformer-master/submit_results_source --video_ids /xxxx/Aicity/data/video_ids.csv --first true
python process0.py  --pose_pkl /xxxx/Aicity/pose/B_pose.pkl --recog_result_dir ../Swin-transformer/Video-Swin-Transformer-master/submit_results_source --bottle_det_dir /xxxx/Aicity/bottle_detect --cellphone_det_dir /xxxx/Aicity/cellphone_detect --video_ids /xxxx/Aicity/data/video_ids.csv
python time_postp.py --input_dir thumos14_rgb.json --output_dir ./output --recog_result_dir ../../Swin-transformer/Video-Swin-Transformer-master/submit_results_source --video_ids /xxxx/Aicity/data/video_ids.csv --first false

# post process
cd ..
cd ..
cd Post_process
python submit_zhy_new.py --pose_pkl /xxxx/Aicity/pose/B_pose.pkl --recog_result_dir ../Swin-transformer/Video-Swin-Transformer-master/submit_results_source --bottle_det_dir /xxxx/Aicity/bottle_detect --cellphone_det_dir /xxxx/Aicity/cellphone_detect --TAL_result_dir ../TAL/aicity/output --video_ids /xxxx/Aicity/data/video_ids.csv

```
### Contact
Hangyue Zhao, zhaohy21315@bupt.edu.cn

Yuchao Xiao, ycxiao@bupt.edu.cn
