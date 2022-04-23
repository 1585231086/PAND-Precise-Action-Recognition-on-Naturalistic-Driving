<<<<<<< HEAD
=======
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
Aicity
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
python tools/batch_inference.py ./work_dirs/exp5/swin_base_patch244_window877_kinetics400_1k.py ./work_dirs/exp4/latest.pth /xxxx/Aicity/data ./label_map.txt --step 4 --device cuda:0 --view right
python tools/batch_inference.py ./work_dirs/exp6/swin_base_patch244_window877_kinetics400_1k.py ./work_dirs/exp4/latest.pth /xxxx/Aicity/data ./label_map.txt --step 4 --device cuda:0 --view rear

# TAL
cd ..
cd TAL/aicity
# Convert video to npy file
python anet_data/video2npy.py configs/thumos14_local.yaml
# inference
python3 AFSD/thumos14/test.py configs/thumos14.yaml --checkpoint_path=models/thumos14/checkpoint-150.ckpt --output_json=thumos14_rgb.json
# NMS process
python process0.py  --pose_pkl /xxxx/Aicity/pose/B_pose.pkl --recog_result_dir ../Swin-transformer/Video-Swin-Transformer-master/submit_results_source --bottle_det_dir /xxxx/Aicity/bottle_detect --cellphone_det_dir /xxxx/Aicity/cellphone_detect
python time_postp.py --input_dir thumos14_rgb.json --output_dir ./output --recog_result_dir ../../Swin-transformer/Video-Swin-Transformer-master/submit_results_source --video_ids /xxxx/Aicity/data/video_ids.csv

# post process
cd ..
cd ..
cd Post_process
python submit_zhy_new.py --pose_pkl /xxxx/Aicity/pose/B_pose.pkl --recog_result_dir ../Swin-transformer/Video-Swin-Transformer-master/submit_results_source --bottle_det_dir /xxxx/Aicity/bottle_detect --cellphone_det_dir /xxxx/Aicity/cellphone_detect --TAL_result_dir ../TAL/aicity/output

```



## Train a Model

### Iteration pipeline

MMAction2 implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

We adopt distributed training for both single machine and multiple machines.
Supposing that the server has 8 GPUs, 8 processes will be started and each process runs on a single GPU.

Each process keeps an isolated model, data loader, and optimizer.
Model parameters are only synchronized once at the beginning.
After a forward and backward pass, gradients will be allreduced among all GPUs,
and the optimizer will update model parameters.
Since the gradients are allreduced, the model parameter stays the same for all processes after the iteration.

### Training setting

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

By default we evaluate the model on the validation set after each epoch, you can change the evaluation interval by modifying the interval argument in the training config

```python
evaluation = dict(interval=5)  # This evaluate the model per 5 epoch.
```

According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or videos per GPU, e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

If you want to specify the working directory in the command, you can add an argument `--work-dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--validate` (**strongly recommended**): Perform evaluation at every k (default value is 5, which can be modified by changing the `interval` value in `evaluation` dict in each config file) epochs during the training.
- `--test-last`: Test the final checkpoint when training is over, save the prediction to `${WORK_DIR}/last_pred.pkl`.
- `--test-best`: Test the best checkpoint when training is over, save the prediction to `${WORK_DIR}/best_pred.pkl`.
- `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--gpus ${GPU_NUM}`: Number of gpus to use, which is only applicable to non-distributed training.
- `--gpu-ids ${GPU_IDS}`: IDs of gpus to use, which is only applicable to non-distributed training.
- `--seed ${SEED}`: Seed id for random state in python, numpy and pytorch to generate random numbers.
- `--deterministic`: If specified, it will set deterministic options for CUDNN backend.
- `JOB_LAUNCHER`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `LOCAL_RANK`: ID for local rank. If not specified, it will be set to 0.

Difference between `resume-from` and `load-from`:
`resume-from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load-from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

Here is an example of using 8 GPUs to load TSN checkpoint.

```shell
./tools/dist_train.sh configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py 8 --resume-from work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb/latest.pth
```

### Train with multiple machines

If you can run MMAction2 on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`. (This script also supports single machine training.)

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} [--work-dir ${WORK_DIR}]
```

Here is an example of using 16 GPUs to train TSN on the dev partition in a slurm cluster. (use `GPUS_PER_NODE=8` to specify a single slurm cluster node with 8 GPUs.)

```shell
GPUS=16 ./tools/slurm_train.sh dev tsn_r50_k400 configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py --work-dir work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb
```

You can check [slurm_train.sh](/tools/slurm_train.sh) for full arguments and environment variables.

If you have just multiple machines connected with ethernet, you can refer to
pytorch [launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility).
Usually it is slow if you do not have high speed networking like InfiniBand.

### Launch multiple jobs on a single machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs, you can set the port in commands.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

If you use launch training jobs with slurm, you need to modify `dist_params` in the config files (usually the 6th line from the bottom in config files) to set different communication ports.

In `config1.py`,

```python
dist_params = dict(backend='nccl', port=29500)
```

In `config2.py`,

```python
dist_params = dict(backend='nccl', port=29501)
```

Then you can launch two jobs with `config1.py` ang `config2.py`.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py [--work-dir ${WORK_DIR}]
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py [--work-dir ${WORK_DIR}]
```
>>>>>>> 36b64e9 (first commit)
