_base_ = [
    '../../_base_/models/swin/swin_base.py', '../../_base_/default_runtime.py'
]
model=dict(backbone=dict(
                         patch_size=(2,4,4),
                         drop_path_rate=0.3),
           cls_head=dict(num_classes=18),
           test_cfg=dict(max_testing_views=4))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/raid/aicity/data/data'
data_root_val = '/raid/aicity/data/data'
ann_file_train = '/raid/aicity/data/annotations/k_fold/rear_24026_anno_ext6_train.txt'
ann_file_val = '/raid/aicity/data/annotations/k_fold/rear_24026_anno_ext6_val.txt'
ann_file_test = '/raid/aicity/data/annotations/k_fold/rear_24026_anno_ext6_val.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    # dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=1, num_clips=1),
    dict(type='RawFrameDecode', pad = True, crop = True, reverse = 0.0),
    dict(type='Resize', scale=(256, 256),keep_ratio=False),
    # dict(type='CenterCrop', crop_size=224),
    # dict(type='RandomResizedCrop'),
    # dict(type='CenterCrop', crop_size=224),
    # dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    # dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode', pad = True, crop = True, reverse = 0.0),
    dict(type='Resize', scale=(256, 256),keep_ratio=False),
    # dict(type='CenterCrop', crop_size=224),
    # dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=4,
        test_mode=True),
    dict(type='RawFrameDecode',pad = True, crop =True,),
    dict(type='Resize', scale=(224, 224),keep_ratio=False),
    # dict(type='ThreeCrop', crop_size=224),
    # dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=24,
    workers_per_gpu=4,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    train=dict(
        type=dataset_type,
        filename_tmpl='frame{:06}.jpg',
        with_offset=True,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        filename_tmpl='frame{:06}.jpg',
        with_offset=True,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        filename_tmpl='frame{:06}.jpg',
        with_offset=True,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(type='AdamW', lr=1e-3*1.5, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)
total_epochs = 10

# runtime settings
checkpoint_config = dict(interval=1)
load_from = './work_dirs/k400_swin_base_patch244_window877_div_24026/exp0/latest.pth'
work_dir = './work_dirs/k400_swin_base_patch244_window877_div_24026/exp2'
find_unused_parameters = False


# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=8,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
