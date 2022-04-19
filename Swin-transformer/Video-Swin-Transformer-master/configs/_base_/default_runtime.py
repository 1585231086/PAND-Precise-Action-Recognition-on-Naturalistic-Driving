checkpoint_config = dict(interval=1)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from='/disk1/zhy/AIcity/mmaction2/Video-Swin-Transformer-master/ckp/swin_base_patch244_window877_kinetics400_1k.pth'
# load_from = '/disk1/zhy/AIcity/mmaction2/Video-Swin-Transformer-master/ckp/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb_20200728-145f1097.pth'
resume_from = None
workflow = [('train', 1)]
