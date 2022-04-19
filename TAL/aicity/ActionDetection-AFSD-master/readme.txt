edit filepath in AFSD/configs/thumos14.yaml, AFSD/configs/thumos14_local.yaml
1.Convert video to npy file
  run:   python anet_data/video2npy.py configs/thumos14_local.yaml
2.test
   python3 AFSD/thumos14/test.py configs/thumos14.yaml --checkpoint_path=models/thumos14/checkpoint-150.ckpt --output_json=thumos14_rgb.json


