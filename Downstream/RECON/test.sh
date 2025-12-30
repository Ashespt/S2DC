export CUDA_VISIBLE_DEVICES=0
export IN_CHANNELS=1
export MODEL_PATH='Your/model/path'
/cpfs01/projects-SSD/cfff-c7cd658afc74_SSD/pantan/env_life/miniconda3/envs/swinunetr/bin/python test.py \
--json_list='BRATS23_GLI.json' --data_dir='./' \
--feature_size=48 --in_channels $IN_CHANNELS --out_channels 4 --exp_name s2dc  --crop_foreground \
--roi_x=96 --roi_y=96 --roi_z=96 \
--pretrained_dir $MODEL_PATH --pretrained_model_name model_final_299.pt 

echo $MODEL_PATH