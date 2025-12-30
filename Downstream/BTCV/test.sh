export CUDA_VISIBLE_DEVICES=0
export MODEL_DIR='./'

/cpfs01/projects-SSD/cfff-c7cd658afc74_SSD/pantan/env_life/miniconda3/envs/swinunetr/bin/python val.py \
    --json_list='dataset_0.json' --data_dir='./data' --feature_size=48 --infer_overlap 0.75 \
    --pretrained_model_name="your/pretrained/model.pt" --pretrained_dir $MODEL_DIR \
    --space_x=1.5 --space_y=1.5 --space_z=1.5 --exp_name s2dc
