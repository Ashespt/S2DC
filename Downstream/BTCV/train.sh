export CUDA_VISIBLE_DEVICES=0
/cpfs01/projects-SSD/cfff-c7cd658afc74_SSD/pantan/env_life/miniconda3/envs/swinunetr/bin/python main.py \
--json_list='dataset_0.json' --data_dir='./data' --feature_size=48 --noamp \
--space_x=1.5 --space_y=1.5 --space_z=1.5 \
--val_every 50 --max_epochs=3000 --logdir './logs/' --optim_lr 2e-3 \
--roi_x=96 --roi_y=96 --roi_z=96  --use_checkpoint --batch_size=1 --sw_batch_size 4 --save_checkpoint \
--random_seed 3407 \
--use_ssl_pretrained='pretrained_model.pt'