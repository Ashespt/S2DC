export CUDA_VISIBLE_DEVICES=0,1,2,3
/cpfs01/projects-SSD/cfff-c7cd658afc74_SSD/pantan/env_life/miniconda3/envs/swinunetr/bin/python main.py \
--json_list='BRATS23_GLI.json' --world_size 1 --crop_foreground --distributed --port 12102 --noamp \
--data_dir='./'  \
--feature_size=48 --noamp --val_every 10 --in_channels 1 --out_channels 4 --optim_lr 0.0002 \
--logdir './logs' --save_every 10 \
--roi_x=96 --roi_y=96 --roi_z=96 --batch_size=1 --max_epochs=2000 --save_checkpoint \
--use_ssl_pretrained "/Your/model/path"
