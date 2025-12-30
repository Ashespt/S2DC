export CUDA_VISIBLE_DEVICES=0
python eval.py \
--csv_list './csv' --data_dir './CC-CCII/npy/' --noamp \
--batch_size 4 \
--logdir 'logs/' \
--fold 2 \
--pretrained_dir 'Your_model_path' \
--pretrained_model_name 'Your_model_name'
