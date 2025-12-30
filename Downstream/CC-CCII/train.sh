export CUDA_VISIBLE_DEVICES=0
python eval.py \
--csv_list './csv' --data_dir './CC-CCII/npy/' --noamp \
--batch_size 4 \
--logdir 'logs/' \
--fold 2 \
--pretrained_dir 'your_model_path' \
--pretrained_model_name 'your_model_name'
