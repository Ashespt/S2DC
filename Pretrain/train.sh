export CUDA_VISIBLE_DEVICES=0
export LOGDIR='Your/Log/Path'
export NUM_NODES=1
export SPACE_SIZE=96
export MASTER_PORT=13222
export FEATURE_SIZE=48
export FEATURE_DIM=768
export ROI_LARGE=384
mkdir -p $LOGDIR
cp train.sh "$LOGDIR/train.sh"
python -m torch.distributed.launch \
--nproc_per_node $NUM_NODES --master_port $MASTER_PORT main.py \
--roi_large $ROI_LARGE \
--data_type 'data_1k' \
--use_last_layer \
--queue_num 90 \
--use_geo \
--use_cl \
--num_geo_layer -1 \
--batch_size=1 \
--num_steps=100000 --lrdecay --eval_num=200 --lr=3e-4 --decay=0.1 \
--feature_size $FEATURE_SIZE --feature_dim $FEATURE_DIM --in_channels 1 --logdir $LOGDIR --roi_x $SPACE_SIZE --roi_y $SPACE_SIZE --roi_z $SPACE_SIZE \
--noamp \
--token_head \
--sinkhorn 

