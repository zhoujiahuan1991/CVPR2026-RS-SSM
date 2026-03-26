export CUDA_VISIBLE_DEVICES=0,1,2,3

GPUS=4
CONFIG="local_configs/rs-ssm/Swin/swint_realshift_w20_s10.480x480.vspw2.160k.py"

CHECKPOINT="work_dirs/vspw_st/ours_new/1030_094218/iter_160000.pth"

PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "Using port: $PORT"

time=$(date +'%m%d_%H%M%S')

log_dir="logs/check/vspw_st"

mkdir -p $log_dir

log_file="$log_dir/1030_094218_160000_3s.log"

res_dir="/data/ckpt/zhukai/TV3S/res_dirs/vspw_st/ours_new"

mkdir -p $res_dir

res_file="$res_dir/${time}.pkl"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch --out $res_file --mode=0 \
    > "$log_file" 2>&1 &

echo "Training log saved to $log_file"
