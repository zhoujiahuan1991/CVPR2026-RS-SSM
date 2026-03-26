export CUDA_VISIBLE_DEVICES=1,2

GPUS=2
CONFIG="local_configs/rs-ssm/Swin/swins_realshift_w20_s10.480x640.nyuv2.40k.py"

PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "Using port: $PORT"

time=$(date +'%m%d_%H%M%S')

CHECKPOINT="work_dirs/nyuv2_ss/ours_new/1105_173020/iter_40000.pth"

log_dir="logs/check/nyuv2_ss"

mkdir -p $log_dir

log_file="$log_dir/1105_173020_40000.log"

res_dir="/data/ckpt/zhukai/TV3S/res_dirs/nyuv2_ss/ours_new"

mkdir -p $res_dir

res_file="$res_dir/${time}.pkl"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch --out $res_file --mode=0 \
    > "$log_file" 2>&1 &

echo "Testing log saved to $log_file"
