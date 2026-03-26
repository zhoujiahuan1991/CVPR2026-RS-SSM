export CUDA_VISIBLE_DEVICES=2,3,4,5

GPUS=4
CONFIG="local_configs/rs-ssm/B1/rs-ssm_realshift_w20_s10.b1.1024x1024.city.160k.py"

CHECKPOINT="work_dirs/city_b1/ours_new/1104_025323/iter_160000.pth"

PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "Using port: $PORT"

log_dir="logs/check/city_b1"


mkdir -p $log_dir

time=$(date +'%m%d_%H%M%S')

log_file="$log_dir/1104_025323_160000_s.log"

res_dir="/data/ckpt/zhukai/TV3S/res_dirs/city_b1/ours_new"

mkdir -p $res_dir

res_file="$res_dir/${time}.pkl"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch \
    --out $res_file --mode=0 \
    > "$log_file" 2>&1

echo "Testing log saved to $log_file"
