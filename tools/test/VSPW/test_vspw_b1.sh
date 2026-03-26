export CUDA_VISIBLE_DEVICES=0,1,2,3

GPUS=4
CONFIG="local_configs/rs-ssm/B1/rs-ssm_realshift_w20_s10.b1.480x480.vspw2.160k.py"

CHECKPOINT="work_dirs/vspw_b1/ours_new/1029_144051/iter_156000.pth"

PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "Using port: $PORT"

log_dir="logs/check/vspw_b1"

mkdir -p $log_dir

time=$(date +'%m%d_%H%M%S')

log_file="$log_dir/1029_144051_156000_3s.log"

res_dir="/data/ckpt/zhukai/TV3S/res_dirs/vspw_b1/ours_new"

mkdir -p $res_dir

res_file="$res_dir/${time}.pkl"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch --out $res_file --mode=0 \
    > "$log_file" 2>&1

echo "Testing log saved to $log_file"
