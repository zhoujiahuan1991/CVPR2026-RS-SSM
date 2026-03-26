export CUDA_VISIBLE_DEVICES=0,1

GPUS=2
CONFIG="local_configs/rs-ssm/B1/rs-ssm_realshift_w20_s10.b1.480x480.camvid.40k.py"

PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "Using port: $PORT"

log_dir="logs/camvid_b1/ours_new"

mkdir -p $log_dir

time=$(date +'%m%d_%H%M%S')

log_file="$log_dir/$time.log"

work_dir="work_dirs/camvid_b1/ours_new/${time}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch --work-dir $work_dir --seed 42 --deterministic \
    > "$log_file" 2>&1

echo "Training log saved to $log_file"


CHECKPOINT="work_dirs/camvid_b1/ours_new/${time}/iter_40000.pth"

log_dir="logs/test_camvid_b1/ours_new"

mkdir -p $log_dir

log_file="$log_dir/${time}_iter_40000.log"

res_dir="/data/ckpt/zhukai/TV3S/res_dirs/camvid_b1/ours_new"

mkdir -p $res_dir

res_file="$res_dir/${time}.pkl"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch --out $res_file --mode=0 \
    > "$log_file" 2>&1

echo "Testing log saved to $log_file"
