# This script launches SSD300 training on a single TPUv3-8 core.
# To launch from SSD main dir: bash examples/start_training.sh

# Usage ./SSD300_FP32_1GPU.sh <path to this repository> <path to dataset> <additional flags>
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export DATA_DIR="/mnt/tpu-disk/home/fot/COCO"
python3 main.py --backbone resnet50 --bs 32 --warmup 300 --data $DATA_DIR --no-cuda --seed 42 --log-interval=2
