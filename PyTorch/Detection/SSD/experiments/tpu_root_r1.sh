# This script launches SSD300 training on a single TPUv3-8 core.
# To launch: bash examples/SSD300_1CORE_BS128.sh

export COCO_DIR=~/COCO
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export CHECKPOINT_DIR="./checkpoints"
export SUMMARY_PATH=$CHECKPOINT_DIR/summary.json
python3 main.py --trace traces/trace_tpu_coco.csv --rule root --seed 42 --backbone resnet50 --warmup 1000 --data $COCO_DIR --no-cuda --log-interval=10 --num_cores=1 --num-workers=8 --save $CHECKPOINT_DIR --json-summary $SUMMARY_PATH --parallel_loader 
