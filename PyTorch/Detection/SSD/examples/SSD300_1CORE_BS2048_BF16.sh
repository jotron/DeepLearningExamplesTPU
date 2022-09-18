# This script launches SSD300 training on a single TPUv3-8 core.
# To launch: bash examples/SSD300_1CORE_BS128.sh

export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export CHECKPOINT_DIR="./checkpoints"
export SUMMARY_PATH=$CHECKPOINT_DIR/summary.json
export XLA_USE_BF16=1
python3 main.py --backbone resnet50 --batch-size 128 --warmup 300 --data $COCO_DIR --no-cuda --seed 42 --log-interval=16 --num_cores=1 --num-workers=8 --suppress_loss_report --save $CHECKPOINT_DIR --json-summary $SUMMARY_PATH --parallel_loader --accumulation=16
