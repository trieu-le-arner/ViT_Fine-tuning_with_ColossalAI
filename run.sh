set -xe
MODEL="google/vit-base-patch16-224"
OUTPUT_PATH="./output_model"

PLUGIN="gemini"
GRAD_CHECKPOINT=1 # 0: disable, 1: enable
GPUNUM=1

BS=16
LR="2e-4"
EPOCH=3
WEIGHT_DECAY=0.05
WARMUP_RATIO=0.3

colossalai run \
  --nproc_per_node ${GPUNUM} \
  --master_port 29505 \
  main.py \
  --model_name ${MODEL} \
  --output_path ${OUTPUT_PATH} \
  --plugin ${PLUGIN} \
  --batch_size ${BS} \
  --num_epoch ${EPOCH} \
  --learning_rate ${LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --warmup_ratio ${WARMUP_RATIO} \
  --grad_checkpoint ${GRAD_CHECKPOINT}