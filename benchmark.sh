set -xe
MODEL="google/vit-base-patch16-224"

GRAD_CHECKPOINT="True"
GPUNUM=1
MEMCAP=0

LR="2e-4"
WEIGHT_DECAY=0.05
WARMUP_RATIO=0.3

for BS in 8 32
do
    for PLUGIN in "torch_ddp" "torch_ddp_fp16" "low_level_zero" "gemini"
    do
        colossalai run \
        --nproc_per_node ${GPUNUM} \
        --master_port 29505 \
        main.py \
        --running_mode "benchmark" \
        --model_name ${MODEL} \
        --plugin ${PLUGIN} \
        --batch_size ${BS} \
        --mem_cap ${MEMCAP} \
        --learning_rate ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_ratio ${WARMUP_RATIO} \
        --max_train_steps 20 \
        --grad_checkpoint ${GRAD_CHECKPOINT}

    done
done