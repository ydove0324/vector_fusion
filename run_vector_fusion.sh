#!/bin/bash

set -e

USE_WANDB=1 # CHANGE IF YOU WANT WANDB
WANDB_USER="none"
SEED=147
OPTIM_PATH=128

EXPERIMENT=baseline

CONCEPT=bicycle
TARGET=("an owl standing on a wire" "bicycle")
USE_IMG_LOCAL=true
USE_SVG_LOCAL=true
# WORD=BUNNY
# fonts=(KaushanScript-Regular)
ARGS="--experiment $EXPERIMENT --seed $SEED --use_wandb ${USE_WANDB} --wandb_user ${WANDB_USER}"
# COMMAND="python code/main.py $ARGS --semantic_concept "${CONCEPT}" --init_svg "${TARGET}""

for t in "${TARGET[@]}"; do
    echo $t
    export CUDA_VISIBLE_DEVICES='0,1'
    python code/main.py $ARGS --semantic_concept "$t" --optim_path $OPTIM_PATH --use_img_local $USE_IMG_LOCAL --use_svg_local $USE_SVG_LOCAL
done
