#!/bin/bash

set -e

USE_WANDB=0 # CHANGE IF YOU WANT WANDB
WANDB_USER="none"
SEED=331

EXPERIMENT=baseline

CONCEPT=bicycle
TARGET="The great pyramids"
# WORD=BUNNY
# fonts=(KaushanScript-Regular)
ARGS="--experiment $EXPERIMENT --seed $SEED --use_wandb ${USE_WANDB} --wandb_user ${WANDB_USER}"
# COMMAND="python code/main.py $ARGS --semantic_concept "${CONCEPT}" --init_svg "${TARGET}""
CUDA_VISIBLE_DEVICES=0 python code/main.py $ARGS --semantic_concept "$TARGET"

# CUDA_VISIBLE_DEVICE = 0 python code/main.py --experiment baseline --seed 0 --semantic_concept bicycle --init_svg bicycle
# python code/main.py --experiment baseline --seed 40605 --semantic_concept "The great pyramids"

