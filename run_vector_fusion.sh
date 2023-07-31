#!/bin/bash

set -e

USE_WANDB=0 # CHANGE IF YOU WANT WANDB
WANDB_USER="none"
SEED=0

EXPERIMENT=baseline

CONCEPT=bicycle
TARGET=bicycle
# WORD=BUNNY
# fonts=(KaushanScript-Regular)
ARGS="--experiment $EXPERIMENT --seed $SEED --use_wandb ${USE_WANDB} --wandb_user ${WANDB_USER}"
# COMMAND="python code/main.py $ARGS --semantic_concept "${CONCEPT}" --init_svg "${TARGET}""
CUDA_VISIBLE_DEVICES=0 python code/main.py $ARGS --semantic_concept "${CONCEPT}" --init_svg "${TARGET}"

# CUDA_VISIBLE_DEVICE = 0 python code/main.py --experiment baseline --seed 0 --semantic_concept bicycle --init_svg bicycle

