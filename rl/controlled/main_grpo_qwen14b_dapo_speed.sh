#!/bin/bash
# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

# The config is optimized for 8xH200
# Assuming using vLLM >= 0.8 such that is V1 is enbaled by default
# Depends on: https://github.com/ganler/verl/tree/opt
set -eux

# IMPORTANT: checkout the specialized verl repository to the `opt-dapo-ds` branch instead of `opt`

export PYTHONPATH=$(pwd)

python -c "import rl.data"

if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi

# Tips for reducing VRAM usage
# 1. Reduce MICRO_BATCH_PER_GPU (and increase GRAD_ACCUM_STEPS accordingly)
# 2. Reduce the factor (6) in PPO_MAX_TOKEN_LEN_PER_GPU to 3

# MAIN CONFIG
DATASET=code-r1-46k-leetcode2k-kodcode-rl-codesec-78k-rl-secqa-11k-rl-safety-8k-single-turn
MODEL_PATH="outputs/purpcode-14b-ctxdistill"
MICRO_BATCH_PER_GPU=48
ROLLOUT_N_SAMPLE=8
MAX_PROMPT_LEN=2048
MAX_RESPONSE_LEN=3072
MAX_EPOCHS=1

# AUTO VALUES
ROLLOUT_N_QUERY=$((MICRO_BATCH_PER_GPU * GPUS_PER_NODE))
PPO_MAX_TOKEN_LEN_PER_GPU=$(( 8 * $(( $MAX_PROMPT_LEN + $MAX_RESPONSE_LEN )) ))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=local_data/$DATASET/train.parquet \
    data.val_files=local_data/$DATASET/test.parquet \
    data.filter_overlong_prompts=True \
    data.train_batch_size=$ROLLOUT_N_QUERY \
    +data.max_roll_factor=4 \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ROLLOUT_N_QUERY \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N_SAMPLE \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    +algorithm.filter_groups.enable=True \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='purpcode' \
    trainer.experiment_name=${DATASET}-dapo-speed \
    trainer.nnodes=1 \
    trainer.default_local_dir=./models/purpcode-rl-${DATASET}-14b-dapo-speed \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.save_freq=32 \
    trainer.test_freq=16 \
    trainer.total_epochs=$MAX_EPOCHS \
    trainer.resume_mode=auto \
    +custom_reward_function.path=./rl/grouped_reward.py \
    reward_model.reward_manager=group $@ 2>&1 | tee grpo.log
