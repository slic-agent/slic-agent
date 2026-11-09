#!/bin/bash
# Supervised fine-tuning on multi-turn chat trajectories (parquet with a `messages` column, scripts/train/sft/README.md) with the
# settings of the paper's SFT runs: Qwen3.5-4B, 4 epochs, batch 8 with 2 sequences per GPU per micro-batch, lr 1e-5 with
# 10 % warm-up, 12288-token sequences, FSDP2 with Liger kernels, HF-format checkpoints. The <think>-prefill chat template of
# the paper's checkpoints is the override +model.chat_template_path=scripts/train/sft/chat_template_think_prefill.jinja.
# `slic train --config configs/sft/<name>.yaml` runs this script with the config's files and overrides appended;
# "$@" comes last and wins. NUM_GPUS (default: every visible GPU) is the number of processes.
# Outputs: outputs/checkpoints/<trainer.experiment_name>/.
cd "$(dirname "$0")/../../.." || exit 1
torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS:-gpu}" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=data/sft/train.parquet \
    data.val_files=data/sft/test.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    +data.ignore_input_ids_mismatch=true \
    data.max_length=12288 \
    data.truncation=left \
    data.micro_batch_size_per_gpu=2 \
    data.train_batch_size=8 \
    optim.lr=1e-5 \
    optim.warmup_steps_ratio=0.1 \
    model.partial_pretrain=Qwen/Qwen3.5-4B \
    model.strategy=fsdp2 \
    model.load_dtype=bfloat16 \
    model.use_liger=True \
    model.enable_gradient_checkpointing=True \
    model.fsdp_config.cpu_offload=false \
    model.fsdp_config.offload_params=false \
    trainer.experiment_name=sft_qwen3.5_4b \
    trainer.project_name=slic-agent-sft \
    "trainer.logger=['console']" \
    'trainer.default_local_dir=outputs/checkpoints/${trainer.experiment_name}' \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=4 \
    +trainer.log_entropy=true \
    +trainer.checkpoint_format=hf \
    "$@"
