#!/bin/bash
#SBATCH --job-name=user_pixel_200k
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --partition=general   
#SBATCH --mem=64G 
#SBATCH --gres=gpu:A6000:4

#SBATCH --time=48:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"

eval "$(conda shell.bash hook)"
conda activate llara


pretrain_model="/data/user_data/jingyuah/HLLM_weights/checkpoints/TinyLlama-1.1B-intermediate-step-1431k-3T"
output_dir="/data/user_data/jingyuah/LLARA/checkpoints"

expt_name="pixel_200k_user_token"
output_path="${output_dir}/${expt_name}"
mkdir -p $output_path

data_path="/data/user_data/jingyuah/LLARA/data/pretrain/Pixel200K_user.jsonl" # toy_pretrain_data.jsonl

bz=16
n_gpus=1

cd /home/jingyuah/FlagEmbedding/research/LLARA/pretrain

export WANDB_PROJECT="LLARA_pretrain"
# token: (`str` or *bool*, *optional*): token to use as HTTP authorization for remote files. passing `token=True` is required when you want to use a private model.
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node $n_gpus --master_port=25678 --rdzv_endpoint=localhost:29400 \
    run.py \
        --output_dir $output_path \
        --model_name_or_path $pretrain_model \
        --train_data $data_path \
        --learning_rate 1e-5 \
        --num_train_epochs 5 \
        --per_device_train_batch_size $bz \
        --per_device_eval_batch_size $bz \
        --evaluation_strategy "steps" \
        --metric_for_best_model "eval_loss" \
        --load_best_model_at_end True \
        --save_total_limit 2 \
        --gradient_accumulation_steps 1 \
        --dataloader_drop_last True \
        --cutoff_len 1024 \
        --logging_steps 50 \
        --save_steps 1000 \
        --eval_steps 1000 \
        --gradient_checkpointing \
        --ddp_find_unused_parameters False \
        --use_flash_attn False \
        --deepspeed ../stage1.json \
        --warmup_ratio 0.1 \
        --remove_stop_words True \
        --use_lora False \
        --bf16 \
        --cache_dir $output_path \
        --cache_path $output_path \
        --token True \
        --report_to wandb 