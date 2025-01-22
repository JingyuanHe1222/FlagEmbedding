#!/bin/bash
#SBATCH --job-name=fineweb_128_pythia1b_5e-6
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --partition=general  
#SBATCH --mem=64G 
#SBATCH --gres=gpu:A6000:2

#SBATCH --exclude=babel-4-[1,17,25,33,37],babel-1-23 

#SBATCH --time=48:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"

eval "$(conda shell.bash hook)"
conda activate llara


# pretrain_model="/data/user_data/jingyuah/HLLM_weights/checkpoints/TinyLlama-1.1B-intermediate-step-1431k-3T"
pretrain_model="EleutherAI/pythia-1b"
output_dir="/data/user_data/jingyuah/LLARA/checkpoints"


# dataset="amazon_books"
# data_path="/data/user_data/jingyuah/LLARA/data/pretrain/${dataset}_item_96k.jsonl" # toy_pretrain_data.jsonl

dataset="fineweb"
data_path="/data/datasets/hf_cache/sample/10BT"

expt_name="${dataset}_pythia_1b_lr_5e-6"
output_path="${output_dir}/${expt_name}"
mkdir -p $output_path

bz=128
n_gpus=2

cd /home/jingyuah/FlagEmbedding/research/LLARA/pretrain

export WANDB_PROJECT="LLARA_pretrain"
# token: (`str` or *bool*, *optional*): token to use as HTTP authorization for remote files. passing `token=True` is required when you want to use a private model.
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node $n_gpus --master_port=24567 --rdzv_endpoint=localhost:29400 \
    ntp_run.py \
        --output_dir $output_path \
        --model_name_or_path $pretrain_model \
        --train_data $data_path \
        --learning_rate 5e-6 \
        --num_train_epochs 10 \
        --per_device_train_batch_size $bz \
        --per_device_eval_batch_size $bz \
        --evaluation_strategy "steps" \
        --metric_for_best_model "eval_loss" \
        --load_best_model_at_end True \
        --gradient_accumulation_steps 1 \
        --dataloader_drop_last True \
        --cutoff_len 128 \
        --logging_steps 5 \
        --save_steps 100 \
        --eval_steps 100 \
        --save_total_limit 2 \
        --gradient_checkpointing \
        --ddp_find_unused_parameters False \
        --use_flash_attn False \
        --deepspeed ../stage1.json \
        --warmup_ratio 0.1 \
        --remove_stop_words True \
        --use_lora False \
        --bf16 \
        --cache_dir $output_path \
        --token True \
        --report_to wandb 