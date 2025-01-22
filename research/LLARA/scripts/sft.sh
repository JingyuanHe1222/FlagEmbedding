#!/bin/bash
#SBATCH --job-name=sft_fixed_prompt
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --partition=general  
#SBATCH --mem=64G 
#SBATCH --gres=gpu:A6000:4

#SBATCH --exclude=babel-4-[1,17,25,33,37],babel-1-23,babel-0-37

#SBATCH --time=48:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"

eval "$(conda shell.bash hook)"
conda activate llara


pretrain_model="/data/user_data/jingyuah/LLARA/checkpoints/pixel_200k_item_token_EBAE_logs/checkpoint-1400"
output_dir="/data/user_data/jingyuah/LLARA/checkpoints/pixel_200k_item_token_EBAE_logs/checkpoint-1400/"

dataset="Pixel200K"
seq_data="/data/user_data/jingyuah/LLARA/data/pretrain/sft/${dataset}_seq_id.jsonl" 
item_data="/data/user_data/jingyuah/LLARA/data/pretrain/sft/${dataset}_seq_id_iid_to_text.jsonl" 

train_ratio=0.2
eval_ratio=0.05

expt_name="${dataset}_SFT_fixed_tr${train_ratio}_lr_1e-6"
output_path="${output_dir}/${expt_name}"
mkdir -p $output_path

bz=32
n_gpus=4

cd /home/jingyuah/FlagEmbedding/research/LLARA/sft

export WANDB_PROJECT="LLARA_sft"
# token: (`str` or *bool*, *optional*): token to use as HTTP authorization for remote files. passing `token=True` is required when you want to use a private model.
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node $n_gpus --master_port=24567 --rdzv_endpoint=localhost:29400 \
    sft.py \
        --output_dir $output_path \
        --model_name_or_path $pretrain_model \
        --seq_data $seq_data \
        --item_data $item_data \
        --max_len 256 \
        --learning_rate 1e-6 \
        --warmup_ratio 0.1 \
        --num_train_epochs 5 \
        --train_ratio $train_ratio \
        --eval_ratio $eval_ratio \
        --per_device_train_batch_size $bz \
        --per_device_eval_batch_size $bz \
        --evaluation_strategy "steps" \
        --metric_for_best_model "eval_loss" \
        --load_best_model_at_end True \
        --gradient_accumulation_steps 1 \
        --dataloader_drop_last True \
        --logging_steps 5 \
        --save_steps 250 \
        --eval_steps 250 \
        --eval_on_start True \
        --save_total_limit 2 \
        --gradient_checkpointing \
        --ddp_find_unused_parameters False \
        --remove_unused_columns False \
        --use_flash_attn False \
        --deepspeed ../stage1.json \
        --remove_stop_words True \
        --use_lora False \
        --bf16 \
        --token True \
        --report_to wandb 