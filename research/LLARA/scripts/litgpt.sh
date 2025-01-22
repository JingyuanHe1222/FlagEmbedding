#!/bin/bash
#SBATCH --job-name=litgit_fineweb_pythia_1b_warmup0
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


dataset="fineweb"
data_dir="/data/user_data/jingyuah/LLARA/data/pretrain"
data_path="${data_dir}/fineweb_item_litgpt.jsonl"

model_path="EleutherAI/pythia-1b"

output_dir="/data/user_data/jingyuah/LLARA/checkpoints"
exp_name="litgpt_fineweb_74k_pythia_1b_warmup_0"

n_gpus=2
bz=128
global_bz=$((n_gpus * bz))

seq_len=128

# cache dir
cd /data/user_data/jingyuah/LLARA/checkpoints

litgpt finetune_full $model_path \
  --data JSON \
  --data.json_path $data_path \
  --data.val_split_fraction 0.05 \
  --devices $n_gpus \
  --train.global_batch_size $global_bz \
  --train.micro_batch_size $bz \
  --train.epochs 10 \
  --train.max_seq_length $seq_len \
  --train.min_lr 1e-5 \
  --train.log_interval 5 \
  --train.lr_warmup_steps 1 \
  --eval.initial_validation true \
  --eval.interval 100 \
  --train.save_interval 1000 \
  --seed 42 \
  --optimizer "Adam" \
  --out_dir "${output_dir}/${exp_name}" \
  --logger_name "wandb"