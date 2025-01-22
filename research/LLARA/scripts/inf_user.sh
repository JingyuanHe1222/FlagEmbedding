#!/bin/bash
#SBATCH --job-name=inf_sim_item_only 
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --partition=general  
#SBATCH --mem=128G 
#SBATCH --gres=gpu:A6000:1

#SBATCH --time=48:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"


eval "$(conda shell.bash hook)"
conda activate llara


echo "Started"

# python /home/jingyuah/FlagEmbedding/research/LLARA/pretrain/inf_sim.py

python /home/jingyuah/FlagEmbedding/research/LLARA/pretrain/item_inf_sim.py 

echo "Eneded"