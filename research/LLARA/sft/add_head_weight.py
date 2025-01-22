import os

import torch
from transformers import AutoModelForCausalLM


old_model_path = "/data/user_data/jingyuah/LLARA/checkpoints/pixel_200k_item_token_EBAE_logs/checkpoint-1400"
sft_model_path = "/data/user_data/jingyuah/LLARA/checkpoints/pixel_200k_item_token_EBAE_logs/checkpoint-1400/Pixel200K_SFT_fixed_tr0.2_lr_1e-6/checkpoint-1500"

old_model_weight = AutoModelForCausalLM.from_pretrained(old_model_path)
sft_model_weight = AutoModelForCausalLM.from_pretrained(sft_model_path)

state_dict = sft_model_weight.state_dict()

head_weight = old_model_weight.lm_head.weight

del old_model_path

state_dict['lm_head.weight'] = head_weight

dest_path = os.path.join(sft_model_path, "causal_lm_dir")
os.mkdir(dest_path)
torch.save(state_dict, os.path.join(dest_path, 'pytorch_model.bin'))