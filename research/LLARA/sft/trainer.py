from typing import Dict, List, Union, Optional 

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach


class SFTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.n_neg = 10
        self.label_names = ['label']

    def compute_loss(self, model, inputs, return_outputs=False):
        # extract query item embedding 
        q_ids = inputs['query_ids'].to(model.device)
        q_masks = inputs['query_attention_masks'].to(model.device)

        p_ids = inputs['passage_ids'].to(model.device)
        p_masks = inputs['passage_attention_masks'].to(model.device)

        # last token is the <unk> item token 
        q_hiddens = model(q_ids, q_masks).last_hidden_state[:, -1, :]
        p_hiddens = model(p_ids, p_masks).last_hidden_state[:, -1, :]

        # do cosine sim 
        q_hiddens = torch.nn.functional.normalize(q_hiddens, dim=-1)
        p_hiddens = torch.nn.functional.normalize(p_hiddens, dim=-1)
        scores = torch.matmul(q_hiddens, p_hiddens.transpose(0, 1))

        target = inputs['label'].to(model.device)

        # Compute pairwise ranking loss
        loss = self.loss_fn(scores, target)

        return (loss, {"q_embed": q_hiddens, "p_embed": p_hiddens}) if return_outputs else loss
    
