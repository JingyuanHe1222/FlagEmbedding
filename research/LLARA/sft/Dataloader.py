import numpy as np
import os 
from tqdm import tqdm 

from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset

import os
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments, DataCollatorWithPadding

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "If passed, will use LORA (low-rank parameter-efficient training) to train the model."}
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "The rank of lora."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."}
    )
    save_merged_lora_model: bool = field(
        default=False,
        metadata={"help": "If passed, will merge the lora modules and save the entire model."}
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "If passed, will use flash attention to train the model."}
    )
    use_slow_tokenizer: bool = field(
        default=False,
        metadata={"help": "If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library)."}
    )
    token: str = field(
        default=""
    )


@dataclass
class DataArguments:

    seq_data: str = field(
        metadata={"help": "Path to train seq data"}
    )

    item_data: str = field(
        metadata={"help": "Path to train item data"}
    )

    train_ratio: float = field(
        default=0.1, metadata={"help": "the number of training examples to use"}
    )
    eval_ratio: float = field(
        default=0.1, metadata={"help": "the number of training examples to use"}
    )
    test_ratio: float = field(
        default=0.1, metadata={"help": "the number of training examples to use"}
    )

    max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    remove_stop_words: bool = field(
        default=False
    )

    


def load_random_neagtive_items(seed, sample_num, item_num, data_num, train_data_ids):
    np.random.seed(seed)
    negative_samples = {}
    for i in tqdm(range(data_num)):
        samples = []
        for _ in range(sample_num):
            item = np.random.choice(item_num) + 1
            while item == train_data_ids[i] or item in samples: # only one item 
                item = np.random.choice(item_num) + 1
            samples.append(item)
        negative_samples[i] = samples
    return negative_samples


def load_item_name(filename):
    # load name
    blank = 0 
    item_desc = dict()
    lines = open(filename, 'r').readlines()
    for line in lines[1:]: 
        line = line.strip().split('\t')
        item_id = int(line[0])
        try: 
            item_text = line[1]
        except: 
            item_text = ""
            blank += 1
        item_desc[item_id] = item_text
    print(f"blank items: {blank}")
    return item_desc


def load_data(filename,item_desc):
    """
    Return the last item in the seq to represent item 
    """
    seqs = [] # text reps of seqs
    seq_ids = [] # id reps of seqs 
    target_ids = [] # target ids 
    lines = open(filename, 'r').readlines()
    for line in lines[1:]:
        line = line.strip().split('\t')
        # positive item 
        target_ids.append(int(line[-1]))
        # seq processing 
        for id in line[1:-1]:
            if int(id) == 0:
                break
            seq_id = int(id)
        # seq text 
        seqs.append(item_desc[seq_id])
        # seq id 
        seq_ids.append(seq_id)
    return seqs, seq_ids, target_ids


class CustomData(Dataset): 
    
    def __init__(self, 
        seq_path,
        item_path,  
        tokenizer, 
        q_max_len=256, 
        p_max_len=32, 
        sample_num=9, 
        seed=2022,
        split="train",  
        neg_file=None,
    ):

        assert split in ["train", "valid", "test"], \
            "The specified split is invalid, choose from [\"train\", \"valid\", \"test\"]"
        np.random.seed(seed)

        self.sample_num = sample_num
        self.tokenizer = tokenizer 
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len

        # get item text
        self.item_desc = load_item_name(item_path) # dict[iid]: i_text
        item_num = len(self.item_desc)
        # get seq text 
        self.seq_texts, data_ids, targets = load_data(seq_path, self.item_desc)
        assert len(self.seq_texts) == len(data_ids)
        data_num = len(self.seq_texts)

        print("using rand negatives...")
        random_neg_dict = load_random_neagtive_items(seed, 10, item_num, data_num, data_ids)
        del data_ids 

        # construct pairsc
        self.item_idx = []
        self.labels = []
        for idx, seq in enumerate(tqdm(self.seq_texts)):

            # select negative items 
            negative_ids = random_neg_dict[idx]

            negative_ids = np.random.choice(negative_ids, self.sample_num, replace=False) # random sample HN, no replacement

            # item is always pos + negs 
            self.item_idx.append([targets[idx]] + list(negative_ids))


        # create prefix and suffix for the prompt + <item> tokens to fully align 
        self.prefix = '"'
        self.suffix = '", compress the above item sequence into embeddings: <unk>'
        # max_len doesn't matter here -> assume always larger... 
        self.prefix_ids = self.tokenizer(self.prefix, truncation=True, max_length=self.q_max_len, return_tensors=None)['input_ids']
        self.suffix_ids = self.tokenizer(self.suffix, truncation=True, max_length=self.q_max_len, return_tensors=None, add_special_tokens=False)['input_ids']


    
    def create_example(self, psg, max_len): 

        # add special token 
        input_ids = self.tokenizer(
            psg,
            truncation=True,
            max_length=max_len - len(self.prefix_ids) - len(self.suffix_ids), # leave enough space 
            padding=False,
            return_tensors=None,
            add_special_tokens=False
        )

        result = dict()
        result['input_ids'] = self.prefix_ids + input_ids['input_ids'] + self.suffix_ids
        result['attention_mask'] = [1] * len(result['input_ids'])

        return result
            

    def padding(self, inputs, max_len): 
        # padding
        result = self.tokenizer.pad(
            inputs,
            padding='max_length',
            max_length=max_len,
            return_tensors='pt',
        )
        return result

    
    def __getitem__(self, idx): 

        q_inputs = self.create_example(self.seq_texts[idx], self.q_max_len)

        p_inputs = {'input_ids': [], 'attention_mask': []}
        for iidx in self.item_idx[idx] : 
            # tokenize curr item 
            result = self.create_example(self.item_desc[iidx], self.p_max_len)
            p_inputs['input_ids'].append(result['input_ids'])
            p_inputs['attention_mask'].append(result['attention_mask'])

        # paddings
        q_inputs = self.padding(q_inputs, self.q_max_len)
        p_inputs = self.padding(p_inputs, self.p_max_len)

        return {
            'query_ids': q_inputs['input_ids'],
            'query_attention_masks': q_inputs['attention_mask'], 
            'passage_ids': p_inputs['input_ids'],
            'passage_attention_masks': p_inputs['attention_mask'],
        }

  
    def __len__(self):
        return len(self.seq_texts)


def custom_collator(batch):
    """
    Custom data collator 
    Args:
        batch (List[Dict]): List of samples from the dataset.
    Returns:
        Dict[str, torch.Tensor]: Batched inputs.
    """

    input_ids_a = torch.stack([item['query_ids'] for item in batch])
    attention_mask_a = torch.stack([item['query_attention_masks'] for item in batch])

    input_ids_b = torch.cat([item['passage_ids'] for item in batch])
    attention_mask_b = torch.cat([item['passage_attention_masks'] for item in batch])

    target = torch.arange(
            input_ids_a.size(0),
            dtype=torch.long
    )
    n_neg = input_ids_b.size(0) // input_ids_a.size(0)
    target = target * n_neg
    
    return {
        'query_ids': input_ids_a,
        'query_attention_masks': attention_mask_a,
        'passage_ids': input_ids_b,
        'passage_attention_masks': attention_mask_b,
        'label': target, 
    }
    



class CustomInferenceData(Dataset): 
    
    def __init__(self, 
        data_path, 
        tokenizer, 
        q_max_len=256, 
        p_max_len=32, 
        sample_num=9, 
        seed=2022,
        name="train.txt",  
    ):
        
        self.sample_num = sample_num
        self.tokenizer = tokenizer 
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len

        # get item text
        self.item_desc = load_item_name(os.path.join(data_path, "item.txt")) # dict[iid]: i_text
        # get seq text 
        self.seq_texts, data_ids, self.targets = load_data(os.path.join(data_path, name), self.item_desc)

        # construct pairs
        self.seq_idx = []
        self.item_idx = []
        self.labels = []
        for idx, seq in enumerate(tqdm(self.seq_texts)):

            # put the positive id in
            self.item_idx.append(self.targets[idx])
            self.labels.append(1)

            # same sequennce 
            self.seq_idx.append(idx)
            
    
    def __getitem__(self, idx): 
        # find the seq index and the item index 
        seq_idx = self.seq_idx[idx] # seq index 
        item_idx = self.item_idx[idx] # item id
        # # return the tokenized inputs 
        input_text = "query: " + self.seq_texts[seq_idx] + " item: " + self.item_desc[item_idx]
        # avoid the extra dimension caused by return_tensors="pt", trainer will convert 
        tokenized = self.tokenizer(input_text, padding='max_length', truncation=True, max_length=self.q_max_len, return_tensors='pt')
        return {
            'input_ids': tokenized['input_ids'].squeeze(), 
            'attention_mask': tokenized['attention_mask'].squeeze(), 
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

  
    def __len__(self):
        return len(self.labels)




def pairwise_inference_collator(batch): 
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }