import torch
from transformers import AutoModel, AutoTokenizer, LlamaModel
from torch.utils.data import DataLoader
import datasets

import faiss 
import numpy as np
from tqdm import tqdm 

import pickle

def recall(pos_index, pos_len):
    return np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)

def ndcg(pos_index,pos_len):
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)
    iranks = np.zeros_like(pos_index, dtype=float)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]
    ranks = np.zeros_like(pos_index, dtype=float)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    return result

def get_metrics_dict(rank_indices, n_seq, n_item, Ks):
    rank_indices = torch.tensor(rank_indices)
    pos_matrix = torch.eye(n_seq, n_item, dtype=torch.int)
    pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
    pos_idx = torch.gather(pos_matrix, dim=1, index=rank_indices)
    pos_idx = pos_idx.to(torch.bool).cpu().numpy()
    pos_len_list = pos_len_list.squeeze(-1).cpu().numpy()
    recall_result = recall(pos_idx, pos_len_list)
    avg_recall_result = recall_result.mean(axis=0)
    ndcg_result = ndcg(pos_idx, pos_len_list)
    avg_ndcg_result = ndcg_result.mean(axis=0)
    metrics_dict = {}
    for k in Ks:
        metrics_dict[k] = {}
        metrics_dict[k]["recall"] = round(avg_recall_result[k - 1], 4)
        metrics_dict[k]["ndcg"] = round(avg_ndcg_result[k - 1], 4)
    return metrics_dict


def get_query_inputs(queries, tokenizer, max_length=512):
    prefix = '"'
    suffix = '", predict the next item embedding: <unk>'
    prefix_ids = tokenizer(prefix, return_tensors=None)['input_ids']
    suffix_ids = tokenizer(suffix, return_tensors=None)['input_ids'][1:]
    queries_inputs = []
    for query in queries:
        inputs = tokenizer(query,
                           return_tensors=None,
                           max_length=max_length,
                           truncation=True,
                           add_special_tokens=False)
        inputs['input_ids'] = prefix_ids + inputs['input_ids'] + suffix_ids
        inputs['attention_mask'] = [1] * len(inputs['input_ids'])
        queries_inputs.append(inputs)
    return tokenizer.pad(
            queries_inputs,
            padding=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            return_tensors='pt',
        )

def get_passage_inputs(passages, tokenizer, max_length=512):
    prefix = '"'
    suffix = '", summarize the above user sequence within eight words: <s1><s2><s3><s4><s5><s6><s7><s8>'
    prefix_ids = tokenizer(prefix, return_tensors=None)['input_ids']
    suffix_ids = tokenizer(suffix, return_tensors=None)['input_ids'][1:]
    passages_inputs = []
    for passage in passages:
        inputs = tokenizer(passage,
                           return_tensors=None,
                           max_length=max_length,
                           truncation=True,
                           add_special_tokens=False)
        inputs['input_ids'] = prefix_ids + inputs['input_ids'] + suffix_ids
        inputs['attention_mask'] = [1] * len(inputs['input_ids'])
        passages_inputs.append(inputs)
    return tokenizer.pad(
            passages_inputs,
            padding=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            return_tensors='pt',
        )

def custom_collator(batch):
    return {
        'query': [query['input'] for query in batch],
        'item': [query['output_predict'] for query in batch],
    }


# Load the tokenizer and model
model_path="/data/user_data/jingyuah/HLLM_weights/checkpoints/TinyLlama-1.1B-intermediate-step-1431k-3T"
# '/data/user_data/jingyuah/LLARA/checkpoints/pixel_200k_user_2/checkpoint-14000'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=True)
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
model = AutoModel.from_pretrained(model_path)

data_path = "/data/user_data/jingyuah/LLARA/data/pretrain/Pixel200K_user.jsonl"
dataset = datasets.load_dataset("json", data_files=data_path, split='train')
dataset = dataset.train_test_split(test_size=0.05, seed=42)

eval_dataset = dataset["test"]
eval_dataset = eval_dataset.train_test_split(test_size=0.1, seed=42)['test']

evalset = DataLoader(dataset=eval_dataset,
                      batch_size=8,
                      shuffle=False ,
                      collate_fn=custom_collator, 
                    )

query_embeddings = []
passage_embeddings = []

model.eval()

num_query_token = 1
num_passage_token = 1

with torch.no_grad():

    for idx, batch in tqdm(enumerate(evalset), total=len(evalset)): 

        query_input = get_query_inputs(batch['query'], tokenizer, 1024)
        passage_input = get_passage_inputs(batch['item'], tokenizer, 128)

        # compute query embedding
        query_outputs = model(**query_input, return_dict=True, output_hidden_states=True)
        query_embedding = query_outputs.hidden_states[-1][:, -num_query_token:, :].detach().cpu().squeeze()
        query_embedding = torch.nn.functional.normalize(query_embedding, dim=-1)
        query_embeddings.append(query_embedding.numpy())

        # compute passage embedding
        passage_outputs = model(**passage_input, return_dict=True, output_hidden_states=True)
        passage_embedding = passage_outputs.hidden_states[-1][:, -num_passage_token:, :].detach().cpu()
        passage_embedding = torch.mean(passage_embedding, dim=1)
        passage_embedding = torch.nn.functional.normalize(passage_embedding, dim=-1)
        passage_embeddings.append(passage_embedding.numpy())


query_embeddings = np.concatenate(query_embeddings, 0)
passage_embeddings = np.concatenate(passage_embeddings, 0)

with open('/data/user_data/jingyuah/LLARA/checkpoints/pixel_200k_user_2/eval_3000_embed.pkl', 'wb') as f:
    pickle.dump((query_embeddings, passage_embeddings), f)

# compute similarity score
index = faiss.IndexFlatIP(passage_embeddings.shape[1])
index.add(passage_embeddings)
D, I = index.search(query_embeddings, k=10)

metrics = get_metrics_dict(I, query_embeddings.shape[0], passage_embeddings.shape[0], [1, 10])

output = 'Recall@1：{:.4f}, Recall@10:{:.4f}, NDCG@10:{:.4f}'.format(
            metrics[1]['recall'], metrics[10]['recall'],
            metrics[1]['ndcg'],)

print(output)


    
    

