import json
import os
import argparse
import pandas as pd
import numpy as np

def load_content(text_path, text_keys):
    item_texts = pd.read_csv(text_path, delimiter=',', dtype={'item_id': str})
    item_texts = item_texts[text_keys + ['item_id']]
    item_texts = item_texts.set_index('item_id').T.to_dict()
    print(f"Text Item num: {len(item_texts)}")
    return item_texts

def output_content_to_json_dict(dataload, output_path): 
    with open(output_path, "w") as f:
        for key in dataload.sequences: 
            user_seq_text, next_item = dataload.sequences[key]
            json.dump({"input": user_seq_text, "output_summarize": user_seq_text, "output_predict": next_item}, f)
            f.write("\n")

def output_ids_to_json_dict(dataload, seq_output_path, item_output_path): 

    # output seq file
    with open(seq_output_path, 'w') as f:
        f.write("user_id\tseq\ttarget\n")
        for key in dataload.sequences: 
            output_seq = [key] + dataload.sequences[key]
            output_seq = map(str, output_seq)
            f.write("\t".join(output_seq))
            f.write("\n")

    # output item file 
    with open(item_output_path, 'w') as f:
        f.write("item_id\titem_name\n")
        for iid in dataload.item_id_to_text: 
            f.write(f"{iid}\t{dataload.item_id_to_text[iid]}")
            f.write("\n")
            

class Data(): 
    def __init__(self, data_path, dataset, text_path, text_keys, data_split=None, item_data=None):
        self.dataset_path = data_path
        self.text_keys = text_keys
        self.text_path = text_path
        self.dataset_name = dataset
        self.data_split = data_split
        self.item_data = item_data
        self._from_scratch()


    def _from_scratch(self):
        print(f'Loading {self.__class__} from scratch with {self.data_split = }.')
        self._load_inter_feat(self.dataset_name, self.dataset_path, self.item_data)
        self.env = load_content(os.path.join(self.text_path, f"{self.dataset_name}.csv"), self.text_keys)
        self._data_processing()
        self.build()


    def _load_inter_feat(self, token, dataset_path, item_data=None):
        inter_feat_path = os.path.join(dataset_path, f'{token}.csv')
        if not os.path.isfile(inter_feat_path):
            raise ValueError(f'File {inter_feat_path} not exist.')

        df = pd.read_csv(
            inter_feat_path, delimiter=',', dtype={'item_id': str, 'user_id': str, 'timestamp': int}, header=0, names=['item_id', 'user_id', 'timestamp']
        )
        self.inter_feat = df
        print(f'Interaction feature loaded successfully from [{inter_feat_path}].')

        if item_data: # do not enter 
            item_data_path = os.path.join(dataset_path, f'{item_data}.csv')
            item_df = pd.read_csv(
                item_data_path, delimiter=',', dtype={'item_id': str, 'user_id': str, 'timestamp': int}, header=0, names=['item_id', 'user_id', 'timestamp']
            )
            self.item_feat = item_df
            print(f'Item feature loaded successfully from [{item_data}].')


    def _data_processing(self):
        self.id2token = {}
        self.token2id = {}
        remap_list = ['user_id', 'item_id']
        for feature in remap_list:
            if feature == 'item_id' and self.item_data:
                feats = self.item_feat[feature]
                feats_raw = self.inter_feat[feature]
            else:
                feats = self.inter_feat[feature]
            new_ids_list, mp = pd.factorize(feats)
            mp = ['[PAD]'] + list(mp)
            token_id = {t: i for i, t in enumerate(mp)}
            if feature == 'item_id' and self.item_data:
                _, raw_mp = pd.factorize(feats_raw)
                for x in raw_mp:
                    if x not in token_id:
                        token_id[x] = len(token_id)
                        mp.append(x)
            mp = np.array(mp)

            self.id2token[feature] = mp
            self.token2id[feature] = token_id
            self.inter_feat[feature] = self.inter_feat[feature].map(token_id)

        self.user_num = len(self.id2token['user_id'])
        self.item_num = len(self.id2token['item_id'])
        self.inter_num = len(self.inter_feat)
        self.uid_field = 'user_id'
        self.iid_field = 'item_id'
        self.user_seq = None
        self.train_feat = None
        self.feat_name_list = ['inter_feat']  # self.inter_feat

        self.id2token_map = self.id2token['item_id']


    def sort(self, by, ascending=True):

        if isinstance(self.inter_feat, pd.DataFrame):
            self.inter_feat.sort_values(by=by, ascending=ascending, inplace=True)

        else:
            if isinstance(by, str):
                by = [by]

            if isinstance(ascending, bool):
                ascending = [ascending]

            if len(by) != len(ascending):
                if len(ascending) == 1:
                    ascending = ascending * len(by)
                else:
                    raise ValueError(f'by [{by}] and ascending [{ascending}] should have same length.')
            for b, a in zip(by[::-1], ascending[::-1]):
                index = np.argsort(self.inter_feat[b], kind='stable')
                if not a:
                    index = index[::-1]
                for k in self.inter_feat:
                    self.inter_feat[k] = self.inter_feat[k][index]


    def _grouped_index(self, group_by_list):
        index = {}
        for i, key in enumerate(group_by_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        return index
    

    def process_item(self, item):
        if item != self.id2token_map[0] and item not in self.env:
            print(f"{item} not in self.env")
        item_i = self.env.get(item, {})
        text_str = ""
        if len(item_i):
            text_str = ""
            for key in self.text_keys:
                value = item_i[key]
                if value and str(value) != 'nan':
                    text_str += f" {key}: {value}"
        return text_str
    
    def map_item_id_to_iid_text(self): 
        self.item_id_to_text = {0: self.id2token_map[0]}
        for i in range(1, self.id2token_map.shape[0]): 
            item_str = self.process_item(self.id2token_map[i]) 
            self.item_id_to_text[i] = item_str


    def build(self):
        print(f"build {self.dataset_name} dataload")
        # sort by time and organize by users 
        self.sort(by='timestamp')
        user_list = self.inter_feat['user_id'].values
        item_list = self.inter_feat['item_id'].values
        timestamp_list = self.inter_feat['timestamp'].values
        grouped_index = self._grouped_index(user_list)

        user_seq = {}
        time_seq = {}
        for uid, index in grouped_index.items():
            user_seq[uid] = item_list[index]
            time_seq[uid] = timestamp_list[index]

        self.user_seq = user_seq
        self.time_seq = time_seq
        self.map_item_id_to_iid_text()


    def build_text(self): 
        self.sequences = {}
        for key in self.user_seq: 
            user_seq = list(self.user_seq[key])
            # ignore those with no history 
            if len(user_seq) <= 1: 
                continue

            next_item = self.item_id_to_text[user_seq[-1]]

            # keep more recent historical interactions 
            user_seq = user_seq[:-1] # get rid of "target" item 
            user_seq.reverse()
            word_count = 0
            seq_text = []
            # leave last item out as the prediction 
            for i in range(len(user_seq) - 1): 
                curr_item = self.item_id_to_text[user_seq[i]]
                word_count += len(curr_item.split()) + 1 # add space afterward 
                # control seq length by getting most recent interactions
                if word_count > 1024: 
                    break                 
                seq_text.append(curr_item)
            # reverse to maintain sequential order
            seq_text.reverse()
            seq_text = "; ".join(seq_text)
            
            self.sequences[key] = (seq_text, next_item)


    def build_ids(self):
    
        self.sequences = {}
        for key in self.user_seq: 
            user_seq = list(self.user_seq[key])
            # ignore those with no history 
            if len(user_seq) <= 1: 
                continue

            next_item = user_seq[-1]

            # keep more recent historical interactions 
            user_seq = user_seq[:-1] # get rid of "target" item 
            user_seq.reverse()
            word_count = 0
            seq_ids = []
            # leave last item out as the prediction 
            for i in range(len(user_seq) - 1): 
                curr_item = self.item_id_to_text[user_seq[i]]
                word_count += len(curr_item.split()) + 1 # add space afterward 
                # control seq length by getting most recent interactions
                if word_count > 1024 or len(seq_ids) >= 10: 
                    break                 
                seq_ids.append(user_seq[i])
            # reverse to maintain sequential order
            seq_ids.reverse()

            # prepend 0
            if len(seq_ids) < 10: 
                seq_ids.extend([0] * (10 - len(seq_ids)))

            # add target item at the end 
            seq_ids.append(next_item)
            
            self.sequences[key] = seq_ids







if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", type=str, default="/data/user_data/jingyuah/HLLM_weights/data/information")
    parser.add_argument("--data_path", type=str, default="/data/user_data/jingyuah/HLLM_weights/data/dataset")
    parser.add_argument("--dataset", type=str, default="Pixel200K")
    parser.add_argument("--text_keys", type=list, default=["title","tag","description"])
    parser.add_argument("--output_path", type=str, default="/data/user_data/jingyuah/LLARA/data/pretrain")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--id_seq", action='store_true')
    args = parser.parse_args()

    dataload = Data(
        args.data_path, 
        args.dataset, 
        args.text_path, 
        args.text_keys
    )

    dataload.build()

    if args.output_name: 
        output_path = os.path.join(args.output_path, f"{args.output_name}.jsonl") 
    else: 
        output_path = os.path.join(args.output_path, f"{args.dataset}_user.jsonl") 

    if args.id_seq: 
        # id outputs 
        dataload.build_ids()
        output_ids_to_json_dict(
            dataload, 
            seq_output_path=output_path, 
            item_output_path=os.path.join(args.output_path, f"{args.output_name}_iid_to_text.jsonl") 
        )


    else: 
        # textual outputs 
        dataload.build()

        output_content_to_json_dict(
            dataload, 
            output_path, 
        )




