import json
import os
import argparse
import pandas as pd

def load_content(text_path, text_keys):
    item_texts = pd.read_csv(text_path, delimiter=',', dtype={'item_id': str})
    # drop nan entries 
    item_texts.dropna(subset=['title', 'description'], inplace=True)
    item_texts = item_texts[text_keys + ['item_id']]
    item_texts = item_texts.set_index('item_id').T.to_dict()
    print(f"Text Item num: {len(item_texts)}")
    return item_texts


def load_non_dup_des(text_path, text_keys):
    item_texts = pd.read_csv(text_path, delimiter=',', dtype={'item_id': str})
    # drop nan entries 
    item_texts.dropna(subset=['title', 'description'], inplace=True)
    # drop dup desc
    item_texts.drop_duplicates(subset=['description'], inplace=True)
    item_texts = item_texts[text_keys + ['item_id']]
    item_texts = item_texts.set_index('item_id').T.to_dict()
    print(f"Text Item num: {len(item_texts)}")
    return item_texts
    

def item_stats(item_texts): 

    same_title = {}
    same_tag = {}
    same_description = {}
    combinations = set()
    counts = 0
    for iid in item_texts.keys(): 
        value = item_texts[iid]
        if value['title'] not in same_title: 
            same_title[value['title']] = 0
        same_title[value['title']] += 1
        if value['description'] not in same_description: 
            same_description[value['description']] = 0
        same_description[value['description']] += 1   

        # counts of samples 
        counts += 1
        # unique item rep 
        combinations.add(f"title: {value['title']}, description: {value['description']}")


    same_title = sorted(same_title.items(), key=lambda item: item[1])
    same_tag = sorted(same_tag.items(), key=lambda item: item[1])
    same_description = sorted(same_description.items(), key=lambda item: item[1])

    print(f"{counts} items has {len(combinations)} unique reps")
    print(f"largest duplicate title: same_title: {same_title[-5:]}")
    print(f"largest duplicate tag: same_tag: {same_tag[-5:]}")
    print(f"largest duplicate description: same_description: {same_description[-5:]}")

    return combinations


def output_content_to_json_dict(item_texts, output_path): 
    with open(output_path, "w") as f:
        for item_text in item_texts: 
            json.dump({"input": item_text, "output_summarize": item_text}, f)
            f.write("\n")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", type=str, default="/data/user_data/jingyuah/HLLM_weights/data/information")
    parser.add_argument("--dataset", type=str, default="amazon_books")
    parser.add_argument("--text_keys", type=list, default=["title","description"])
    parser.add_argument("--output_path", type=str, default="/data/user_data/jingyuah/LLARA/data/pretrain")
    parser.add_argument("--output_name", type=str, default=None)
    args = parser.parse_args()

    item_texts = load_content(
        os.path.join(args.text_path, f"{args.dataset}.csv"), 
        args.text_keys, 
    )

    combinations = item_stats(item_texts)
    
    if args.output_name: 
        output_file = os.path.join(args.output_path, f"{args.output_name}.jsonl")
    else: 
        output_file = os.path.join(args.output_path, f"{args.dataset}.jsonl")
    print(f"outputing to {output_file}...")
    output_content_to_json_dict(
        combinations, 
        output_file, 
    )