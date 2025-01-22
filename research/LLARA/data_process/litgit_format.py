import datasets

data_dir = "/data/user_data/jingyuah/LLARA/data/pretrain"
data_name = "Pixel200K"
data_path = f"{data_dir}/{data_name}_item.jsonl"

data_dir = "/data/user_data/jingyuah/LLARA/data/pretrain"
data_name = "fineweb"
data_path = "/data/datasets/hf_cache/sample/10BT"


if "10BT" in data_path: 
    print("fineweb...")
    dataset = datasets.load_dataset(data_path, split='train') 
    dataset = dataset.train_test_split(0.005)['test'] # (5.5m tokens if truncated to 128)
    dataset = dataset.map(lambda x: {'instruction': 'compress the following sentence into embedding: ', 'input': x['text'], 'output': x['text']}, remove_columns=dataset.column_names)
else: 
    dataset = datasets.load_dataset("json", data_files=data_path, split='train') # (5.8m tokens if truncated to 128)
    dataset = dataset.map(lambda x: {'instruction': 'compress the following sentence into embedding: ', 'input': x['input'], 'output': x['input']}, remove_columns=dataset.column_names)

dataset.to_json(f"{data_dir}/{data_name}_item_litgpt.jsonl")
