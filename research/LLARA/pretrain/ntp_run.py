import argparse
import os 
from pathlib import Path

import datasets 

from dataclasses import dataclass, field
import logging
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    HfArgumentParser, 
    set_seed, 
    AutoConfig, 
)

from arguments import ModelArguments, DataArguments, PretrainTrainingArguments as TrainingArguments


logger = logging.getLogger(__name__)



class CustomDataCollatorForPadding(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, max_length=512):
        super().__init__(tokenizer=tokenizer, mlm=False)  # Set mlm to False to disable masking
        self.max_length = max_length

    def __call__(self, examples):
        # Tokenize and pad to custom max_length
        batch = self.tokenizer.pad(
            examples,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create input_ids and labels without masking
        input_ids = batch['input_ids']
        labels = input_ids.clone()
        labels[batch['attention_mask'] == 0] = -100 # ignored in ce loss 
        batch['labels'] = labels

        return batch




def main(): 
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments


    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=1,
        cache_dir=model_args.cache_dir,
    )
    logger.info('Config: %s', config)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        use_flash_attention_2=True if model_args.use_flash_attn else False,
        token=model_args.token,
        cache_dir=model_args.cache_dir,
        config=config,
        trust_remote_code=True, 
    )

    # model = AutoModelForCausalLM.from_config(
    #     config, 
    #     torch_dtype=torch.bfloat16, 
    #     use_flash_attention_2=True if model_args.use_flash_attn else False,
    # )



    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        cache_dir=model_args.cache_dir,
        use_fast=True, # required for GPTNeoX
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # load dadtdaset and map it to tokenized ver
    if "10BT" in data_args.train_data: 
        logger.info("fineweb...")
        dataset = datasets.load_dataset(data_args.train_data, split='train') 
        dataset = dataset.train_test_split(0.0035)['test'] # (5.5m tokens if truncated to 128)
    else: 
        dataset = datasets.load_dataset("json", data_files=data_args.train_data, split='train',
                                                    cache_dir=data_args.cache_path) # (5.8m tokens if truncated to 128)
        # remove cols 
        dataset = dataset.map(lambda x: {'text': x['input']}, remove_columns=dataset.column_names)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", max_length=data_args.cutoff_len, truncation=True)
    dataset = dataset.map(tokenize_function, batched=True)

    dataset_split = dataset.train_test_split(test_size=0.05, seed=training_args.seed)

    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    logger.info(f"train dataset length: {len(train_dataset)}")
    logger.info(f"eval dataset length: {len(eval_dataset)}")

    # labels = inputs, pad -100 ignored 
    data_collator = CustomDataCollatorForPadding(
        tokenizer=tokenizer, 
        max_length=data_args.cutoff_len,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    if training_args.deepspeed:
        trainer.deepspeed.save_checkpoint(training_args.output_dir)



if __name__ == "__main__": 
    main(); 