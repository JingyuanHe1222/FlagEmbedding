import argparse
import os 
from pathlib import Path

import logging
import torch

from dataclasses import dataclass, field
from torch.utils.data import random_split

from transformers import (
    LlamaModel, 
    AutoTokenizer,
    HfArgumentParser, 
    TrainingArguments, 
    set_seed, 
)

from Dataloader import DataArguments, ModelArguments  
from Dataloader import CustomData, custom_collator

from trainer import SFTTrainer


logger = logging.getLogger(__name__)



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

    model = LlamaModel.from_pretrained(
        model_args.model_name_or_path,
        use_flash_attention_2=True if model_args.use_flash_attn else False,
        token=model_args.token,
        trust_remote_code=True, 
    )


    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        use_fast=True, # required for GPTNeoX
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    dataset = CustomData(
        seq_path=data_args.seq_data, 
        item_path=data_args.item_data, 
        tokenizer=tokenizer, 
        q_max_len=data_args.max_len, 
        p_max_len=data_args.max_len, 
        sample_num=9, 
        seed=training_args.seed, 
        split="train"
    )

    train_size = int(data_args.train_ratio * len(dataset))
    eval_size = int(data_args.eval_ratio * len(dataset))
    test_size = len(dataset) - train_size - eval_size
    train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size])

    logger.info("Train dataset length: %d", len(train_dataset))
    logger.info("Eval dataset length: %d", len(eval_dataset))

    data_collator = custom_collator

    trainer = SFTTrainer(
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