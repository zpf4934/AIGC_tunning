# -*- coding:utf-8 -*-
"""
# File       : tunning.py
# Time       ：2023/6/26 14:49
# Author     ：andy
# version    ：python 3.9
"""

import torch
from accelerate import Accelerator
from peft import get_peft_model, PromptTuningConfig, PrefixTuningConfig, TaskType, \
    LoraConfig, PromptEncoderConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, get_linear_schedule_with_warmup, \
    AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

from glm import ChatGLMForConditionalGeneration, ChatGLMTokenizer, Seq2SeqDataSet, coll_fn

LM_TYPE = {
    TaskType.CAUSAL_LM: AutoModelForCausalLM,
    TaskType.SEQ_2_SEQ_LM: AutoModelForSeq2SeqLM,
    TaskType.SEQ_CLS: AutoModelForSequenceClassification
}


class Tunning:
    def __init__(self, config):
        self.lr_scheduler = None
        self.optimizer = None
        self.model = None
        self.config = config
        self.task_type = config.task_type
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if config.distributed == 'accelerate':
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def tokenize(self, text):
        if "glm" in self.config.model_name_or_path.lower():
            dataset = Seq2SeqDataSet(text, self.tokenizer, self.config)
            return dataset
        data = []
        inputs = [t['input'] for t in text]
        outputs = [t['output'] for t in text]
        max_length = self.config.max_source_length + self.config.max_output_length
        for i in tqdm(range(len(inputs))):
            input_tokens = self.tokenizer.tokenize(inputs[i])
            output_tokens = self.tokenizer.tokenize(outputs[i])
            input_tokens = input_tokens[:self.config.max_source_length]
            output_tokens = output_tokens[:self.config.max_output_length]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            label_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
            input_ids = input_ids + label_ids + [self.tokenizer.pad_token_id]
            label_ids = [-100] * (len(input_ids) - len(label_ids) - 1) + label_ids + [self.tokenizer.pad_token_id]
            attention_mask = [1] * len(input_ids)
            input_ids = [self.tokenizer.pad_token_id] * (max_length - len(input_ids)) + input_ids
            attention_mask = [0] * (max_length - len(attention_mask)) + attention_mask
            label_ids = [-100] * (max_length - len(label_ids)) + label_ids
            input_ids = torch.tensor(input_ids[:max_length])
            attention_mask = torch.tensor(attention_mask[:max_length])
            label_ids = torch.tensor(label_ids[:max_length])
            data.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids})
        return data

    def dataloader(self, dataset):
        coll = coll_fn if "glm" in self.config.model_name_or_path.lower() else default_data_collator
        return DataLoader(dataset, collate_fn=coll, batch_size=self.config.batch_size, pin_memory=True,
                          num_workers=0)

    def create_optimizer(self, dataloader):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=int(len(dataloader) * self.config.epochs * 0.1),
            num_training_steps=(len(dataloader) * self.config.epochs),
        )

    def create_model(self, peft_config):
        if "glm" in self.config.model_name_or_path.lower():
            self.model = ChatGLMForConditionalGeneration.from_pretrained(self.config.model_name_or_path).half()
            self.tokenizer = ChatGLMTokenizer.from_pretrained(self.config.model_name_or_path)
        else:
            self.model = LM_TYPE[self.task_type].from_pretrained(self.config.model_name_or_path)
        if peft_config:
            self.model = get_peft_model(self.model, peft_config)


class PromptTuning(Tunning):
    def __init__(self, config):
        super().__init__(config)
        self.prompt_tuning_init = config.prompt_tuning_init
        self.num_virtual_tokens = config.num_virtual_tokens
        self.prompt_tuning_init_text = config.prompt_tuning_init_text

    def load_model(self):
        peft_config = PromptTuningConfig(
            task_type=self.task_type,
            prompt_tuning_init=self.prompt_tuning_init,
            num_virtual_tokens=self.num_virtual_tokens,
            prompt_tuning_init_text=self.prompt_tuning_init_text,
            tokenizer_name_or_path=self.config.model_name_or_path,
        )
        self.create_model(peft_config)


class PrefixTuning(Tunning):
    def __init__(self, config):
        super().__init__(config)
        self.num_virtual_tokens = config.num_virtual_tokens

    def load_model(self):
        peft_config = PrefixTuningConfig(
            task_type=self.task_type,
            num_virtual_tokens=self.num_virtual_tokens
        )
        self.create_model(peft_config)


class LoraTuning(Tunning):
    def __init__(self, config):
        super().__init__(config)
        self.inference_mode = config.inference_mode
        self.r = config.r
        self.lora_alpha = config.lora_alpha
        self.lora_dropout = config.lora_dropout

    def load_model(self):
        peft_config = LoraConfig(
            task_type=self.task_type,
            inference_mode=self.inference_mode,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout
        )
        self.create_model(peft_config)


class PTuning(Tunning):
    def __init__(self, config):
        super().__init__(config)
        self.num_virtual_tokens = config.num_virtual_tokens
        self.encoder_hidden_size = config.encoder_hidden_size

    def load_model(self):
        if "glm" in self.config.model_name_or_path.lower():
            self.create_model(None)
            if self.config.prefix_projection:
                self.model.gradient_checkpointing_enable()
            for name, param in self.model.named_parameters():
                if not any(nd in name for nd in ["prefix_encoder"]):
                    param.requires_grad = False
        else:
            peft_config = PromptEncoderConfig(
                task_type=self.task_type,
                num_virtual_tokens=self.num_virtual_tokens,
                encoder_hidden_size=self.encoder_hidden_size
            )
            self.create_model(peft_config)


class FreezeTuning(Tunning):
    def __init__(self, config):
        super().__init__(config)
        self.freeze_layer = config.freeze_layer

    def load_model(self):
        self.create_model(None)
        if "glm" in self.config.model_name_or_path.lower():
            for name, param in self.model.named_parameters():
                if not any(nd in name for nd in self.config.freeze_layer):
                    param.requires_grad = False


TUNING = {
    "prompt": PromptTuning,
    "prefix": PrefixTuning,
    "lora": LoraTuning,
    "p": PTuning,
    "freeze": FreezeTuning
}
