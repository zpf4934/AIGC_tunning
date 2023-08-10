# -*- coding:utf-8 -*-
"""
# File       : arguments.py
# Time       ：2023/6/28 9:54
# Author     ：andy
# version    ：python 3.9
"""
from dataclasses import dataclass, field
from typing import List


def default_list() -> List[List[str]]:
    return []


def freeze_layer() -> List[str]:
    return ["layers.27", "layers.26", "layers.25", "layers.24", "layers.23"]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="/aigc/modelclub/chatglm-6b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    history: List[List[str]] = field(
        default_factory=default_list, metadata={"help": "历史对话"}
    )
    max_source_length: int = field(
        default=150,
        metadata={"help": "max length of input text"},
    )
    max_output_length: int = field(
        default=150,
        metadata={"help": "max output length"},
    )
    lr: float = field(
        default=1e-8,
        metadata={"help": "learning rate"},
    )
    epochs: int = field(
        default=20,
        metadata={"help": "epochs"},
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "natch of size"},
    )
    task_type: str = field(
        default="CAUSAL_LM", metadata={"help": "The type of task to perform, choose from [SEQ_CLS, SEQ_2_SEQ_LM, "
                                               "CAUSAL_LM]"}
    )
    prompt_tuning_init: str = field(
        default="RANDOM",
        metadata={"help": "How to initialize the prompt tuning parameters, choose from [TEXT, RANDOM]"}
    )
    prompt_tuning_init_text: str = field(
        default="robot:",
        metadata={
            "help": "The text to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"},
    )
    num_virtual_tokens: int = field(
        default=20,
        metadata={"help": "Number of virtual tokens"},
    )
    inference_mode: bool = field(
        default=False,
        metadata={"help": "Whether to use inference mode"},
    )
    r: int = field(
        default=8,
        metadata={"help": "Lora attention dimension"},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "Lora alpha"},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "lora dropout"},
    )
    encoder_hidden_size: int = field(
        default=128,
        metadata={"help": "The hidden size of the prompt encoder"},
    )
    distributed: str = field(
        default="",
        metadata={"help": "distributed type deepspeed or accelerate"},
    )
    prefix_projection: bool = field(
        default=True,
        metadata={"help": "Whether to project the prefix tokens"},
    )
    early_stop_step: int = field(
        default=20,
        metadata={"help": "early stop if eval loss not down on the training step"},
    )
    seed: int = field(
        default=None,
        metadata={"help": "set the random seed"},
    )
    freeze_layer: List[str] = field(
        default_factory=freeze_layer, metadata={"help": "需要冻结的层"}
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "Reserved for deepspeed framework"},
    )
    deepspeed_config: str = field(
        default='',
        metadata={"help": "deepspeed conf"},
    )


@dataclass
class TrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default="doctor.json",
        metadata={"help": "tunning datasets"}
    )
    tuning_method: str = field(
        default="lora",
        metadata={"help": "choose tuning method, choose from [prompt, prefix, lora, p, freeze]"}
    )
    save_path: str = field(
        default="glm_lora_doctor",
        metadata={"help": "tuning result model"}
    )


@dataclass
class GenerateArguments(TrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    top_p: float = field(
        default=0.7,
        metadata={"help": "只截断选取从高到低累积概率达到top_p的词"}
    )
    top_k: int = field(
        default=50,
        metadata={"help": "只截断选取概率最高的前K个词"}
    )
    temperature: float = field(
        default=0.95,
        metadata={"help": "温度越低（小于1），softmax输出的贫富差距越大；温度越高，softmax差距越小"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "生成序列的最大长度"}
    )
    min_length: int = field(
        default=1,
        metadata={"help": "生成序列的最短长度"}
    )
    stop_char: str = field(
        default="",
        metadata={"help": "stop char split with “,”"},
    )
    distributed_type: str = field(
        default="",
        metadata={"help": "can choose accelerate"}
    )
