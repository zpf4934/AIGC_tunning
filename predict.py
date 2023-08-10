# -*- coding:utf-8 -*-
"""
# File       : predict.py
# Time       ：2023/6/27 17:54
# Author     ：andy
# version    ：python 3.9
"""
import os
import warnings

import torch
from peft import PeftModel, PeftConfig

from glm import ChatGLMForConditionalGeneration, ChatGLMTokenizer, ChatGLMConfig
from tunning import LM_TYPE, AutoTokenizer

warnings.filterwarnings('ignore')


def load_model(model_name):
    if not model_name.strip():
        return
    model_name = "outputs/" + model_name
    peft = "adapter_config.json" in os.listdir(model_name)
    if peft:
        config = PeftConfig.from_pretrained(model_name)
        if "glm" in model_name.lower():
            model = ChatGLMForConditionalGeneration.from_pretrained(config.base_model_name_or_path,
                                                                    device_map='cuda:1').half()
            model = PeftModel.from_pretrained(model, model_name)
            tokenizer = ChatGLMTokenizer.from_pretrained(config.base_model_name_or_path)
        else:
            model = LM_TYPE[config.task_type].from_pretrained(config.base_model_name_or_path, device_map='auto')
            model = PeftModel.from_pretrained(model, model_name)
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    else:
        conf = ChatGLMConfig.from_pretrained(model_name)
        model = ChatGLMForConditionalGeneration.from_pretrained(model_name, device_map='auto').half()
        tokenizer = ChatGLMTokenizer.from_pretrained(conf.name_or_path)
    return model, tokenizer


def generate(query, model, tokenizer, history=None, top_p=0.7, top_k=50, temperature=0.95, max_length=2048,
             min_length=1, **kwargs):
    gen_kwargs = {"max_length": max_length, "top_k": top_k, "min_length": min_length, "top_p": top_p,
                  "temperature": temperature, **kwargs}
    if history is None:
        history = []
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    model.eval()
    prompt = ""
    for old_query, response in history:
        prompt += "{}{}".format(old_query, response)
    prompt += query
    stream = model.stream_chat(tokenizer, query, history=history, **gen_kwargs)
    return stream
    # with torch.no_grad():
    #     inputs = tokenizer(prompt, return_tensors="pt")
    #     inputs = inputs.to(device)
    #     outputs = model.generate(**inputs, **gen_kwargs)
    #     prompt = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    # return prompt



