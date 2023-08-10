# -*- coding:utf-8 -*-
"""
# File       : cli_demo.py
# Time       ：2023/7/17 14:12
# Author     ：andy
# version    ：python 3.9
"""
from predict import load_model, generate
from transformers import HfArgumentParser
from arguments import GenerateArguments, ModelArguments


def main():
    parser = HfArgumentParser((ModelArguments, GenerateArguments))
    _, generate_args = parser.parse_args_into_dataclasses()
    model, tokenizer = load_model(generate_args.save_path)
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        current_length = 0
        result = generate(query, model, tokenizer, [], generate_args.top_p, generate_args.top_k,
                          generate_args.temperature, generate_args.max_length, generate_args.min_length)
        print("\nAIGC：", end="")
        for response, history in result:
            print(response[current_length:], end="", flush=True)
            current_length = len(response)


main()
