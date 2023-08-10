# -*- coding:utf-8 -*-
"""
# File       : prompt_init.py
# Time       ：2023/7/3 17:15
# Author     ：andy
# version    ：python 3.9
"""
import json
from tqdm import tqdm
from prompt import doctor

datas = []
prompt_data = []
with open('doctor.json', 'w') as fw:
    with open('train_data.json') as fr:
        datas.extend(json.load(fr))
    with open('test_data.json') as fr:
        datas.extend(json.load(fr))
    with open('validate_data.json') as fr:
        datas.extend(json.load(fr))

    for data in tqdm(datas):
        for i in range(0, len(data), 2):
            a = {'input': doctor['query'].format(query=data[i].strip()[3:]),
                 'output': doctor['answer'].format(content=data[i+1].strip()[3:])}
            prompt_data.append(a)
    fw.write(json.dumps(prompt_data, ensure_ascii=False))
