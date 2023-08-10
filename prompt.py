# -*- coding:utf-8 -*-
"""
# File       : prompt.py
# Time       ：2023/7/3 16:49
# Author     ：andy
# version    ：python 3.9
"""
poetryAuthor = {
    "query": "你现在是一位诗人，请根据以下关键词>>>{query}<<<生成诗歌。",
    "answer": "标题：{title}。\n内容：{content}。"
}
coupletsAuthor = {
    "query": "上联：{query}",
    "answer": "下联：{content}"
}
doctor = {
    "query": "病人：{query}",
    "answer": "医生：{content}"
}