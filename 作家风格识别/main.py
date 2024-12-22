import copy
import os
import random
import numpy as np
import jieba as jb
import jieba.analyse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as f
from torchtext import data
from torchtext import datasets
from torchtext.data import Field
from torchtext.data import Dataset
from torchtext.data import Iterator
from torchtext.data import Example
from torchtext.data import BucketIterator

class RNN(nn.Module):
    def __init__(self,Text_vocab):
        super(RNN, self).__init__()
        self.emb = nn.Embedding(num_embeddings=len(Text_vocab), embedding_dim=300)
        self.lstm = torch.nn.LSTM(300, 128, bidirectional=True, dropout=0.2, batch_first=True)
        self.fc1 = nn.Linear(256, 5)
        self.fc2 = nn.Linear(128, 5)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.emb(x).permute(1, 0, 2)
        output, hidden = self.lstm(x)
        # print(hidden[0].shape)
        # x = output[:, 0, :]
        x = output.mean(1)
        x = self.fc1(x)
        # x = self.fc2(x)
        out = x
        return out

def load_data(path):
    """
    读取数据和标签
    :param path:数据集文件夹路径
    :return:返回读取的片段和对应的标签
    """
    sentences = [] # 片段
    target = [] # 作者

    # 定义lebel到数字的映射关系
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(file):
            f = open(path + "/" + file, 'r', encoding='UTF-8');  # 打开文件
            for index,line in enumerate(f.readlines()):
                sentences.append(line)
                target.append(labels[file[:-4]])

    return list(zip(sentences, target))

# 定义Field
TEXT  = Field(sequential=True, tokenize=lambda x: jb.lcut(x), lower=True, use_vocab=True)
LABEL = Field(sequential=False, use_vocab=False)
FIELDS = [('text', TEXT), ('category', LABEL)]

# 读取数据，是由tuple组成的列表形式
mydata = load_data(path='dataset/')

# 使用Example构建Dataset
examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), mydata))
dataset = Dataset(examples, fields=FIELDS)
# 构建中文词汇表
TEXT.build_vocab(dataset)


# 创建模型实例
model = RNN(TEXT.vocab)
model_path ="results/loss_model1.pth" 
model.load_state_dict(torch.load(model_path))

# -------------------------请勿修改 predict 函数的输入和输出-------------------------
def predict(text):
    """
    :param text: 中文字符串
    :return: 字符串格式的作者名缩写
    """
     # ----------- 实现预测部分的代码，以下样例可代码自行删除，实现自己的处理方式 -----------
    labels = {0: 'LX', 1: 'MY', 2: 'QZS', 3: 'WXB', 4: 'ZAL'}
    # 自行实现构建词汇表、词向量等操作
    # 将句子做分词，然后使用词典将词语映射到他的编号

    text2idx = [TEXT.vocab.stoi[i] for i in jb.lcut(text) ]

    # 转化为Torch接收的Tensor类型
    text2idx = torch.Tensor(text2idx).long()

    # 模型预测部分
    results = model(text2idx.view(-1,1))
    prediction = labels[torch.argmax(results,1).numpy()[0]]
    # --------------------------------------------------------------------------

    return prediction
