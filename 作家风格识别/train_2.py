# 导入相关包
import copy
import os
import random
import numpy as np
import jieba as jb
import matplotlib.pyplot as plt
import math

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

# 继承 Module 类并实现其中的forward方法
class Net(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_encoder_layers, dim_feedforward, max_seq_length):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = self._get_positional_encoding(max_seq_length, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(embed_dim, 5)  # 假设有5个作家

    def _get_positional_encoding(self, max_seq_length, embed_dim):
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def forward(self, x):
        """
        前向传播
        :param x: 模型输入
        :return: 模型输出
        """
        embedded = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        embedded = embedded + self.positional_encoding[:embedded.size(0), :]
        output = self.transformer_encoder(embedded)
        output = self.fc_out(output.mean(dim=0))
        return output

def processing_data(data_path, split_ratio=0.7):
    """
    数据处理
    :data_path：数据集路径
    :validation_split：划分为验证集的比重
    :return：train_iter,val_iter,TEXT.vocab 训练集、验证集和词典
    """
    # --------------- 已经实现好数据的读取，返回和训练集、验证集，可以根据需要自行修改函数 ------------------
    sentences = [] # 片段
    target = [] # 作者

    # 定义label到数字的映射关系
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    files = os.listdir(data_path)
    for file in files:
        if not os.path.isdir(file):
            f = open(data_path + "/" + file, 'r', encoding='UTF-8');  # 打开文件
            for index, line in enumerate(f.readlines()):
                sentences.append(line)
                target.append(labels[file[:-4]])

    mydata = list(zip(sentences, target))

    TEXT = Field(sequential=True, tokenize=lambda x: jb.lcut(x),
                 lower=True, use_vocab=True, batch_first=True)
    LABEL = Field(sequential=False, use_vocab=False)

    FIELDS = [('text', TEXT), ('category', LABEL)]

    examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS),
                                     mydata))

    dataset = Dataset(examples, fields=FIELDS)

    TEXT.build_vocab(dataset)

    train, val = dataset.split(split_ratio=split_ratio)

    # BucketIterator可以针对文本长度产生batch，有利于训练
    train_iter, val_iter = BucketIterator.splits(
        (train, val),  # 数据集
        batch_sizes=(16, 16),
        device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        repeat=False  # 
    )
    # --------------------------------------------------------------------------------------------

    return train_iter, val_iter, TEXT.vocab

def model(train_iter, val_iter, Text_vocab, save_model_path):
    """
    创建、训练和保存深度学习模型
    """
    # --------------------- 实现模型创建、训练和保存等部分的代码 ---------------------
    # 创建模型实例
    vocab_size = len(Text_vocab)
    embed_dim = 256
    nhead = 8
    num_encoder_layers = 3
    dim_feedforward = 512
    max_seq_length = 100  # 根据实际情况调整
    model = Net(vocab_size, embed_dim, nhead, num_encoder_layers, dim_feedforward, max_seq_length)
    epochs = 10
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_acc_list, train_loss_list = [], []
    val_acc_list, val_loss_list = [], []
    for epoch in range(epochs):
        train_acc, train_loss = 0, 0
        val_acc, val_loss = 0, 0
        for idx, batch in enumerate(train_iter):
            text, label = batch.text, batch.category
            optimizer.zero_grad()
            out = model(text)
            loss = loss_fn(out, label.long())
            loss.backward()
            optimizer.step()
            accuracy = np.mean((torch.argmax(out, 1) == label).cpu().numpy())
            # 计算每个样本的acc和loss之和
            train_acc += accuracy * len(batch)
            train_loss += loss.item() * len(batch)

            print("\r epoch:{} loss:{}, train_acc:{}".format(epoch, loss.item(), accuracy), end=" ")

        # 在验证集上预测
        with torch.no_grad():
            for idx, batch in enumerate(val_iter):
                text, label = batch.text, batch.category
                out = model(text)
                loss = loss_fn(out, label.long())
                accuracy = np.mean((torch.argmax(out, 1) == label).cpu().numpy())
                # 计算一个batch内每个样本的acc和loss之和
                val_acc += accuracy * len(batch)
                val_loss += loss.item() * len(batch)

        train_acc /= len(train_iter.dataset)
        train_loss /= len(train_iter.dataset)
        val_acc /= len(val_iter.dataset)
        val_loss /= len(val_iter.dataset)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        # 绘制曲线
    plt.figure(figsize=(15, 5.5))
    plt.subplot(121)
    plt.plot(train_acc_list)
    plt.plot(val_acc_list)
    plt.title("acc")
    plt.subplot(122)
    plt.plot(train_loss_list)
    plt.plot(val_loss_list)
    plt.title("loss")

    # 保存模型
    torch.save(model.state_dict(), save_model_path)

def main():
    """
    深度学习模型训练流程,包含数据处理、创建模型、训练模型、模型保存、评价模型等。
    如果对训练出来的模型不满意,你可以通过调整模型的参数等方法重新训练模型,直至训练出你满意的模型。
    如果你对自己训练出来的模型非常满意,则可以提交作业!
    :return:
    """
    data_path = "./dataset"  # 数据集路径
    save_model_path = "results/model1.pth"  # 保存模型路径和名称
    train_val_split = 0.7  # 验证集比重

    # 获取数据、并进行预处理
    train_iter, val_iter, Text_vocab = processing_data(data_path, split_ratio=train_val_split)

    # 创建、训练和保存模型
    model(train_iter, val_iter, Text_vocab, save_model_path)

if __name__ == '__main__':
    main()