import warnings
# 忽视警告
warnings.filterwarnings('ignore')

import cv2
from PIL import Image
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch_py.Utils import plot_image
from torch_py.MTCNN.detector import FaceDetector
from torch_py.MobileNetV1 import MobileNetV1
from torch_py.FaceRec import Recognition
from torchvision import models

def processing_data(data_path, height=224, width=224, batch_size=32,
                    test_split=0.1):
    """
    数据处理部分
    :param data_path: 数据路径
    :param height:高度
    :param width: 宽度
    :param batch_size: 每次读取图片的数量
    :param test_split: 测试集划分比例
    :return: 
    """
    transforms = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(0.1),  # 进行随机水平翻转
        T.RandomVerticalFlip(0.1),  # 进行随机竖直翻转
        T.ToTensor(),  # 转化为张量
        T.Normalize([0], [1]),  # 归一化
    ])

    dataset = ImageFolder(data_path, transform=transforms)
    # 划分数据集
    train_size = int((1-test_split)*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # 创建一个 DataLoader 对象
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    valid_data_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

    return train_data_loader, valid_data_loader

data_path = './datasets/5f680a696ec9b83bb0037081-momodel/data/image'

# 加载 MobileNet 的预训练模型权
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
train_data_loader, valid_data_loader = processing_data(data_path=data_path, height=160, width=160, batch_size=32)


# model = MobileNetV1(classes=2).to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 优化器
# # 学习率下降的方式，acc三次不下降就下降学习率继续训练，衰减学习率
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max', factor=0.5,patience=2)
# # 损失函数
# criterion = nn.CrossEntropyLoss()  

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)


model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2)
criterion = nn.CrossEntropyLoss()


best_loss = 1e9
best_model_weights = copy.deepcopy(model.state_dict())
loss_list = []  # 存储损失函数值
acc_list = []

# 训练和验证
epochs = 10
best_acc = 0
#best_model_weights = copy.deepcopy(model.state_dict())

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for x, y in tqdm(train_data_loader):
        x = x.to(device)
        y = y.to(device)
        pred_y = model(x)
        loss = criterion(pred_y, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_data_loader)

    model.eval()
    total = 0
    right_cnt = 0
    valid_loss = 0.0

    with torch.no_grad():
        for b_x, b_y in valid_data_loader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)
            loss = criterion(output, b_y)
            valid_loss += loss.item()
            pred_y = torch.max(output, 1)[1]
            right_cnt += (pred_y == b_y).sum()
            total += b_y.size(0)

    valid_loss = valid_loss / len(valid_data_loader)
    accuracy = right_cnt.float() / total    
    print(f'Epoch: {epoch+1}/{epochs} || Train Loss: {train_loss:.4f} || Val Loss: {valid_loss:.4f} || Val Acc: {accuracy:.4f}')
    # 更新学习率
    scheduler.step(valid_loss)
    # 保存最佳模型权重
    if train_loss < best_loss:
        best_model_weights = copy.deepcopy(model.state_dict())
        best_loss = train_loss
        torch.save(best_model_weights, './results/loss_model1.pth')
        
    if accuracy > best_acc:
        best_model_weights = copy.deepcopy(model.state_dict())
        best_acc = accuracy
        torch.save(best_model_weights, './results/acc_model1.pth')

    loss_list.append(train_loss)
    acc_list.append(accuracy)
    
plt.figure(figsize=(15,5.5))
plt.subplot(122)
plt.plot(loss_list)
plt.plot(acc_list)

print(f'Best Accuracy: {best_acc:.4f}')
print(f'Best Loss: {best_loss:.4f}')
print('Finish Training.')