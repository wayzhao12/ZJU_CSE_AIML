import warnings
# 忽视警告
warnings.filterwarnings('ignore')

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import models
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import random

random_number = random.randint(1000,9999)


def processing_data(data_path, height=224, width=224, batch_size=32, test_split=0.1):
    """
    数据处理部分
    :param data_path: 数据路径
    :param height: 高度
    :param width: 宽度
    :param batch_size: 每次读取图片的数量
    :param test_split: 测试集划分比例
    :return: train_data_loader, valid_data_loader
    """
    transforms_pipeline = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(0.1),  # 随机水平翻转
        T.RandomVerticalFlip(0.1),    # 随机竖直翻转
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    dataset = ImageFolder(data_path, transform=transforms_pipeline)
    # 划分数据集
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_data_loader, valid_data_loader


def initialize_model(num_classes=2):
    """
    初始化 ResNet50 模型并修改输出层
    :param num_classes: 分类的类别数
    :return: model
    """
    model = models.resnet50(pretrained=False)
    local_pretrained_path = "resnet50-19c8e357.pth"
    state_dict = torch.load(local_pretrained_path)
    model.load_state_dict(state_dict, strict=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def train_model(model, train_loader, valid_loader, device, epochs=10, learning_rate=1e-3):
    """
    训练模型
    :param model: ResNet50 模型
    :param train_loader: 训练数据加载器
    :param valid_loader: 验证数据加载器
    :param device: 训练设备（CPU/GPU）
    :param epochs: 训练的轮数
    :param learning_rate: 学习率
    :return: best_model_weights, loss_list, acc_list
    """
    # 定义优化器、损失函数和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())

    loss_list = []
    acc_list = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                valid_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        valid_loss /= len(valid_loader)
        accuracy = correct / total

        # 打印每个 epoch 的结果
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}, Val Acc: {accuracy:.4f}")

        # 更新学习率
        scheduler.step(valid_loss)


        # 保存最佳模型权重
        if valid_loss <= best_loss:
            best_loss = valid_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, f"./results/gpu_best_loss_resnet50_pret_{random_number}.pth")

        if accuracy >= best_acc:
            best_acc = accuracy
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, f"./results/gpu_best_acc_resnet50_pret_{random_number}.pth")

        loss_list.append(train_loss)
        acc_list.append(accuracy)

    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Best Loss: {best_loss:.4f}")

    return best_model_weights, loss_list, acc_list


def plot_metrics(loss_list, acc_list):
    """
    绘制训练损失和验证准确率的变化曲线
    :param loss_list: 训练损失列表
    :param acc_list: 验证准确率列表
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acc_list, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.show()



data_path = './datasets/5f680a696ec9b83bb0037081-momodel/data/image'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(random_number)

# 数据处理
train_loader, valid_loader = processing_data(data_path=data_path, height=160, width=160, batch_size=32)

# 初始化模型
model = initialize_model(num_classes=2).to(device)

# 训练模型
best_weights, losses, accuracies = train_model(model, train_loader, valid_loader, device, epochs=10, learning_rate=1e-3)