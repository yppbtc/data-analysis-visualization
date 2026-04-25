"""
CNN 图像分类（CIFAR-10）
使用PyTorch搭建卷积神经网络，完成10分类任务
包含数据加载、模型构建、训练、测试全流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import os

# 全局参数
BATCH_SIZE = 8
EPOCH = 10


def load_dataset():
    """加载CIFAR-10数据集"""
    train_dataset = CIFAR10(root='data', train=True, transform=ToTensor(), download=True)
    test_dataset = CIFAR10(root='data', train=False, transform=ToTensor(), download=True)
    return train_dataset, test_dataset


class CNNModel(nn.Module):
    """CNN模型：2层卷积 + 3层全连接"""
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(576, 120)
        self.linear2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.out(x)


def train_model(model, train_dataset):
    """训练模型并保存"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCH):
        total_loss = 0
        total_samples = 0
        dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        for x, y in dataloader:
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total_samples += len(y)

        print(f"epoch: {epoch + 1}, loss: {total_loss / total_samples:.5f}")

    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), "model/model.pth")
    print("模型已保存到 model/model.pth")


def test_model(model, test_dataset):
    """加载模型并在测试集上评估"""
    model.load_state_dict(torch.load("model/model.pth"))
    model.eval()

    with torch.no_grad():
        dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        total_correct = 0
        total_samples = 0
        for x, y in dataloader:
            output = model(x)
            pred = torch.argmax(output, dim=-1)
            total_correct += (pred == y).sum().item()
            total_samples += len(y)
        acc = total_correct / total_samples
        print(f"测试集准确率: {acc:.2%}")
        return acc


if __name__ == '__main__':
    model = CNNModel()
    train_dataset, test_dataset = load_dataset()
    summary(model, input_size=(3, 32, 32), batch_size=1)
    train_model(model, train_dataset)
    test_model(model, test_dataset)