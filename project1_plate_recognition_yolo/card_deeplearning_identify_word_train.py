# -*- codeing = utf-8 -*-
# @Time : 2024/4/26 10:51
# @Author :李子园
# @File : new.py
# @Software:PyCharm
import cv2
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms, models
from torchvision import datasets as tv_datasets
from torch.utils.data import DataLoader
import time
from PIL import Image
import numpy as np
import torch
import torch.nn as nn


# 车牌汉字
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )


        self.feature = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4

        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 15 * 8, 1024),  # 调整全连接层的输入尺寸
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 31)
        )# 假设有31个类别

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train(model, device, train_loader, optimizer, loss_func, epoch_count):
    for epoch in range(epoch_count):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}/{epoch_count}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return model


def binarize_image(img):
    if isinstance(img, Image.Image):
        img = np.array(img)

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img = clahe.apply(img)

    threshold = 128  # 设置阈值
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)  # 二值化
    # cv2.imshow('1',binary_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return Image.fromarray(binary_img)  # 将NumPy数组转换回PIL图像


def main():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
        transforms.Resize((30, 16)),  # 调整图像大小
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),  # 转换为tensor
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    dataset = tv_datasets.ImageFolder('./annCh', transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()

    start_time = time.time()
    model = train(model, device, loader, optimizer, loss_func, 30)
    torch.save(model.state_dict(), './weights/last_epoch_word_model.pt')
    print("Model saved.")

    end_time = time.time() - start_time
    print(f"Total training time: {end_time:.2f} seconds")


if __name__ == '__main__':
    main()
