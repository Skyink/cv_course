# 深度学习方法
# 1. 预处理
# 2. 构建模型
# 3. 模型训练
# 4. 模型测试

import cv2
import numpy as np
import torch
from collections import Counter
from sklearn import datasets
import torch.nn.functional as Fun

import torch.utils.data as TorchData

from torchvision import datasets, models, transforms
import os
import time

BATCH_SIZE = 64
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps")  # mac M1


# ===== 功能函数 =====
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    s = 20
    M = np.float32([[1, skew, -0.5 * s * skew], [0, 1, 0]])
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    size = (28, 28)  # 每个数字的图像的尺寸
    img = cv2.warpAffine(img, M, size, flags=affine_flags)
    return img


def deskew_tensor(tensor):
    img = np.array(tensor[0])
    img = deskew(img)
    new_tensor = torch.tensor(img)
    new_tensor = new_tensor.reshape((1, 28, 28))
    return new_tensor


# 1. 数据准备
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(deskew_tensor),
                                transforms.Normalize(mean=[0.5], std=[0.5])])
train_dataset = datasets.MNIST(root="./data/",
                               transform=transform,
                               train=True,
                               download=True)

test_dataset = datasets.MNIST(root="./data/",
                              transform=transform,
                              train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                )

data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)


# 2. 定义模型
class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


# 3. 定义优化器损失函数
model = Model()
model.to(DEVICE)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.02) # 优化器选用随机梯度下降方式
optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()  # 对于多分类一般采用的交叉熵损失函数,

# 4. 训练模型
EPOCH = 5
time_open = time.time()
for t in range(EPOCH):
    training_loss = 0.0
    training_correct = 0
    for step, (input, label) in enumerate(data_loader_train):  # 一次读一个batch
        # print("batch={}/{}".format(step + 1, int(len(train_dataset) / len(input))))
        input, label = input.to(DEVICE), label.to(DEVICE)
        out = model(input)  # 输入input,输出out
        _, pred = torch.max(out.data, 1)
        loss = loss_func(out, label)  # 输出与label对比
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 前馈操作
        optimizer.step()  # 使用梯度优化器

        training_loss += loss.data
        training_correct += torch.sum(pred == label.data)
    testing_correct = 0
    for data in data_loader_test:
        X_test, y_test = data
        X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)

    loss = float(training_loss * BATCH_SIZE / len(train_dataset))
    train_accuracy = 100 * training_correct / len(train_dataset)
    test_accuracy = 100 * testing_correct / len(test_dataset)
    print(
        "Epoch {:d}/{:d}: Train Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(t + 1, EPOCH,
                                                                                                           loss,
                                                                                                           train_accuracy,
                                                                                                           test_accuracy))

time_end = time.time() - time_open
print(time_end)
