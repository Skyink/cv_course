import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

import glob
import cv2


# 加载Mnist数据集
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])
mnist_train = datasets.MNIST(root="./data/",
                               transform=transform,
                               train=True,
                               download=True)

mnist_test = datasets.MNIST(root="./data/",
                              transform=transform,
                              train=False)


# 查看数据集大小
train_data = torch.utils.data.DataLoader(mnist_train, batch_size=1, shuffle=True)
test_data = torch.utils.data.DataLoader(mnist_test, batch_size=1, shuffle=True)
print(len(train_data))
print(len(test_data))


# 保存训练数据为图片
path_train = './data/mnist_img/train'
for i in range(len(train_data)):
    data, target = next(iter(train_data))  # 迭代器
    new_data = data[0][0].clone().numpy()  # 拷贝数据
    save_path = path_train + '/' + str(target)
    # 判断目录是否存在
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img_path = save_path + '/' + str(target) + str(i) + '.bmp'
    plt.imsave(img_path, new_data)
    # 图像二值化处理
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)  # 阈值设为127
    cv2.imwrite(img_path, binary_img)  # 保存二值化图片
    # 输出完成
    print(img_path)

# 保存测试数据为图片
path_test = './data/mnist_img/test'
for i in range(len(test_data)):
    data, target = next(iter(test_data))  # 迭代器
    new_data = data[0][0].clone().numpy()  # 拷贝数据
    save_path = path_test + '/' + str(target)
    # 判断目录是否存在
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img_path = save_path + '/' + str(target) + str(i) + '.bmp'
    plt.imsave(img_path, new_data)
    # 图像二值化处理
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)  # 阈值设为127
    cv2.imwrite(img_path, binary_img)  # 保存二值化图片
    # 输出完成
    print(img_path)


