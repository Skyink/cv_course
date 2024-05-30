from torchvision.datasets import ImageFolder
import glob
import cv2
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch
import torch.nn as nn


class Model1(torch.nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
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




# 定义模型架构，确保与训练时完全一致
class Model2(torch.nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
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
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.feature = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4

        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 15 * 8, 1024),  # 调整全连接层的输入尺寸
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 34)
        )  # 假设有34个类别

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def binarize_image(img):
    threshold = 128  # 设置阈值
    img = np.array(img)  # 将PIL图像转换为NumPy数组
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转换为灰度图
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)  # 二值化
    return Image.fromarray(binary_img)  # 将NumPy数组转换回PIL图像



def loadPlates(directory):
    files = glob.glob(f'{directory}/*.jpg')
    plates = []
    for file in files:
        img = cv2.imread(file)
        if img is not None:
            plates.append(img)
    return plates


# ======================预处理函数，图像去噪等处理=================
def preprocessor(image):
    # 色彩空间转换（RGB-->GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 去噪处理
    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image



# 图像预处理函数
def preprocess_image(img):
    if len(img.shape) == 2 or img.shape[2] == 1:
        # 图像已经是灰度图，无需转换
        pass
    else:
        # 图像不是灰度图，转换为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (16, 30))  # 使用 OpenCV 调整图像大小
    img = 255 - img
    img = img / 255.0  # 归一化
    img = img.reshape(1, 1, 30, 16)  # 重塑为1x1x16x30
    img = torch.tensor(img, dtype=torch.float32)  # 转换为张量
    return img


def find_end(start_):
    end_ = start_ + 1
    for m in range(start_ + 1, width - 1):
        if (black[m] if arg else white[m]) > (
        0.95 * black_max if arg else 0.95 * white_max):  # 0.95这个参数请多调整，对应下面的0.05（针对像素分布调节）
            end_ = m
            break
    return end_




# 加载模型
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = Model1()
model2 = Model2()
model1.load_state_dict(torch.load('./weights/last_epoch_word_model.pt', map_location=DEVICE))
model2.load_state_dict(torch.load('./weights/last_epoch_NL_model.pt', map_location=DEVICE))
model1.to(DEVICE)
model2.to(DEVICE)
model1.eval()  # 设置为评估模式
model2.eval()

# 加载模型
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
    transforms.Resize((30, 16)),  # 调整图像大小
    transforms.Lambda(binarize_image),  # 应用二值化
    transforms.ToTensor(),  # 转换为tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 加载数据集
dataset1 = ImageFolder('./annCh', transform=transform)
dataset2 = ImageFolder('./annGray', transform=transform)
classes1 = dataset1.classes  # 获取子文件夹名称列表
classes2 = dataset2.classes

# 图像路径
plates = loadPlates("result/crops")

#plate就是车牌
for plate in plates:
    #plateChars = splitPlate(plate)  # 分割车牌，将每个字符独立出来

    #plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    B, G, R = cv2.split(plate)
    B[:] = 0
    R[:] = 0
    plate = cv2.merge([B, G, R])
    plate = G

    ret, img_thre = cv2.threshold(plate, 100, 255, cv2.THRESH_BINARY_INV)
    white = []  # 记录每一列的白色像素总和
    black = []  # ..........黑色.......
    height = img_thre.shape[0]
    width = img_thre.shape[1]
    white_max = 0
    black_max = 0
    for i in range(width):
        s = 0  # 这一列白色总数
        t = 0  # 这一列黑色总数
        for j in range(height):
            if img_thre[j][i] == 255:
                s += 1
            if img_thre[j][i] == 0:
                t += 1
        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)
    arg = False  # False表示白底黑字；True表示黑底白字
    if black_max > white_max:
        arg = True

    n = 1
    start = 1
    end = 2
    plateChars = []

    while n < width - 2:
        n += 1
        if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):
            # 上面这些判断用来辨别是白底黑字还是黑底白字
            # 0.05这个参数请多调整，对应上面的0.95
            start = n
            end = find_end(start)
            n = end
            if end - start > 5:
                cj = plate[1:height, start:end]
                cj = cv2.resize(cj, (16, 30))
                cv2.imshow("123", cj)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                plateChars.append(cj)



    result = []
    for idx, plateChar in enumerate(plateChars):
        input_tensor = preprocess_image(plateChar)
        input_tensor = input_tensor.to(DEVICE)
        if idx == 0:
            output = model1(input_tensor)  # 对第一个字符使用Model1
        else:
            output = model2(input_tensor)  # 对剩余字符使用Model2
        _, predicted = torch.max(output, 1)
        class_name = classes1[predicted.item()] if idx == 0 else classes2[predicted.item()]
        result.append(class_name)
    print(" ".join(result))  # 打印整个车牌的预测结果
