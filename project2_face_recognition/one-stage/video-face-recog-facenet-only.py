import numpy as np
import torch
import torchvision.transforms as transforms
import PIL.Image
from torch import nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import cv2
import time


# 模型网络结构
class CombinedModel(nn.Module):
    def __init__(self, pre_trained_model):
        super(CombinedModel, self).__init__()
        self.pre_trained_model = pre_trained_model
        self.new_layer = nn.Linear(512, 3)
        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x):
        pre_trained_output = self.pre_trained_model(x)
        output = self.new_layer(pre_trained_output)
        probability = self.softmax_layer(output)
        return probability


# %%
# 人脸检测，获得人脸区域选框
def get_boxes(img: PIL.Image.Image):
    mtcnn = MTCNN()
    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        return None
    return [list(box) for box in boxes]


# 载入模型
device = torch.device('mps')
# my_model = torch.load("./model/my_model3.pt").to(device)
my_model = torch.load("./model/my_model_softmax.pt").to(device)
my_model.eval()
# 定义转换器
transform = transforms.Compose([transforms.ToTensor()])
label_class = ["汤师爷", "张麻子", "黄老爷"]
label_color = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]


# 读取输入视频视频
# video_file = Path("../data/input/original_video.mp4")
# video_file = Path("../data/input/cut.mp4")
# video_file = Path("../data/input/cut-multiple.mp4")
# video_file = Path("../data/input/cut-test.mp4")
video_file = Path("../data/input/cut-test2.mp4")


# %%
# 读取视频流
cap = cv2.VideoCapture(video_file.as_posix())

FRAME_INTERVAL = 24    # 帧间隔

i = 0
face_idx = 0
face_recognized_idx = 0

train_start = time.time_ns()
# 设置字体及大小
font = ImageFont.truetype("../resource/MS_yahei.ttf", 36)
# 保存路径
face_crop_dir = Path("../data/output/face_crop")
if not face_crop_dir.exists():
    face_crop_dir.mkdir()

face_recognized_dir = Path("../data/output/face_recognized")
if not face_recognized_dir.exists():
    face_recognized_dir.mkdir()
while True:
    i += 1
    # ret 返回标识，当视频结束时 ret = false
    ret, frame = cap.read()

    if not ret:
        break

    # # 帧间隔
    # if i % FRAME_INTERVAL != 0:
    #     continue
    # if i > 240:
    #     break

    # 处理帧画面
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    boxes = get_boxes(img)
    if boxes is not None:
        draw = ImageDraw.Draw(img)
        for box in boxes:
            # 获取人脸区域坐标
            x, y, w, h = box
            # # 尺寸小于160*160的框舍弃
            # if abs(y-x) < 160 or abs(w-h) < 160:
            #     print("识别框尺寸过小，舍弃")
            #     continue
            # 裁切人脸区域并保存
            face = img.crop(box)
            face.save(f"{face_crop_dir}/{face_idx}.jpg")
            # 调用模型识别人脸
            face_img = cv2.imread(f"{face_crop_dir}/{face_idx}.jpg")
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (160, 160))
            face_input = transform(face_img).to(device)
            face_tensor = torch.unsqueeze(face_input, 0)    # 将原先的 (C, H, W) 张量转换成期望的 (1, C, H, W) 形式, 1代表batch_size为 1
            predicted_label = 0
            probability = 0
            with torch.no_grad():
                outputs = my_model(face_tensor)
                probability, predicted_label = torch.max(outputs, dim=1)
                probability = probability.item()
                predicted_label = predicted_label.item()
            # 舍弃置信度过低的区域
            if probability < 0.85:
                print("人脸{} 识别类型：{}-{} 识别概率过低：{} 舍弃该识别区域".format(face_idx, predicted_label, label_class[predicted_label], probability))
                continue
            print("人脸{} 识别类别：{}-{} 识别概率：{}".format(face_idx, predicted_label, label_class[predicted_label], probability))
            color = label_color[predicted_label]
            # 框出人脸区域
            draw.rectangle(box, outline=color, width=2)
            # 在识别框上方添加文字
            text = label_class[predicted_label]
            draw.text((x, y), text, font=font, fill=color)

            face_idx += 1
    img.save(f"{face_recognized_dir}/{face_recognized_idx}.jpg")
    face_recognized_idx += 1

train_end = time.time_ns()
train_time = (train_end - train_start) / (1000*1000*1000)
print("用时：{}s".format(train_time))
