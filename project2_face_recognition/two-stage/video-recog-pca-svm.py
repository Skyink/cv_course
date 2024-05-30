import numpy as np
import cv2
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from facenet_pytorch import MTCNN

import PIL.Image
from PIL import Image, ImageDraw, ImageFont

import random
import time


# %%
# PCA_NUM = 10
type0 = [(file, 0) for file in Path("../data/dataset/0/").iterdir()]
type1 = [(file, 1) for file in Path("../data/dataset/1/").iterdir()]
type2 = [(file, 2) for file in Path("../data/dataset/2/").iterdir()]

raw = [type0, type1, type2]

random.seed(123)
for i in raw:
    random.shuffle(i)

trainset = []
testset = []

for i in raw:
    trainset.extend(i[:5])
    testset.extend(i[5:])

# %%
IMAGE_SIZE = (50, 50)


def load_one_file(file: Path):
    img = cv2.imread(file.as_posix(), cv2.IMREAD_GRAYSCALE)
    img: np.ndarray = cv2.resize(img, IMAGE_SIZE)
    img = img.flatten()
    return img


# %%
def load_files(dir: list):
    return np.vstack([load_one_file(file) for file in dir])

raw_face = load_files([file for file, type in trainset]) # (15, 2500)
train_label = [type for file, type in trainset]

test_face = load_files([file for file, type in testset])
test_label = [type for file, type in testset]

# %%
# PCA recognize
pca = PCA(n_components=0.9)
projectedFace = pca.fit_transform(raw_face)

# %%
# SVM way
clf = SVC(kernel='linear')
clf.fit(projectedFace, train_label)


# %%
# 人脸检测，获得人脸区域选框
def get_boxes(img: PIL.Image.Image):
    mtcnn = MTCNN()
    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        return None
    return [list(box) for box in boxes]


# %%
# 定义分类
label_class = ["汤师爷", "张麻子", "黄老爷"]
label_color = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

# 读取视频流
video_file = Path("../data/input/cut.mp4")
cap = cv2.VideoCapture(video_file.as_posix())

FRAME_INTERVAL = 24    # 帧间隔

i = 0
face_idx = 0
face_recognized_idx = 0

train_start = time.time_ns()
# 设置字体及大小
font = ImageFont.truetype("../resource/MS_yahei.ttf", 36)
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
            face.save(f"../data/output/face_crop/{face_idx}.jpg")
            # 调用SVM识别人脸
            face_img = cv2.imread(f"../data/output/face_crop/{face_idx}.jpg", cv2.IMREAD_GRAYSCALE)
            face_img: np.ndarray = cv2.resize(face_img, IMAGE_SIZE)
            face_img = face_img.flatten()
            face_for_pca = face_img[np.newaxis, :]  # should be (1, 2500)
            projected_test_face = pca.transform(face_for_pca)
            predicted_label = clf.predict(projected_test_face)[0]
            print("人脸{} 识别类别：{}-{}".format(face_idx, predicted_label, label_class[predicted_label]))
            color = label_color[predicted_label]
            # 框出人脸区域
            draw.rectangle(box, outline=color, width=2)
            # 在识别框上方添加文字
            text = label_class[predicted_label]
            draw.text((x, y), text, font=font, fill=color)

            face_idx += 1
    img.save(f"../data/output/face_recognized/{face_recognized_idx}.jpg")
    face_recognized_idx += 1

train_end = time.time_ns()
train_time = (train_end - train_start) / (1000*1000*1000)
print("用时：{}s".format(train_time))
