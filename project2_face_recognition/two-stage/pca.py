import numpy as np
import cv2
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.svm import SVC

import random

# %%
# PCA_NUM = 10
type0 = [(file, 0) for file in Path("./dataset/0/").iterdir()]
type1 = [(file, 1) for file in Path("./dataset/1/").iterdir()]
type2 = [(file, 2) for file in Path("./dataset/2/").iterdir()]

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

#%%
# norm way
predict_label = []

for face in test_face:
    face = face[np.newaxis, :] # should be (1, 2500)
    projected_test_face = pca.transform(face)
    distance = [np.linalg.norm(projected_train_face - projected_test_face)
                for projected_train_face in projectedFace]
    predict_label.append(train_label[distance.index(min(distance))])

acc = [i == j for i,j in zip(test_label, predict_label)]
print(f"acc: {sum(acc)}/{len(acc)}, {sum(acc)/len(acc):.2%}")

# %%
# SVM way
clf = SVC(kernel='linear')
clf.fit(projectedFace, train_label)

predict_label_svm = []

for face in test_face:
    face = face[np.newaxis, :] # should be (1, 2500)
    projected_test_face = pca.transform(face)
    predict = clf.predict(projected_test_face)
    predict_label_svm.append(predict[0])

acc = [i == j for i,j in zip(test_label, predict_label_svm)]
print(f"acc: {sum(acc)}/{len(acc)}, {sum(acc)/len(acc):.2%}")