import numpy as np
import cv2
from pathlib import Path
import torch
import torchvision.transforms as transforms

from sklearn.svm import SVC
from facenet_pytorch import MTCNN, InceptionResnetV1
import random

# %%

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

train_face = [file for file, type in trainset]
train_label = [type for file, type in trainset]

test_face = [file for file, type in testset]
test_label = [type for file, type in testset]

# %%

resnet = InceptionResnetV1(pretrained='vggface2').eval()
transform = transforms.Compose([transforms.ToTensor()])


def image_to_feature(file: Path):
    img = cv2.imread(file.as_posix())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))
    face_tensor = transform(img)
    face_tensor = torch.unsqueeze(face_tensor, 0)
    with torch.no_grad():
        features = resnet(face_tensor)
        features = features.squeeze()

    return np.array(features)


# %%
projectedFace = np.array([image_to_feature(file) for file in train_face])

#%%
predict_label = []

for face in test_face:
    projected_test_face = image_to_feature(face)
    distance = [np.linalg.norm(projected_train_face - projected_test_face)
                for projected_train_face in projectedFace]
    predict_label.append(train_label[distance.index(min(distance))])

acc = [i == j for i,j in zip(test_label, predict_label)]
print(f"acc: {sum(acc)}/{len(acc)}, {sum(acc)/len(acc):.2%}")


#%%
clf = SVC(kernel='linear')
clf.fit(projectedFace, train_label)

predict_label_svm = []

for face in test_face:
    projected_test_face = image_to_feature(face)
    predict = clf.predict(projected_test_face.reshape((1,-1)))
    predict_label_svm.append(predict[0])

acc = [i == j for i,j in zip(test_label, predict_label_svm)]
print(f"acc: {sum(acc)}/{len(acc)}, {sum(acc)/len(acc):.2%}")