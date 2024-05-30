import numpy as np
import cv2
from pathlib import Path
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import random
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

# 如果使用M1芯片加速用mps
device = torch.device('cpu')
print('在该设备上运行: {}'.format(device))

# %%
type0 = [(file, 0) for file in Path("../data/dataset/0/").iterdir()]
type1 = [(file, 1) for file in Path("../data/dataset/1/").iterdir()]
type2 = [(file, 2) for file in Path("../data/dataset/2/").iterdir()]

raw = [type0, type1, type2]

random.seed(233)
for i in raw:
    random.shuffle(i)

trainset = []
testset = []

for i in raw:
    trainset.extend(i[:10])
    testset.extend(i[10:])


# %%
# 定义加载器
class MyDataLoader(Dataset):
    def __init__(self, my_image_list, transform=None):
        """
        element in my_image_list: (Path, label)
        """
        self.image_list = my_image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        file, label = self.image_list[idx]
        img = cv2.imread(file.as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 160))
        if self.transform:
            img = self.transform(img).to(device)
        return img, label


# %%
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MyDataLoader(trainset, transform)
test_dataset = MyDataLoader(testset, transform)

train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=True)

# %%
# 冻结预训练模型的参数
pre_trained_resnet = InceptionResnetV1(pretrained='vggface2').eval()
for param in pre_trained_resnet.parameters():
    param.requires_grad = False


# %%
# 定义微调模型结构
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


myModel = CombinedModel(pre_trained_resnet).to(device)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(myModel.parameters(), lr=0.01)

# # 打印模型结构
# print(myModel)

myModel.train()

EPOCH = 20

for epoch in range(EPOCH):
    print(f"training: {epoch}/{EPOCH}")
    for data, label in train_dataloader:
        optimizer.zero_grad()
        output = myModel(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

# %%
# test acc
total_correct = 0
total_test_samples = 0

with torch.no_grad():
    for data, label in test_dataloader:
        outputs = myModel(data)
        predicted_labels = torch.argmax(outputs, dim=1)

        correct = (predicted_labels == label).sum().item()
        total_correct += correct
        total_test_samples += len(label)

print(f"acc: {total_correct}/{total_test_samples},\
        {total_correct / total_test_samples:.2f}")

torch.save(myModel, "./model/my_model_softmax.pt")
