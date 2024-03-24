# 从MNIST数据集中抽取训练及测试样本

import os, random, shutil

data_base_dir = "./data/mnist_img"  # 源图片文件夹路径
target_dir = "data/num_dataset/new"  # 移动到新的文件夹路径
TRAIN_NUM = 2000
# TEST_NUM = 10

# 选取训练集图片
train_data_base_dir = data_base_dir + "/train"
train_data_target_dir = target_dir + "/train2000"
for i in range(0, 10):
    number_dir = train_data_base_dir + "/tensor([{}])".format(i)
    number_target_dir = train_data_target_dir + "/{}".format(i)
    # 判断目录是否存在
    if not os.path.exists(number_target_dir):
        os.mkdir(number_target_dir)
    # 抽取数据
    path_source_dir = os.listdir(number_dir)
    sample = random.sample(path_source_dir, TRAIN_NUM)  # 随机选取picknumber数量的样本图片
    # print(sample)
    cnt = 0
    for name in sample:
        cnt += 1
        shutil.copy(number_dir + "/" + name, number_target_dir + "/{}_{}.bmp".format(i, cnt))
    print("COPY FINISHED! " + number_target_dir)


# # 选取训练集图片
# test_data_base_dir = data_base_dir + "/test"
# test_data_target_dir = target_dir + "/test"
# for i in range(0, 10):
#     number_dir = test_data_base_dir + "/tensor([{}])".format(i)
#     number_target_dir = test_data_target_dir + "/{}".format(i)
#     # 判断目录是否存在
#     if not os.path.exists(number_target_dir):
#         os.mkdir(number_target_dir)
#     # 抽取数据
#     path_source_dir = os.listdir(number_dir)
#     sample = random.sample(path_source_dir, TEST_NUM)  # 随机选取picknumber数量的样本图片
#     # print(sample)
#     cnt = 0
#     for name in sample:
#         cnt += 1
#         shutil.copy(number_dir + "/" + name, number_target_dir + "/{}_{}.bmp".format(i, cnt))
#     print("COPY FINISHED! " + number_target_dir)

