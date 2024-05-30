## UESTC 高级计算机视觉

2024.03-2024.05

课程代码



### 课程作业

1. `cv01_image_filter` 作业1：图像滤波
   - 自定义高斯滤波核函数、自定义高斯滤波函数，实现对图像的高斯滤波。
   - 对比自定义函数与OpenCV库函数的滤波效果。
   - 调整卷积核大小及参数，分析对滤波效果的影响。
2. `cv02_image_enhancement` 作业2：图像增强
   - 直方图均衡化效果（自定义函数与OpenCV库函数）
   - CLAHE不同参数值效果比较
   - Gamma校正不同参数值效果比较
   - CLAHE与Gamma校正的处理效果比较

3. `cv03_number_detection` 作业3：手写数字识别
   - 实现传统机器学习方法（HOG特征+SVM分类器）。
   - 实现深度学习方法（MNIST）。
   - 将MNIST原始数据集中解析为图片，并二值化为保存。
   - 从MNIST解析后的数据集中抽取样本。



### 课程项目

- 项目代码来源小组各成员。
- 项目使用的图片及视频相关数据过大，仅保留各级目录。具体数据省略上传。

1. `project1_plate_recognition_traditional` 为车牌识别项目中的传统方法相关代码。具体说明详见该目录下的`readme.md`。

2. `project1_plate_recognition_yolo` 为车牌识别项目中的YOLO方法相关代码。具体说明详见该目录下的`readme.docx`。

3. `project2_face_recognition` 为人脸检测与识别项目相关代码。具体说明详见该目录下的`README.md`。

   - 特征提取+分类器方法相关代码在其下`two-stage`目录中

   - 神经网络端到端方法相关代码在其下`one-stage`目录中
