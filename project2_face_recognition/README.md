## CV项目2：人脸检测和识别项目任务

根据参考代码，编程实现视频中的人脸识别，具体要求：

a) 用MTCNN模型实现视频中的人脸检测；

b) 利用（a）的结果，提取出视频中的人脸，构建人脸库，并进行标注；

c) 使用特征提取+分类器的方式对视频人脸进行识别，
   - 特征采用PCA和另一种人脸特征（如FaceNet），可更多特征类型；
   - 分类器采用特征距离和SVM两种，可更多分类器类型；

d) 使用神经网络端到端方式对视频人脸进行识别；

e) 性能分析比较：
   - 任务c中各组合方法;
   - 任务c中最好的方法与任务d;




### 程序使用说明

`./data`为数据存放的文件路径。
- `./data/splite`为生成数据集时MTCNN从源视频素材中检测到的人脸区域，截取并保存。
- `./data/dataset`为手动分类的数据集，三个角色分别对应0、1、2三个子目录。
- `./data/input`为输入视频的存放路径。
- `./data/output`为输出数据的存放路径。
  - `./data/output/xxx_face_crop`存放MTCNN识别并裁切出的人脸区域。
  - `./data/output/xxx_face_recognized`存放识别完成并对单帧进行标注的输出图像。
  - `./data/output/mac_join_video.bash`为使用ffmpeg合成视频的脚本，在输出帧目录下执行即可。

`./one-stage`为端到端方法文件路径
- `./one-stage/model`为训练好的模型保存目录。
- `./one-stage/fine_tune.py`为模型微调+训练程序。
- `./one-stage/video-face-recog-facenet-only.py`为使用端到端方法实现输入视频到识别输出流程的程序。
- `./one-stage/facenet-finetune.py`为参考模型代码，是一个封装好的端到端识别程序。（可忽略）
- `./one-stage/runs`为参考模型代码运行时记录的日志存放目录。（可忽略）

`./two-stage`为特征提取+分类器方法文件路径
- `./two-stage/crop.py`使用 pathlib 新建文件夹，用于后续人脸的存放；使用 opencv 的 VideoCapture 函数逐帧读取视频文件；使用 MTCNN 对隔一定间距的帧进行识别，并存放到响应的文件夹。
- `./two-stage/pca.py`对人脸数据集进行加载和划分，一部分用作测试集，一部分用作训练集；将人脸转成 np.ndarray 格式后，使用 pca 方法对特征进行提取，之后通过特征距离和 SVM 两种方法进行判断。
- `./two-stage/facenet.py`对人脸数据集进行加载和划分，一部分用作测试集，一部分用作训练集；将人脸转成 np.ndarray 格式后，使用 facenet 预训练模型对特征进行提取，之后通过特征距离和 SVM 两种方法进行判断。
- `./two-stage/video-recog-pca-svm.py`为使用pca+svm方法实现输入视频到识别输出流程的程序。
- `./two-stage/video-recog-facenet-svm.py`为使用facenet+svm方法实现输入视频到识别输出流程的程序。

`./resource`为项目中使用的资源文件路径
- `./resource/MS_yahei.ttf`为输出图像中标注的字体资源（微软雅黑）