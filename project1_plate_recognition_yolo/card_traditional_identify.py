#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ==========================导入库==============================
import cv2
# from matplotlib import pyplot as plt
import numpy as np
import glob


# In[ ]:
# 读取车牌
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


# ==========================一个字构成一个整体====================
def GetOne(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray_image", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    cv2.imshow("blurred_image", blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ret, threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow("threshold_image", threshold_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated_image = cv2.dilate(threshold_image, kernel)
    cv2.imshow("dilated_image", dilated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("threshold_image", threshold_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    height, width = dilated_image.shape
    corners = [dilated_image[0, 0], dilated_image[0, width - 1], dilated_image[height - 1, 0],
               dilated_image[height - 1, width - 1]]
    avg_corner_value = np.mean(corners)

    # 绿牌：白底黑字
    # 蓝牌：黑底白字（我们是以这个为基准）

    if avg_corner_value > 127:
        ret, dilated_image = cv2.threshold(dilated_image, 127, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow("last", dilated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return dilated_image, threshold_image


# ===========拆分车牌函数，将车牌内各个字符分离==================
def splitPlate(image):
    original = image.copy()
    image, binary_image = GetOne(image)  # 预处理车牌图像

    # 找轮廓，返回轮廓本身（contours）和每一个轮廓的层次结构(hierarchy)

    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # 画出轮廓（仅可以画出黑底白字的）
    contoured_image = cv2.drawContours(original.copy(), contours, -1, (0, 0, 255), 1)

    cv2.imshow("Contours", contoured_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Number of contours found:", len(contours))  # 输出轮廓数量
    # 遍历所有轮廓，寻找最小包围框
    chars = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        chars.append((x, y, w, h))

        # 在原始图像上画出轮廓
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 1)

    # 将包围框按照x轴坐标值排序（自左向右），使得字符可以从左到右依次处理
    chars = sorted(chars, key=lambda char: char[0])

    # 筛选出可能是字符的轮廓
    plate_chars = []
    for char in chars:
        # char[0]~char[4]依次表示矩形左上角的x坐标，y坐标，宽度，高度
        if char[3] > (char[2] * 1.5) and char[3] < (char[2] * 8) and char[2] > 3:
            char_image = binary_image[char[1]:char[1] + char[3], char[0]:char[0] + char[2]]
            cv2.imshow("123", char_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            plate_chars.append(char_image)
    return plate_chars


# =================模板，部分省份，使用字典表示==============================
templateDict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H',
                18: 'J', 19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'P', 24: 'Q', 25: 'R',
                26: 'S', 27: 'T', 28: 'U', 29: 'V', 30: 'W', 31: 'X', 32: 'Y', 33: 'Z',
                34: '京', 35: '津', 36: '冀', 37: '晋', 38: '蒙', 39: '辽', 40: '吉', 41: '黑',
                42: '沪', 43: '苏', 44: '浙', 45: '皖', 46: '闽', 47: '赣', 48: '鲁', 49: '豫',
                50: '鄂', 51: '湘', 52: '粤', 53: '桂', 54: '琼', 55: '渝', 56: '川', 57: '贵',
                58: '云', 59: '藏', 60: '陕', 61: '甘', 62: '青', 63: '宁', 64: '新',
                65: '港', 66: '澳', 67: '台'}


# ==================获取所有字符的路径信息===================
def getcharacters():
    c = []
    for i in range(0, 68):
        words = []
        words.extend(glob.glob('template/' + templateDict.get(i) + '/*.*'))
        c.append(words)
    return c


# =============计算匹配值函数=====================
def getMatchValue(template, image):
    # 读取模板图像
    # templateImage=cv2.imread(template)   #cv2读取中文文件名不友好
    # np.fromfilel()从文件读数据并将其转换为NumPy数组
    # template为文件路径
    # cv2.imdecode(..., 1):
    # cv2.imdecode是OpenCV 的一个函数，用于从内存缓存数据解码图像。它是处理从网络或复杂文件系统中读取的原始图像数据的常用方法。第一个参数是包含图像数据的字节数组，这里是由
    # np.fromfile，读取的数据。
    # 第二个参数
    # 1表示加载图像为彩色模式。在OpenCV中，这个标志对应于cv2.IMREAD_COLOR，表示加载一个彩色图像，任何透明度的通道都会被忽略。它的值实际上是1
    templateImage = cv2.imdecode(np.fromfile(template, dtype=np.uint8), 1)

    # 模板图像色彩空间转换，BGR-->灰度
    templateImage = cv2.cvtColor(templateImage, cv2.COLOR_BGR2GRAY)

    # 模板图像阈值处理， 灰度-->二值（0~255->0/1）
    ret, templateImage = cv2.threshold(templateImage, 0, 255, cv2.THRESH_OTSU)

    # 获取待识别图像的尺寸
    height, width = image.shape

    # 将模板图像调整为与待识别图像尺寸一致
    templateImage = cv2.resize(templateImage, (width, height))

    # 计算模板图像、待识别图像的模板匹配值
    result = cv2.matchTemplate(image, templateImage, cv2.TM_CCOEFF)
    # 将计算结果返回
    return result[0][0]


# ===========对车牌内字符进行识别====================
# plates，要识别的字符集，
# 也就是从车牌图像“GUA211”中分离出来的每一个字符的图像"G","U","A","2","1","1"
# chars，所有字符的模板集合，也就是0-9，A-Z，京-台，每一个字符模板
def matchChars(plates, chars):
    results = []  # 存储所有的识别结果
    # 最外层循环：逐个遍历要识别的字符。
    # 例如，逐个遍历从车牌图像“GUA211”中分离出来的每一个字符的图像
    # 如"G","U","A","2","1","1"
    # plateChar分别存储，"G","U","A","2","1","1"
    index = 0
    for plateChar in plates:  # 逐个遍历要识别的字符
        # bestMatch，存储的是待识别字符与每个特征字符的所有模板中最匹配的模板
        # 例如，待识别图像“G”，与所有的字符0-9，A-Z，京-台，每一个字符最匹配的模板

        cv2.imwrite(f'image_{index}.png', plateChar)
        index=index+1
        bestMatch = []  # 最佳匹配
        # 中间层循环：针对模板内的字符，进行逐个遍历（每次循环针对一个特定的字符），
        # words 对应的是每一个字符（例如字符A）的所有模板
        for words in chars:  # 遍历字符。chars：所有模板，words：某个字符的所有模板
            # match，存储的是每个特征字符的所有匹配值
            # 例如：待识别图像“G”，与字符7的所有模板的匹配值
            match = []  # 每个字符的匹配值
            # 最内层循环：针对的是单个字符的所有模板，找到最佳的模板
            #  word对应的是单个模板
            for word in words:  # 遍历模板。words：某个字符所有模板，word单个模板
                result = getMatchValue(word, plateChar)
                match.append(result)
            bestMatch.append(max(match))  # 将每个字符模板的最佳匹配加入bestMatch
        i = bestMatch.index(max(bestMatch))  # i是最佳匹配的字符模板的索引值
        r = templateDict[i]  # r是单个待识别字符的识别结果
        results.append(r)  # 将每一个分割字符的识别结果加入到results内
    return results  # 返回所有的识别结果


plates = loadPlates("result/crops")
chars = getcharacters()  # 加载所有模板文件（文件名）
for plate in plates:
    # plateChars是所有车牌的字符集合
    plateChars = splitPlate(plate)  # 分割车牌，将每个字符独立出来

    results = matchChars(plateChars, chars)  # 使用模板chars逐个识别字符集plates
    results = "".join(results)  # 将列表转换为字符串
    print("识别结果为：", results)  # 输出识别结果
