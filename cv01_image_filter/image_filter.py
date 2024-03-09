import cv2 as cv
import numpy as np
import math


# 显示图片函数
def img_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 自定义高斯核函数，默认3*3，sigma=1.5
def my_gauss_kernel(size=3, sigma=1.5):
    my_kernel = np.zeros((size, size), np.float32)
    for i in range(size):
        for j in range(size):
            numerator = math.pow(i-size-1, 2) + math.pow(j-size-1, 2)
            # 求高斯卷积
            my_kernel[i, j] = math.exp(-numerator/(2*math.pow(sigma, 2))) / (2*math.pi*sigma**2)
    sum = np.sum(my_kernel)     # 求和
    kernel = my_kernel/sum      # 归一化
    return kernel


# 自定义高斯滤波函数（单通道）
def my_gauss_filter(img, ksize=3, sigma=1.5, kernel=[]):
    height = img.shape[0]
    width = img.shape[1]
    result_img = np.zeros((height, width), np.uint8)
    # 计算高斯卷积核
    half_ksize = int(ksize/2)
    ksize_up = half_ksize + 1
    ksize_below = 0 - half_ksize
    for i in range(half_ksize, height - half_ksize):
        for j in range(half_ksize, width - half_ksize):
            sum = 0
            for k in range(ksize_below, ksize_up):
                for l in range(ksize_below, ksize_up):
                    sum += img[i + k, j + l] * kernel[k + 1, l + 1]  # 高斯滤波
            result_img[i, j] = sum
    return result_img


# 自定义高斯滤波函数（灰度图像）
def my_gauss_filter_single(img, ksize=3, sigma=1.5):
    # 分别对三通道进行滤波
    kernel = my_gauss_kernel(ksize, sigma)
    result_img_bgr = my_gauss_filter(img, ksize, sigma, kernel)
    return result_img_bgr


# 自定义高斯滤波函数（BGR彩色图像）
def my_gauss_filter_bgr(img, ksize=3, sigma=1.5):
    img_b, img_g, img_r = cv.split(img)
    # 分别对三通道进行滤波
    kernel = my_gauss_kernel(ksize, sigma)
    ch_b = my_gauss_filter(img_b, ksize, sigma, kernel)
    ch_g = my_gauss_filter(img_g, ksize, sigma, kernel)
    ch_r = my_gauss_filter(img_r, ksize, sigma, kernel)
    result_img_bgr = cv.merge([ch_b, ch_g, ch_r])
    return result_img_bgr


# 自定义高斯滤波函数（HSV彩色图像）
def my_gauss_filter_hsv(img, ksize=3, sigma=1.5):
    # 获得HSV彩色图像，拆分通道
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_h, img_s, img_v = cv.split(img)
    # 仅对V通道进行滤波
    kernel = my_gauss_kernel(ksize, sigma)
    ch_h = img_h
    ch_s = img_s
    ch_v = my_gauss_filter(img_v, ksize, sigma, kernel)
    result_img_hsv = cv.merge([ch_h, ch_s, ch_v])
    # 转换回BGR色彩
    result_img = cv.cvtColor(result_img_hsv, cv.COLOR_HSV2BGR)
    return result_img


# ==== main ========================================================

# 设置参数
K_SIZE = 3
SIGMA = 1.5

# 读取图像(彩色)
img = cv.imread('image/sighman.png')
# 处理为灰度图像
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# 处理灰度图像 img_gray
# 使用自定义高斯滤波函数
img_filter_gray = my_gauss_filter_single(img_gray, K_SIZE, SIGMA)
# 使用库函数
img_guassian_gray = cv.GaussianBlur(img_gray, (K_SIZE, K_SIZE), SIGMA)
# 对比灰度滤波效果
img_compare_gray = np.hstack([img_gray, img_filter_gray, img_guassian_gray])
img_show('compare_gray', img_compare_gray)


# 处理彩色图像 img
# 使用自定义高斯滤波函数(BGR)
img_filter_bgr = my_gauss_filter_bgr(img, K_SIZE, SIGMA)
# 使用自定义高斯滤波函数(HSV)
img_filter_hsv = my_gauss_filter_hsv(img, K_SIZE, SIGMA)
# 使用库函数
img_guassian_bgr = cv.GaussianBlur(img, (K_SIZE, K_SIZE), SIGMA)

# 对比彩色滤波效果
img_compare_color = np.hstack([img, img_filter_bgr, img_filter_hsv, img_guassian_bgr])
img_show('compare_color', img_compare_color)
