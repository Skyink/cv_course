import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os


# 显示图片函数
def img_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 对灰度图像进行直方图均衡化
def equalize_hist(img, nbr_bins=256):
    # 图像直方图统计
    img_hist, bins = np.histogram(img.flatten(), nbr_bins)
    # 累积分布函数
    cdf = img_hist.cumsum()
    cdf = 255.0 * cdf / cdf[-1]
    # 使用累积分布函数的线性插值，计算新的像素值
    img2 = np.interp(img.flatten(), bins[:-1], cdf)  # 分段线性插值函数
    img_hist_result = img2.reshape(img.shape)
    return img_hist_result, cdf


# Gamma校正
def adjust_gamma(img, gamma):
    # 除以像素最大值进行归一化
    img_float = img / float(np.max(img))
    img_gamma = np.power(img_float, gamma) * float(np.max(img))
    img_gamma_int = img_gamma.astype(np.uint8)
    return img_gamma_int


# ==== main ========================================================

# 1 直方图均衡化
# 2 CLAHE
# 3 GAMMA校正，不同参数（大于1，小于1）

# 读取灰度图像
img = cv.imread('image/spaceman.png', cv.IMREAD_GRAYSCALE)
cv.imwrite('image/img_gray.png', img)


# ==== part one: equalize hist ======
# 直方图均衡化
img_equalize_hist, cdf = equalize_hist(img)
img_eh_int = img_equalize_hist.astype(np.uint8)
# 使用库函数
img_eh_opencv = cv.equalizeHist(img)

# 对比 原始图像与均衡化图
img_eh_compare = np.hstack([img, img_eh_int, img_eh_opencv])
img_show('img_eh_compare', img_eh_compare)
cv.imwrite('image/compare_eh.png', img_eh_compare)

# 统计原始图像的直方图，灰度256阶
plt.subplot(2, 1, 1)
plt.hist(img.flatten(), 256)
plt.suptitle('Original Image Gray Hist')
# 累计分布函数
plt.subplot(2, 1, 2)
plt.plot(cdf, color='r', label='cumulative distribution function')
plt.legend()
# plt.title('Cumulative Distribution Function')
plt.savefig('image/grayhist_cdf_original_image.png')
plt.show()

# 统计均衡化图的直方图（自定义函数与库函数）
plt.hist(img_eh_int.flatten(), 256, label='custom function')
plt.legend(loc=1)
plt.twinx()
plt.hist(img_eh_opencv.flatten(), 256, color='r', label='opencv function')
plt.legend(loc=2)
plt.title('Equalized Image Gray Hist')
plt.savefig('image/gray_hist_equalized_image.png')
plt.show()


# ==== part two: Contrast Limited AHE (CLAHE) ======
# 限制对比对的自适应直方图均衡化
# clipLimit参数表示对比度限制，默认值为40
# tileGridSize参数表示分块的大小，默认为8*8
clahe = cv.createCLAHE()    # 默认参数
img_clahe = clahe.apply(img)

# 对比 原始图像和CLAHE增强图像
img_clahe_compare = np.hstack([img, img_clahe])
img_show('img_clahe_compare', img_clahe_compare)
cv.imwrite("image/compare_clahe.png", img_clahe_compare)

# 统计CLAHE图的直方图
plt.hist(img_clahe.flatten(), 256)
plt.title('CLAHE Image Gray Hist')
plt.savefig('image/gray_hist_clahe_image.png')
plt.show()

# 对比 EH与CLAHE（库函数默认参数）
compare_eh_clahe = np.hstack([img, img_eh_opencv, img_clahe])
img_show('compare_he_clahe', compare_eh_clahe)
# 保存至本地
cv.imwrite('image/compare_he_clahe.png', compare_eh_clahe)

# 调整CLAHE参数比较效果
# 参数clipLimit
clahe_limit2 = cv.createCLAHE(clipLimit=2.0)
clahe_limit5 = cv.createCLAHE(clipLimit=5.0)
clahe_limit10 = cv.createCLAHE(clipLimit=10.0)
clahe_limit20 = cv.createCLAHE(clipLimit=20.0)
clahe_limit40 = cv.createCLAHE(clipLimit=40.0)
clahe_limit100 = cv.createCLAHE(clipLimit=100.0)
img_clahe_limit2 = clahe_limit2.apply(img)
img_clahe_limit5 = clahe_limit5.apply(img)
img_clahe_limit10 = clahe_limit10.apply(img)
img_clahe_limit20 = clahe_limit20.apply(img)
img_clahe_limit40 = clahe_limit40.apply(img)
img_clahe_limit100 = clahe_limit100.apply(img)
# 显示
img_clahe_compare_limit = np.hstack([img, img_clahe_limit2, img_clahe_limit5, img_clahe_limit10, img_clahe_limit20, img_clahe_limit40, img_clahe_limit100])
img_show('img_clahe_compare_limit', img_clahe_compare_limit)
cv.imwrite('image/img_clahe_compare_limit.png', img_clahe_compare_limit)

# clahe增强的直方图（limit）
f, ax = plt.subplots(3, 3, sharex='col',sharey='row')
plt.subplots_adjust(hspace=0.3)
# 删除空余
f.delaxes(ax[-0][-1])
f.delaxes(ax[-0][-2])
# 原图
f.suptitle('Gray Hist of Different CLAHE ClipLimit')
ax[0][0].hist(img.flatten(), 256)
ax[0][0].set_title('original img')
# limit = 2
ax[1][0].hist(img_clahe_limit2.flatten(), 256)
ax[1][0].set_title('clipLimit=2')
# limit = 5
ax[1][1].hist(img_clahe_limit5.flatten(), 256)
ax[1][1].set_title('clipLimit=5')
# limit = 10
ax[1][2].hist(img_clahe_limit10.flatten(), 256)
ax[1][2].set_title('clipLimit=10')
# limit = 20
ax[2][0].hist(img_clahe_limit20.flatten(), 256)
ax[2][0].set_title('clipLimit=2')
# limit = 40
ax[2][1].hist(img_clahe_limit40.flatten(), 256)
ax[2][1].set_title('clipLimit=5')
# limit = 100
ax[2][2].hist(img_clahe_limit100.flatten(), 256)
ax[2][2].set_title('clipLimit=10')
# 保存及显示
plt.savefig('image/hist_compare_clahe_limit.png')
plt.show()

# 参数tileGridSize
clahe_size2 = cv.createCLAHE(tileGridSize=(2,2))
clahe_size4 = cv.createCLAHE(tileGridSize=(4,4))
clahe_size8 = cv.createCLAHE(tileGridSize=(8,8))
clahe_size16 = cv.createCLAHE(tileGridSize=(16,16))
img_clahe_size2 = clahe_size2.apply(img)
img_clahe_size4 = clahe_size4.apply(img)
img_clahe_size8 = clahe_size8.apply(img)
img_clahe_size16 = clahe_size16.apply(img)
img_clahe_compare_size = np.hstack([img, img_clahe_size2, img_clahe_size4, img_clahe_size8, img_clahe_size16])
img_show('img_clahe_compare_size', img_clahe_compare_size)
cv.imwrite('image/img_clahe_compare_size.png', img_clahe_compare_size)

# clahe增强的直方图（size）
f, ax = plt.subplots(3, 2, sharex='col',sharey='row')
plt.subplots_adjust(hspace=0.3)
# 删除空余
f.delaxes(ax[-0][-1])
# 原图
f.suptitle('Gray Hist of Different CLAHE TileGridSize')
ax[0][0].hist(img.flatten(), 256)
ax[0][0].set_title('original img')
# size = 2
ax[1][0].hist(img_clahe_size2.flatten(), 256)
ax[1][0].set_title('tileGridSize=2')
# size = 4
ax[1][1].hist(img_clahe_size4.flatten(), 256)
ax[1][1].set_title('tileGridSize=5')
# size = 8
ax[2][0].hist(img_clahe_size8.flatten(), 256)
ax[2][0].set_title('tileGridSize=10')
# size = 16
ax[2][1].hist(img_clahe_size16.flatten(), 256)
ax[2][1].set_title('tileGridSize=2')
# 保存及显示
plt.savefig('image/hist_compare_clahe_size.png')
plt.show()


# ==== part three: gamma adjust ======
# Gamma校正
# gamma=1
img_gamma_eq = adjust_gamma(img, 1)
# gamma>1
img_gamma_gr1 = adjust_gamma(img, 1.5)
img_gamma_gr2 = adjust_gamma(img, 2)
img_gamma_gr3 = adjust_gamma(img, 5)
img_gamma_gr4 = adjust_gamma(img, 10)
# gamma<1
img_gamma_ls1 = adjust_gamma(img, 1 / 1.5)
img_gamma_ls2 = adjust_gamma(img, 1 / 2)
img_gamma_ls3 = adjust_gamma(img, 1 / 5)
img_gamma_ls4 = adjust_gamma(img, 1 / 10)
# 对比
compare_gamma_1 = np.hstack([img, img_gamma_eq, img_gamma_gr1, img_gamma_gr2, img_gamma_gr3, img_gamma_gr4])
compare_gamma_2 = np.hstack([img, img_gamma_eq, img_gamma_ls1, img_gamma_ls2, img_gamma_ls3, img_gamma_ls4])
compare_gamma = np.vstack([compare_gamma_1, compare_gamma_2])
img_show('compare_gamma', compare_gamma)
cv.imwrite('image/compare_gamma.png',compare_gamma)


# ==== 比较 CLAHE 和 GAMMA校正 效果 ======
# 图1
img1 = cv.imread('image/spaceman.png', cv.IMREAD_GRAYSCALE)
clahe1 = cv.createCLAHE()
img_1_clahe = clahe1.apply(img1)
img_1_gamma_gr = adjust_gamma(img1, 2)
img_1_gamma_ls = adjust_gamma(img1, 1/2)
img_compare_clahe_gamma_1 = np.hstack([img1, img_1_clahe, img_1_gamma_gr, img_1_gamma_ls])
img_show('img_compare_clahe_gamma1', img_compare_clahe_gamma_1)
cv.imwrite('image/img_compare_clahe_gamma1.png', img_compare_clahe_gamma_1)

# 图2
img2 = cv.imread('image/doge1.png', cv.IMREAD_GRAYSCALE)
clahe2 = cv.createCLAHE()
img_2_clahe = clahe2.apply(img2)
img_2_gamma_gr = adjust_gamma(img2, 2)
img_2_gamma_ls = adjust_gamma(img2, 1/2)
img_compare_clahe_gamma_2 = np.hstack([img2, img_2_clahe, img_2_gamma_gr, img_2_gamma_ls])
img_show('img_compare_clahe_gamma2', img_compare_clahe_gamma_2)
cv.imwrite('image/img_compare_clahe_gamma2.png', img_compare_clahe_gamma_2)

# 图3
img3 = cv.imread('image/doge2.png', cv.IMREAD_GRAYSCALE)
clahe3 = cv.createCLAHE()
img_3_clahe = clahe3.apply(img3)
img_3_gamma_gr = adjust_gamma(img3, 2)
img_3_gamma_ls = adjust_gamma(img3, 1/2)
img_compare_clahe_gamma_3 = np.hstack([img3, img_3_clahe, img_3_gamma_gr, img_3_gamma_ls])
img_show('img_compare_clahe_gamma3', img_compare_clahe_gamma_3)
cv.imwrite('image/img_compare_clahe_gamma3.png', img_compare_clahe_gamma_3)
