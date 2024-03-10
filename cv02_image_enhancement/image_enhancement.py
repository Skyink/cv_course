import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

vis_stds = []
infrar_stds = []



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
    return img_hist_result

# 1 直方图均匀化、CLAHE
# 2 GAMMA矫正，不同参数（大于1，小于1）
# 3 比较CLAHE和GAMMA

# ==== main ========================================================

# 读取灰度图像
img = cv.imread('image/doge2.png', cv.IMREAD_GRAYSCALE)
# 显示原始图片
# plt.imshow(img, cmap='gray')
# plt.show()

# ==== part one ======
# 统计原始图像的直方图，灰度256阶
plt.hist(img.flatten(), 256)
plt.show()

# 图像均衡化
img_equalize_hist = equalize_hist(img)
img_eh_int = img_equalize_hist.astype(np.uint8)

# 显示均衡化图
plt.imshow(img_equalize_hist, cmap='gray')
plt.show()
# 对比原始图像与均衡化图
# img_eh_compare = np.hstack([img, img_eh_int])
# plt.imshow(img_eh_compare, cmap='gray')
# plt.show()
# img_show('img_eh_compare', img_eh_compare)

# 统计均衡化图的直方图
plt.hist(img_equalize_hist.flatten(), 256)
plt.show()

# CLAHE 图像增强
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
img_clahe = clahe.apply(img)

# 显示clahe增强图像
img_clahe_compare = np.hstack([img, img_clahe])
img_show('img_clahe_compare', img_clahe_compare)

# 对比直方图均衡化与clahe增强
compare_he_clahe = np.hstack([img, img_eh_int, img_clahe])
img_show('compare_he_clahe', compare_he_clahe)
# 保存至本地
cv.imwrite('image/compare_he_clahe.png',compare_he_clahe)

# ==== part two ======
# Gamma矫正
# 除以像素最大值进行归一化
img_float = img/float(np.max(img))
# gamma=1
img_gamma_eq = np.power(img_float, 1) * float(np.max(img))
img_gamma_eq_int = img_gamma_eq.astype(np.uint8)
# gamma>1
img_gamma_gr1 = np.power(img_float, 1.5) * float(np.max(img))
img_gamma_gr1_int = img_gamma_gr1.astype(np.uint8)
img_gamma_gr2 = np.power(img_float, 2) * float(np.max(img))
img_gamma_gr2_int = img_gamma_gr2.astype(np.uint8)
img_gamma_gr3 = np.power(img_float, 5) * float(np.max(img))
img_gamma_gr3_int = img_gamma_gr3.astype(np.uint8)
img_gamma_gr4 = np.power(img_float, 10) * float(np.max(img))
img_gamma_gr4_int = img_gamma_gr4.astype(np.uint8)
# gamma<1
img_gamma_ls1 = np.power(img_float, 1/1.5) * float(np.max(img))
img_gamma_ls1_int = img_gamma_ls1.astype(np.uint8)
img_gamma_ls2 = np.power(img_float, 1/2) * float(np.max(img))
img_gamma_ls2_int = img_gamma_ls2.astype(np.uint8)
img_gamma_ls3 = np.power(img_float, 1/5) * float(np.max(img))
img_gamma_ls3_int = img_gamma_ls3.astype(np.uint8)
img_gamma_ls4 = np.power(img_float, 1/10) * float(np.max(img))
img_gamma_ls4_int = img_gamma_ls4.astype(np.uint8)
# 对比
compare_gamma_1 = np.hstack([img, img_gamma_eq_int, img_gamma_gr1_int, img_gamma_gr2_int, img_gamma_gr3_int, img_gamma_gr4_int])
compare_gamma_2 = np.hstack([img, img_gamma_eq_int, img_gamma_ls1_int, img_gamma_ls2_int, img_gamma_ls3_int, img_gamma_ls4_int])
compare_gamma = np.vstack([compare_gamma_1, compare_gamma_2])
img_show('compare_gamma', compare_gamma)
cv.imwrite('image/compare_gamma.png',compare_gamma)

# ==== part three ======
# TODO: compara CLAHE and GAMMA

