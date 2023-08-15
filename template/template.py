"""
opencv模板匹配----单目标匹配
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np

# 读取目标图片 optical image [256,256]
target = cv2.imread("ROIs1868_summer_s2_59_p443.png")
# 读取模板图片 SAR image [192,192]
template = cv2.imread("ROIs1868_summer_s1_59_p443.png")
# plt.imshow(template, cmap='gray')
# plt.show()
h, w = template.shape[:2]
# 相关系数匹配方法: cv2.TM_CCOEFF
# 返回矩阵大小 w_r-w_f+1
res = cv2.matchTemplate(target, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

left_top = max_loc   # heatmap最大位置索引 对应于 光学图像中左上角坐标
right_bottom = (left_top[0] + w, left_top[1] + h)   # 右下角坐标
center_x = max_loc[0] + w/2 -1  # 匹配区域中心x
center_y = max_loc[1] + h/2 -1  # 匹配区域中心y
print('匹配中心在光学图像中的坐标为：x=%d, y=%d' %(int(center_x),int(center_y)))


def transfer_to_8bit(image):
    """
    将16位图像转为8位图像
    """
    min_ = np.min(image)
    max_ = np.max(image)
    image_8bit = np.array(np.rint(255 * ((image - min_) / (max_ - min_))), dtype=np.uint8)
    return image_8bit


H8bit = transfer_to_8bit(res)
cv2.imwrite('heatmap.jpg', H8bit)
# plt.imshow(H8bit, cmap='jet')
# plt.show()

plt.subplot(131), plt.imshow(template, cmap='gray')
plt.title('SAR img')

targetRGB = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
cv2.rectangle(targetRGB, left_top, right_bottom, (0, 0, 255), 2)  # 画出矩形位置
point_size = 3
point_color = (255, 0, 0) # RGB
thickness = 4 #  0 、4、8
cv2.circle(targetRGB, (int(center_x),int(center_y)), point_size, point_color, thickness)  # 画出匹配区域中心位置
plt.subplot(132), plt.imshow(targetRGB, cmap='gray')
plt.title('optical img')

plt.subplot(133), plt.imshow(res, cmap='jet')
plt.title('Heatmap')
plt.show()
