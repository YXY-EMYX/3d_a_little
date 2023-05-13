import cv2
import numpy as np
from PIL import Image

def bump_mapping(image, kernel_size=5, intensity=1, blur_size=3):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算x和y方向的梯度
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # 计算梯度的大小和方向
    magnitude, angle = cv2.cartToPolar(sobel_x, sobel_y)

    # 对梯度进行高斯模糊
    magnitude = cv2.GaussianBlur(magnitude, (blur_size, blur_size), 0)

    # 根据梯度调整图片的亮度
    adjusted_image = np.float32(image) + intensity * magnitude.reshape(*magnitude.shape, 1)

    # 裁剪数值范围以确保它们位于[0, 255]之间
    adjusted_image = np.clip(adjusted_image, 0, 255)

    return adjusted_image.astype(np.uint8)

# 使用Pillow读取图像
pil_image = Image.open(r'C:\Users\YXY\PycharmProjects\pythonProject3\测试图片1.jpg')

# 将PIL图像转换为NumPy数组（OpenCV格式）
opencv_image = np.array(pil_image)[:, :, ::-1].copy()

# 应用凹凸映射效果
bump_mapped_image = bump_mapping(opencv_image, intensity=0.02, blur_size=3)

# 显示原图和处理后的图片
cv2.imshow('Original Image', opencv_image)
cv2.imshow('Bump Mapped Image', bump_mapped_image)

# 等待按键，然后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
