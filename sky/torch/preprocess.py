# 图像预处理格式

# opencv默认使用bgr格式,numpy使用rgb格式
# 两者都是使用的packed格式,排布为(h,w,c),而planar的排布为(c,h,w)
# 两种框架默认使用的dtype为uint8,深度学习一般使用fp32 planar格式,所以需要归一化到(0,1) 即/255转为float32

import numpy as np
import cv2
from PIL import Image

def test_opencv(image):
    # image此时是bgr格式
    print(image.shape) # (100, 300, 3)
    print(image)
    print("-"*20)
    # 保存图像
    cv2.imwrite('output_image.jpg', image)

    # 读取并显示保存的图像以确认
    saved_image = cv2.imread('output_image.jpg')
    print(type(saved_image),saved_image.dtype) # <class 'numpy.ndarray'>
    print(saved_image)
    print("-"*20)

    rgb_image = cv2.cvtColor(saved_image, cv2.COLOR_BGR2RGB)
    print(rgb_image)
    print("-"*20)

def test_PIL(image):
    # image此时是bgr格式
    print(image.shape)  # (2, 3, 3)
    print(image)
    print("-"*20)

    # 将 BGR 转换为 RGB
    image_rgb = image[:, :, ::-1]

    # 转换为 PIL 图像
    pil_image = Image.fromarray(image_rgb)

    # 保存图像
    pil_image.save('output_image_pil.jpg')

    # 读取并显示保存的图像以确认
    saved_image_pil = Image.open('output_image_pil.jpg')
    print(type(saved_image_pil)) # <class 'PIL.JpegImagePlugin.JpegImageFile'>
    print(np.array(saved_image_pil),np.array(saved_image_pil).dtype)
    print("-"*20)

    # 转换为 NumPy 数组并显示 RGB 格式
    rgb_image_pil = np.array(saved_image_pil)
    print(rgb_image_pil)
    print("-"*20)

if __name__ == "__main__":
    height = 2
    width = 4
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # 数据排布格式为H,W,C

    image[:, :, 2] = 128  # 设置 R 通道 栗色
    # image[:, :, 1] = 128  # 设置 G 通道 绿色
    # image[:, :, 0] = 128  # 设置 B 通道 海军

    # bgr和rgb
    # test_opencv(image)
    test_PIL(image)

    # planar和packaed
    # 转换为 planar 格式
    image_planar = np.transpose(image, (2, 0, 1))
    print(image_planar)
    print(image_planar.shape) # (3,2,4)
