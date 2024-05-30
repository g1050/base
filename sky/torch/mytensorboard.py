from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("data")
# for i in range(100):
#     writer.add_scalar("y=2x",2*i,i) # (title,value[y轴],step[x轴])
# writer.close()

img_path1 = "data/img.png" 
img_PIL1 = Image.open(img_path1)
img_array1 = np.array(img_PIL1)

img_path2 = "data/img.png" 
img_PIL2 = Image.open(img_path2)
img_array2 = np.array(img_PIL2)

# writer = SummaryWriter("logs") 
# 多张图片拼接的时候用add_images
writer.add_image("test",img_array1,1,dataformats="HWC") # 1 表示该图片在第1步
writer.add_image("test",img_array2,2,dataformats="HWC") # 2 表示该图片在第2步                   
writer.close()