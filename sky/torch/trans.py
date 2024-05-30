from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter
wrtiter = SummaryWriter("logs")

img_path = "data/img.png"
img = Image.open(img_path) # PIL image
img_tensor = transforms.ToTensor()(img)
print(img_tensor)
img_np = np.array(img)
print(img_np)

# to tensor
img_np = cv2.imread(img_path)
img_tensor = transforms.ToTensor()(img)
print(img_tensor)
print((img_tensor[0][0][0]-0.5)/0.5)

# normalize
img_tensor_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])(img_tensor)
print(img_tensor_norm[0][0][0])
wrtiter.add_image("img_tensor image",img_tensor,1)
wrtiter.add_image("img_tensor_norm image",img_tensor_norm,1)

# resize
print(img.size)
# 只给一个参数时，则匹配hw中的较小值然后按照等比例缩放
img_resize =  transforms.Resize((512,512))(img)
img_resize = transforms.ToTensor()(img_resize)
print(img_resize.shape)

# compose
# transforms.Compose([]) transform操作组合

# 随机裁剪
# transforms.RandomCrop

wrtiter.add_image("img_resize image",img_resize,1) # tensorboard绘制时候使用的是torch.tensor

wrtiter.close()