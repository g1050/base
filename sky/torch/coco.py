import torch
import torch.utils
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
data_dir = "../download/coco"
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
# train_dataset = datasets.CocoDetection(root=data_dir,annFile=f"{data_dir}/annotations/instances_train2017.json",transform=transform)
val_dataset = datasets.CocoDetection(root=data_dir+"/val2017",annFile=f"{data_dir}/annotations/instances_val2017.json",transform=transform)
batch_size = 1
# train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=True)

step = 0
for data in val_loader:
    imgs,target = data
    print(type(imgs),type(target))
    # imgs = transforms.ToTensor(Image.fromarray(
    #                         np.transpose(vutils.make_grid(data[:4], padding=2, normalize=True).cpu(), (1, 2, 0))))
    # print(target[:4])\
    writer.add_images("coco",imgs,step)
    step += 1
    break
writer.close()

