from torch.utils.data import Dataset
# import cv2
from PIL import Image
import os
class Mydata(Dataset):
    # 初始化
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)
    # 读取一张图
    def __getitem__(self, index):
        image_name = self.img_path[index]
        img_item_path = os.path.join(self.path,image_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label
    
    def __len__(self,):
        return len(self.img_path)

root_dir = '../download/hymenoptera_data/train'
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = Mydata(root_dir,ants_label_dir)
bees_dataset = Mydata(root_dir,bees_label_dir)

# 顺序：ants + bees
train_dataset = ants_dataset + bees_dataset

# ant_img,ant_label = ants_dataset[0]
# ant_img.save('data/img.png')
# print(ant_label)
