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
import json
from pprint import pprint
def coco_visualize():
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
        print(imgs.shape)
        # imgs = transforms.ToTensor(Image.fromarray(
        #                         np.transpose(vutils.make_grid(data[:4], padding=2, normalize=True).cpu(), (1, 2, 0))))
        # print(target[:4])\
        writer.add_images("coco",imgs,step)
        step += 1
        break
    writer.close()

# {
#     "info":   info, # dict
#               info.description 
#               info.url
#               info.version
#               info.year
#               info.contributor
#               info.date_created
#      "licenses": [license], # list ，内部是dict
#      "images": [image], # list ，内部是dict
#      "annotations": [annotation], # list ，内部是dict
#      "categories": # list ，内部是dict
# }

############### images字段 ###############
# "images": [
#             {
#              "license":4 #可以忽略
#             "file_name":000.jpg #可以忽略
#             "coco_url":"http://****" #可以忽略
#              "id": 1, 
#              "file_name": "000.tif", 
#              "width": 48.0, 
#              "height": 112.0
#              "date_captured":"2022-02-02 17:02:02" #可以忽略
#              "flickl_url":"http://****" #可以忽略
#             }
#             ...
#             ...
#             ]

############### annotations字段 ###############
# {'area': 702.1057499999998, 区域面积
#  'bbox': [473.07, 395.93, 38.65, 28.67], 定位框 （x,y,w,h） 但是voc中是（x1,y1,x2,y2）
#  'category_id': 18, 类别id
#  'id': 1768, 对象id，一个图片可能有多个对象
#  'image_id': 289343, 图片id
#  'iscrowd': 0, # 0 单个对象,1 表示是群体多个对象标在一个框里
#  'segmentation': [[510.66, 分割的多边形框的点，每两个数据是一个点
#                    423.01,
#                    ...,
#  ]]
# }

############### annotations字段 ###############
# {'id': 1, 'name': 'person', 'supercategory': 'person'} 
# {'id': 2, 'name': 'bicycle', 'supercategory': 'vehicle'} supercategory 主类别

def coco_annotations_format():
    with open(f"../download/coco/annotations/image_info_test2017.json") as f:
        data = json.load(f)
    # print(data)
    def print_keys(obj, parent_key=''):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                print(new_key)
                print_keys(v, new_key)
        elif isinstance(obj, list):
            pass
            # for i, item in enumerate(obj):
            #     new_key = f"{parent_key}[{i}]"
            #     print_keys(item, new_key)
    print_keys(data)
    # pprint(data["annotations"][0])
    # pprint(data["categories"][0])
    # pprint(data["categories"][1])
    # print(len(data["categories"])) # 80类别
    print(len(data["images"])) # 测试图片数量 val->5000 train->118287(11.8w) test->40670
    # print(len(data["annotations"])) # 标注数量
    
# todo: 分割，关键点，检测
if __name__=="__main__":
    coco_annotations_format()