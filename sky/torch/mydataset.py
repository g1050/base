# CIFAR

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transform = transforms.Compose([
    transforms.ToTensor()
    ])
train_set = torchvision.datasets.CIFAR10(root="../download",train=True,transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root="../download",train=False,transform=dataset_transform,download=True)

# print(train_set[0]) # PIL image label:3
# print(test_set.classes)
img,target = test_set[0]
print(img)
print(target,test_set.classes[target])
# img.save(f"data/{test_set.classes[target]}.jpg")

print(len(test_set))
# dataloader 加载器
# drop_last 按照batch size选取，最后不足的是否舍弃
test_loader = DataLoader(dataset=test_set,batch_size=64,shuffle=True,num_workers=8,drop_last=True)
writer = SummaryWriter()
for epoch in range(2): # shuffle之后两个epoch取到的顺序是不同的，可以看tensorboard验证
    step = 0
    for data in test_loader:
        imgs,targets = data
        writer.add_images(f"Epoch_{epoch}",imgs,step)
        step += 1
writer.close()
