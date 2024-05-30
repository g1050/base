import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="data",train=True,transform=trans,download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="data",train=True,transform=trans,download=True)

def get_label(labels):
    text_labels = ['t-shirt','trouser','pullover','dress','coat',
                   'sandal','shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]
print(f"train lenth {len(mnist_train)} test length {len(mnist_test)}")
# 通道1，28x28的灰度图 label 0-9 10个类别
print(f"img shape {mnist_train[0][0].shape} label {mnist_train[0][1],get_label([mnist_train[0][1]])}")

# 可视化
def show_image(img,label):
    d2l.plt.imshow(img,cmap='gray',vmin=0,vmax=255)
    d2l.plt.axis('off')
    d2l.plt.savefig('data/img.png')
# print(mnist_train[0][0])
# show_image(mnist_train[0][0].reshape(28,28)*255,get_label([mnist_train[0][1]])[0]) # 图像中存储的是归一化的数据，需要成255

# 小批量读入数据集
batch_size = 1024
def get_daloader_workers():
    return 4
train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_daloader_workers())
timer = d2l.Timer()
for x,y in train_iter:
    # continue
    print(f"x {x.shape} y {y[0]}") # x的shape N,C,H,W
    break
print(f"time {timer.stop()}sec")



    