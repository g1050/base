import torchvision
from torch.utils.data import DataLoader
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../download",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root="../download",train=False,transform=torchvision.transforms.ToTensor(),download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)
print("train_data_size {} \ntest_data_size {}".format(train_data_size,test_data_size))
train_data_loader = DataLoader(train_data,batch_size=64)
test_data_loader = DataLoader(test_data,batch_size=64)

# 搭建神经网络 network.py单独写
class Cifar10(nn.Module):
    def __init__(self):
        super(Cifar10,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32,32,5,padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32,64,5,padding=2)
        self.maxpool3 = nn.MaxPool2d(2) # 宽高减半
        self.flatten1 = nn.Flatten()
        self.linear1 = nn.Linear(1024,64)
        self.linear2 = nn.Linear(64,10)

        self.model1 = nn.Sequential(
            self.conv1,
            self.maxpool1,
            self.conv2,
            self.maxpool2,
            self.conv3,
            self.maxpool3,
            self.flatten1,
            self.linear1,
            self.linear2
        )
    def forward(self,x):
        return self.model1(x)
    



def train():
    # 定义
    lr = 1e-2
    loss = nn.CrossEntropyLoss()
    model = Cifar10()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    total_train_step = 0
    total_test_step = 0
    epoch_num = 10
    writer = SummaryWriter()
    # 训练
    for epoch in range(epoch_num):
        print("-"*10 + f"Start epoch {epoch } train" +"-"*10)
        model.train()
        for data in train_data_loader:
            imgs,targets = data
            outputs = model(imgs)
            l = loss(outputs,targets)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            total_train_step += 1
            if total_train_step%100 == 0:
                print(f"step {total_train_step} loss {l.item()}")
                writer.add_scalar("train_loss",l.item(),total_train_step)
        # 测试
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_data_loader:
                imgs,targets = data
                outputs = model(imgs)
                l = loss(outputs,targets)
                total_test_loss = total_test_loss + l.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy

        rate = total_accuracy / test_data_size
        total_test_step += 1
        writer.add_scalar("test_loss",total_test_loss,total_test_step)
        writer.add_scalar("test_accuracy",rate,total_test_step)
        print(f"test loss {total_test_loss}")
        print(f"test accuracy {rate}") # 测试正确率额
        print("-"*40)
        torch.save(model,f"cifar_epoch_{epoch}.pth")
        print(f"cifar_epoch_{epoch}.pth saved.")

    writer.close()

def train_gpu_1():
    # 定义
    lr = 1e-2
    loss = nn.CrossEntropyLoss()
    model = Cifar10()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    total_train_step = 0
    total_test_step = 0
    epoch_num = 100
    writer = SummaryWriter()
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = "cuda:1"
    model = model.to(device)
    loss = loss.to(device)
    # 训练
    for epoch in range(epoch_num):
        print("-"*10 + f"Start epoch {epoch } train" +"-"*10)
        model.train()
        for data in train_data_loader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            l = loss(outputs,targets)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            total_train_step += 1
            if total_train_step%100 == 0:
                print(f"step {total_train_step} loss {l.item()}")
                writer.add_scalar("train_loss",l.item(),total_train_step)
        # 测试
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_data_loader:
                imgs,targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                l = loss(outputs,targets)
                total_test_loss = total_test_loss + l.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy

        rate = total_accuracy / test_data_size
        total_test_step += 1
        writer.add_scalar("test_loss",total_test_loss,total_test_step)
        writer.add_scalar("test_accuracy",rate,total_test_step)
        print(f"test loss {total_test_loss}")
        print(f"test accuracy {rate}") # 测试正确率额
        print("-"*40)
        torch.save(model,f"data/cifar_epoch_{epoch}.pth")
        print(f"cifar_epoch_{epoch}.pth saved.")

    writer.close()
def test():
    from PIL import Image
    img_pth = "data/img.png"
    img = Image.open(img_pth)
    img = img.convert("RGB")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor()
    ])
    img = transform(img)
    img = img.reshape((1,3,32,32))
    img = img.to("cuda:1")
    print(img.shape)

    model_pth = "data/cifar_epoch_54.pth"
    model = torch.load(model_pth)
    # model = torch.load(model_pth,map_location=torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        print(model(img))

if __name__ == "__main__":
    # train()
    # train_gpu_1()
    test()