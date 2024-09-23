import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

# 定义教师和学生模型
teacher_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 10)

student_model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(128 * 7 * 7, 10)
)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer_teacher = optim.SGD(teacher_model.parameters(), lr=0.01, momentum=0.9)
optimizer_student = optim.Adam(student_model.parameters(), lr=0.001)

# 数据集转换
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 蒸馏过程
for epoch in range(10):
    running_loss_teacher = 0.0
    running_loss_student = 0.0
    
    for inputs, labels in trainloader:
        # 教师模型前向传播
        outputs_teacher = teacher_model(inputs)
        loss_teacher = criterion(outputs_teacher, labels)
        
        # 学生模型前向传播
        outputs_student = student_model(inputs)
        distillation_loss = 0.1 * torch.sum((outputs_teacher - outputs_student) ** 2)
        loss_student = criterion(outputs_student, labels) + distillation_loss
        
        # 合并损失后再反向传播
        total_loss = loss_teacher + loss_student
        
        optimizer_teacher.zero_grad()
        optimizer_student.zero_grad()
        
        total_loss.backward()
        
        optimizer_teacher.step()
        optimizer_student.step()
        
        running_loss_teacher += loss_teacher.item()
        running_loss_student += loss_student.item()
    
    print(f'Epoch {epoch+1}/10 \t Loss Teacher: {running_loss_teacher / len(trainloader)} \t Loss Student: {running_loss_student / len(trainloader)}')
