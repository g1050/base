import torch
print(torch.__version__)
print(torch.cuda.is_available())
num = torch.cuda.device_count()
for i in range(num):
    print(torch.cuda.get_device_name(i))
print(f"current device {torch.cuda.current_device()}")
x = torch.rand(5,3)
y = torch.rand(5,3)
if torch.cuda.is_available():
    x.to('cuda:0')
    y.to('cuda:0')
    print(x+y)
