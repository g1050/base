import os
import numpy as np
import pandas as pd
import torch
from numpy import nan as NaN
def main():
    os.makedirs(os.path.join('.','data'),exist_ok=True)
    datafile = os.path.join('.','data','house_tiny.csv')
    with open(datafile,'w') as f:
        f.write('NumRooms,Alley,Price\n')  # 列名
        f.write('NA,Pave,127500\n')  # 第1行的值
        f.write('2,NA,106000\n')  # 第2行的值
        f.write('4,NA,178100\n')  # 第3行的值
        f.write('NA,NA,140000\n')  # 第4行的值
    data = pd.read_csv(datafile)
    print(f"original data \n{data}")
    inputs,outputs = data.iloc[:,0:2],data.iloc[:,2] # 开区间
    inputs.iloc[:,0] = inputs.fillna(inputs.iloc[:,0].mean()) # 均值填充Nan
    print(inputs.shape,type(inputs))
    print(f"\ninputs\n{inputs}")
    print(f"\noutputs\n{outputs}")
    # 独特编码
    inputs = pd.get_dummies(inputs,dummy_na=True)
    print(f"one hot {inputs}")
    # 转torch.tensor
    inputs_tensor = torch.tensor(inputs.to_numpy().astype(np.float32))
    print(inputs_tensor.device)
    inputs_tensor = inputs_tensor.to('cuda:0')
    print(inputs_tensor.device)

if __name__ == "__main__":
    main()