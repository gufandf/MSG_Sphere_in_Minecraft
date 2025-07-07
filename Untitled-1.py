import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from search import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 定义自定义数据集类
class TripleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)  # 输入是三元数组
        self.labels = torch.LongTensor(labels)  # 输出是整数
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 2. 定义简单的神经网络模型
class TripleToIntModel(nn.Module):
    def __init__(self):
        super(TripleToIntModel, self).__init__()
        self.fc1 = nn.Linear(3, 16)  # 输入层: 3个特征
        self.fc2 = nn.Linear(16, 8)   # 隐藏层
        self.fc3 = nn.Linear(8, 1)    # 输出层: 1个值(可以舍入为整数)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()  # 去掉多余的维度

# 3. 准备训练数据
# 假设我们有以下训练数据:
# 输入: 三元数组
train_data = []

for r in range(255):
    for g in range(255):
        for b in range(255):
            train_data.append((r,g,b))

# 输出: 对应的整数标签
# 这里只是一个示例，你需要根据你的实际需求定义这些标签
train_labels = []
for color in train_data:
    train_labels.append(get_block(color))

# 4. 创建数据加载器
dataset = TripleDataset(train_data, train_labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 5. 初始化模型、损失函数和优化器
model = TripleToIntModel().to(device)
criterion = nn.MSELoss()  # 均方误差损失，因为我们要预测数值
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, labels in dataloader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 7. 测试模型
test_data = torch.FloatTensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
with torch.no_grad():
    predictions = model(test_data)
    print("\nTest Predictions:")
    for i, pred in enumerate(predictions):
        print(f"Input: {test_data[i].tolist()} -> Predicted: {round(pred.item())} (Raw: {pred.item():.2f})")