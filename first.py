import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
#加载MNIST数据集
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

#创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


#定义全连接神经网络
class AFullNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784,128)

        self.fc2 = nn.Linear(128,256)

        self.fc3 = nn.Linear(256,10)


    def forward(self, x):
        out = self.fc1(x)
        out = nn.ReLU()(out)
        out = self.fc2(out)
        out = nn.ReLU()(out)
        out = self.fc3(out)
        return out


model = AFullNet()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


#训练代码
for epoch in range(10):
    for i,(images,labels) in enumerate(train_loader):
        images = images.reshape(-1,784)

        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{10}], Loss: {loss.item():.4f}')

#评估
correct = 0
total = 0
predicted_labels = []
true_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.numpy())
        true_labels.extend(labels.numpy())

predicted_labels = np.array(predicted_labels)
true_labels = np.array(true_labels)

TP = np.sum((predicted_labels == true_labels) & (predicted_labels == 1))
TN = np.sum((predicted_labels == true_labels) & (predicted_labels == 0))
FP = np.sum((predicted_labels != true_labels) & (predicted_labels == 1))
FN = np.sum((predicted_labels != true_labels) & (predicted_labels == 0))

print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')

accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * (precision * recall) / (precision + recall)
print(f'Accuracy: {accuracy:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {F1:.2f}')
