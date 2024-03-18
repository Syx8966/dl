import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载 Fashion-MNIST 数据集
train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

s = 128
n = 28 * 28
o = 10

class MyRNN(nn.Module):
    def __init__(self):
        super(MyRNN, self).__init__()
        self.s = s
        self.U = nn.Linear(n + s, s)
        self.V = nn.Linear(s, o)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        # 将输入展平成二维张量
        input = input.view(input.size(0), -1)
        combined = torch.cat((input, hidden), 1)
        hidden = self.U(combined)
        output = self.V(hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.s)

model = MyRNN()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(5):
    for i, (images, labels) in enumerate(train_loader):
        hidden = model.initHidden(images.size(0))

        optimizer.zero_grad()
        loss = 0
        for j in range(images.size(1)):
            output, hidden = model(images[:, j], hidden)
            loss += nn.CrossEntropyLoss()(output, labels)

        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{5}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 评估
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []
    for images, labels in test_loader:
        hidden = model.initHidden(images.size(0))
        loss = 0

        for j in range(images.size(1)):
            output, hidden = model(images[:, j], hidden)
            loss += nn.CrossEntropyLoss()(output, labels)

        _, predicted = torch.max(output, 1)
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
