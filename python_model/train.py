import torch
import threading
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

stop_flag = False
image_data = np.zeros((28, 28))

def show_update(img_data):
	global image_data
	image_data = img_data
	
# 设备定义
if torch.cuda.is_available():
	device = torch.device('cuda:0')
	print('Using GPU: ' + torch.cuda.get_device_name(0))
else:
	device = torch.device('cpu')
	print('Using CPU')

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

valset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False)
# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

import os
checkpoint_path = 'net.pth'
if os.path.exists(checkpoint_path):
	checkpoint = torch.load(checkpoint_path, weights_only=True)

# 定义损失函数和优化器
net = Net().to(device)
if os.path.exists(checkpoint_path):	
    net.load_state_dict(checkpoint)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

def train():
	# 训练模型
	for epoch in range(100):  # 训练10个epoch
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			if stop_flag:
				break
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)

			optimizer.zero_grad()

			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			if i % 100 == 99:  # 每100个batch打印一次loss
				print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.8f}')
				running_loss = 0.0
				show_update(inputs[0].cpu().numpy().reshape(28, 28))

	print('Finished Training')
	# 保存模型
	torch.save(net.state_dict(), checkpoint_path)

import cv2
import time
import torch.nn.functional as F
def test():
	# 每隔一秒显示一张图像并标注预测结果
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			if stop_flag:
				break
			images, labels = data
			images, labels = images.to(device), labels.to(device)
			outputs = net(images)

			_, predicted = torch.max(outputs.data, 1)

			img = images.cpu().numpy().reshape(28, 28)
			cv2.putText(img, str(predicted.item()), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
			show_update(img)
			time.sleep(0.5)

			if predicted != labels:
				print(f'error: labels: {labels.item()}, predickted: {predicted.item()}')
				break

def val():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.cpu().size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.4f}%')
