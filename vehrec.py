import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

img_path = r"D:\Downloads\240416114616-Vehicle.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, (240, 240))
img = img / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

class_names = ['bike', 'bus', 'car', 'truck']

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.fc1 = nn.Linear(200704, 128)  
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 200704) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = ConvNet()
img_tensor = torch.from_numpy(img).float()
predictions = model(img_tensor)
_, class_index = torch.max(predictions, 1)
print("The image is a", class_names[class_index])