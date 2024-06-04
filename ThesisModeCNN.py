# RUN THIS CODE TO START TRAINING AND TESTING A NEW MODEL FILE
# Alter num_params() to change the size of the model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
import os
import pandas as pd
from PIL import Image

torch.manual_seed(678567456742562546)

num_epochs = 30 # change this to change number of epochs

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()

        self.relu = nn.ReLU()

        num_params = 32 #16 8

        params1d2 = (int)(num_params/2)
        params1d4 = (int)(num_params/4)

        self.bn2d = nn.BatchNorm2d(num_params) # 32
        self.bn2d_d2 = nn.BatchNorm2d(params1d2) # 16
        self.bn2d_d4 = nn.BatchNorm2d(params1d4) # 8

        self.bn1d_8 = nn.BatchNorm1d(16)
        self.bn1d_64 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(p=0.2)

        self.line1 = nn.Linear(num_params*25, 64)

        self.line2 = nn.Linear(64, 4) # The number of desired output classes is the second parameter here
        self.flatten = nn.Flatten()

        #5x5 Conv Layer
        self.conv_layer5x5_3_8 = nn.Conv2d(3, params1d4, kernel_size=5, stride=2, padding=2) # 3 > 8

        #3x3 Conv Layer
        self.conv_layer3x3_8_16 = nn.Conv2d(params1d4, params1d2, kernel_size=3, stride=1, padding=0) # 8 > 16
        self.conv_layer3x3_16_16 = nn.Conv2d(params1d2, params1d2, kernel_size=3, stride=1, padding=0) # 16 > 16
        self.conv_layer3x3_16_32 = nn.Conv2d(params1d2, num_params, kernel_size=3, stride=1, padding=0) # 16 > 32
        self.conv_layer3x3_32_32 = nn.Conv2d(num_params, num_params, kernel_size=3, stride=1, padding=0) # 32 > 32

        #1x1 Conv Layer
        self.conv_layer1x1_8_16 = nn.Conv2d(params1d4, params1d2, kernel_size=1, stride=1, padding=2) # 8 > 16
        self.conv_layer1x1_16_16 = nn.Conv2d(params1d2, params1d2, kernel_size=1, stride=1, padding=2) # 16 > 16
        self.conv_layer1x1_16_32 = nn.Conv2d(params1d2, num_params, kernel_size=1, stride=1, padding=2) # 16 > 32
        self.conv_layer1x1_32_32 = nn.Conv2d(num_params, num_params, kernel_size=1, stride=1, padding=2) # 32 > 32

    def forward(self, x):
        #Input Processing Block
        x = self.pool(self.bn2d_d4(self.relu(self.conv_layer5x5_3_8(x)))) # 3 > 8

        #3x3 Conv cycle 1
        con = self.bn2d_d2(self.relu(self.conv_layer1x1_8_16(x))) # 8 > 16
        con = self.bn2d_d2(self.relu(self.conv_layer3x3_16_16(con))) # 16 > 16

        res = self.relu(self.conv_layer1x1_8_16(x)) # 8 > 16
        res = self.bn2d_d2(res)

        res = F.interpolate(res, size=(con.size(2), con.size(3)), mode='nearest')
        x = con + res
        x = self.pool(self.relu(x))

        #3x3 Conv Cycle 2
        con = self.bn2d(self.relu(self.conv_layer1x1_16_32(x))) # 16 > 32
        con = self.bn2d(self.relu(self.conv_layer3x3_32_32(con))) # 32 > 32

        res = self.relu(self.conv_layer1x1_16_32(x))
        res = self.bn2d(res)

        res = F.interpolate(res, size=(con.size(2), con.size(3)), mode='nearest')
        x = con + res
        x = self.pool(self.relu(x))

        #3x3 Conv Cycles 3-4
        for i in range(2):
          con = self.bn2d(self.relu(self.conv_layer1x1_32_32(x)))
          con = self.bn2d(self.relu(self.conv_layer3x3_32_32(con)))

          res = self.relu(self.conv_layer1x1_32_32(x))
          res = self.bn2d(res)

          res = F.interpolate(res, size=(con.size(2), con.size(3)), mode='nearest')
          x = con + res
          x = self.pool(self.relu(x))

        x = self.flatten(x)

        x = self.drop(self.line1(x))
        x = self.relu(self.bn1d_64(x))
        x = self.drop(self.line2(x))

        x = F.softmax(x, dim=1)

        return x

class CustomDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')

        class_label = self.df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, class_label

transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
  ])

model = SimpleCNN(num_classes=4)