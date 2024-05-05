import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()

        self.relu = nn.ReLU()
        self.bn_1 = nn.BatchNorm2d(8)
        self.bn_2 = nn.BatchNorm2d(12)
        self.bn_3 = nn.BatchNorm2d(16)

        self.bn1d_1 = nn.BatchNorm1d(12)
        self.bn1d = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(p=0.2)

        self.line1 = nn.Linear(200, 64)
        #self.line1 = nn.Linear(72, 64)
        #self.line1 = nn.Linear(32, 64)
        self.line2 = nn.Linear(64, 3) # The number of desired output classes is the second parameter here
        self.flatten = nn.Flatten()

        #5x5 Conv Layer
        self.conv_layer5x5 = nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2)

        #3x3 Conv Layers
        self.conv_layer3x3_1 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=0)
        self.conv_layer3x3_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        self.conv_layer3x3_3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0)

        #1x1 Conv Layers
        self.conv_layer1x1_1 = nn.Conv2d(8, 12, kernel_size=1, stride=1, padding=2)
        self.conv_layer1x1_2 = nn.Conv2d(12, 16, kernel_size=1, stride=1, padding=2)
        self.conv_layer1x1_3 = nn.Conv2d(16, 12, kernel_size=1, stride=1, padding=2)
        self.conv_layer1x1_4 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=2)
        self.conv_layer1x1_5 = nn.Conv2d(12, 8, kernel_size=1, stride=1, padding=2)

    def forward(self, x):
        #Input Processing Block
        x = self.pool(self.bn_1(self.relu(self.conv_layer5x5(x)))) # in 3 out 8

        #Cycle 1
        # Conv Block
        con = self.bn_2(self.relu(self.conv_layer1x1_1(x))) # in 8 out 12
        con = self.bn_2(self.relu(self.conv_layer3x3_1(con))) # in 12 out 12

        # Res Block
        res = self.relu(self.conv_layer1x1_1(x)) # in 8 out 12
        res = self.bn_2(res)

        res = F.interpolate(res, size=(con.size(2), con.size(3)), mode='nearest')
        x = con + res
        x = self.pool(self.relu(x))

        #Cycle 2
        con = self.bn_3(self.relu(self.conv_layer1x1_2(x))) # in 12 out 16
        con = self.bn_3(self.relu(self.conv_layer3x3_2(con))) # in 16 out 16

        res = self.bn_3(self.relu(self.conv_layer1x1_2(x))) # in 12 out 16

        res = F.interpolate(res, size=(con.size(2), con.size(3)), mode='nearest')
        x = con + res
        x = self.pool(self.relu(x))

        #Cycle 3
        con = self.bn_2(self.relu(self.conv_layer1x1_3(x))) # in 16 out 12
        con = self.bn_2(self.relu(self.conv_layer3x3_1(con))) # in 12 out 12

        res = self.bn_2(self.relu(self.conv_layer1x1_3(x))) # in 16 out 12

        res = F.interpolate(res, size=(con.size(2), con.size(3)), mode='nearest')
        x = con + res
        x = self.pool(self.relu(x))

        #Cycle 4
        con = self.bn_1(self.relu(self.conv_layer1x1_5(x))) # in 12 out 8
        con = self.bn_1(self.relu(self.conv_layer3x3_3(con))) # in 8 out 8

        res = self.bn_1(self.relu(self.conv_layer1x1_5(x))) # in 12 out 8

        res = F.interpolate(res, size=(con.size(2), con.size(3)), mode='nearest')
        x = con + res
        x = self.pool(self.relu(x))
        
        x = self.flatten(x)
        
        x = self.drop(self.line1(x))
        x = self.relu(self.bn1d(x))
        x = self.drop(self.line2(x))

        x = F.softmax(x, dim=1)

        return x

model = SimpleCNN(num_classes=4)

# Load the trained parameters from the checkpoint file
checkpoint = torch.load('4class_model.pth')
model.load_state_dict(checkpoint)

# Set the model to evaluation mode
model.eval()

# Count the number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params)

# Define the transformation for your input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load and preprocess your input image
image = Image.open('C:\sss\Code\Python\THESIS\FinalThesis\Image\Recording0005_spectrogram.png').convert('RGB')
image = transform(image)
image = image.unsqueeze(0)  # Add batch dimension since the model expects a batch of images

# Perform inference using the loaded model
with torch.no_grad():
    output = model(image)

# Post-process the output if necessary
# For example, you might want to get the predicted class label
_, predicted_class = torch.max(output, 1)
print("Predicted class:", predicted_class.item())
