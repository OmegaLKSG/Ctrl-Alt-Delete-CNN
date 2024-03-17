import torch
from torchvision import transforms
from PIL import Image
from ThesisCNN import SimpleCNN
import os
import sys
from torchvision import transforms

script_directory = os.path.dirname(os.path.abspath(__file__))

model = SimpleCNN(num_classes=2)
pth_path = os.path.join(script_directory, '4class_model.pth')
model.load_state_dict(torch.load(pth_path))
model.eval()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

#image_path = os.path.join(script_directory, 'biden-to-linus_clip_2_segment_1.png')
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

input_image = transform(image).unsqueeze(0)

with torch.no_grad():
    model_output = model(input_image)
    probabilities = torch.nn.functional.softmax(model_output, dim=1)
    _, predicted_class = torch.max(probabilities, 1)

print(f'Predicted Class: {predicted_class.item()}')
print(f'Class Probabilities: {probabilities.squeeze().tolist()}')

output_file_path = os.path.join(script_directory, 'prediction_results.txt')
with open(output_file_path, 'w') as output_file:
    output_file.write(f'Predicted Class: {predicted_class.item()}\n')
    output_file.write(f'Class Probabilities: {probabilities.squeeze().tolist()}\n')

print(f'Prediction results written to: {output_file_path}')