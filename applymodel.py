import torch
from torchvision import transforms
from PIL import Image
from ThesisModeCNN import SimpleCNN
from ThesisModeCNN import SimpleCNN
import os
import sys

script_directory = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
def applyindivmodel(image_path):
    model = SimpleCNN(num_classes=4)
    pth_path = os.path.join(script_directory, '4class_model New Algo New Dataset 4.pth')
    model.load_state_dict(torch.load(pth_path))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    input_image = transform(image).unsqueeze(0)

    with torch.no_grad():
        model_output = model(input_image)
        probabilities = torch.nn.functional.softmax(model_output, dim=1)
        _, predicted_class = torch.max(probabilities, 1)

    print(f'Predicted Class: {predicted_class.item()}')
    print(f'Class Probabilities: {probabilities.squeeze().tolist()}')

    prediction_results_path = os.path.join(script_directory, "prediction_results.txt")

    with open(prediction_results_path, 'w') as f:
        print(f'Predicted Class: {predicted_class.item()}', file=f)
        print(f'Class Probabilities: {probabilities.squeeze().tolist()}', file=f)

