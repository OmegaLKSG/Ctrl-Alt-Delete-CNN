import torch
from torchvision import transforms
from PIL import Image
from ThesisModeCNN import SimpleCNN
import os
import sys
import csv
import io

script_directory = os.path.dirname(os.path.abspath(__file__))

model = SimpleCNN(num_classes=4)
pth_path = os.path.join(script_directory, '4class_model New Algo New Dataset 4.pth')
model.load_state_dict(torch.load(pth_path))
model.eval()

def classify_image(image_path):
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

    return {
        'filename': os.path.basename(image_path),
        'predicted_class': predicted_class.item(),
        'class_probabilities': probabilities.squeeze().tolist()
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]

        output = io.StringIO()
        csv_writer = csv.writer(output)

        header = ['filename', 'predicted_class'] + [f'class_{i}_probability' for i in range(4)]
        csv_writer.writerow(header)
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            result = classify_image(file_path)
            prediction = result['predicted_class']
            confidence = max(result['class_probabilities'])
            
            row = [result['filename'],prediction] + confidence
            csv_writer.writerow(row)

        print(output.getvalue())
        output.close()
