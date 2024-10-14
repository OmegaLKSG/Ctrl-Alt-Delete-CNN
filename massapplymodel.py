import torch
from torchvision import transforms
from PIL import Image
from ThesisModeCNN import SimpleCNN
import os
import csv

script_directory = os.path.dirname(os.path.abspath(__file__))

def massapplymodelfunc(spectrogram_folder_path, output_file_path, audiofilenames):
    model = SimpleCNN(num_classes=4)
    pth_path = os.path.join(script_directory, 'Mimical_Model.pth')
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
    header = ['Filename', 'Predicted Class'] + [f'Class {i} Probability' for i in range(4)]
    
    with open(output_file_path, 'w', newline='', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        new_filename = ''
        print(spectrogram_folder_path)
        for filename in os.listdir(spectrogram_folder_path):
            stripped_filename = filename.replace("_spectrogram.png", "")
            """if stripped_filename in audiofilenames:
                print(stripped_filename)"""
            new_filename = stripped_filename
            file_path = os.path.join(spectrogram_folder_path, filename)
            if os.path.isfile(file_path):
                #new_filename = filename.replace("_spectrogram.png", "")
                result = classify_image(file_path)
                row = [new_filename, result['predicted_class']] + result['class_probabilities']
                csv_writer.writerow(row)
                
    return output_file_path
