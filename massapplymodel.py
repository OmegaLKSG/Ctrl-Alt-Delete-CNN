import torch
from torchvision import transforms
from PIL import Image
from ThesisModeCNN import SimpleCNN
import os
import csv

script_directory = os.path.dirname(os.path.abspath(__file__))

def massapplymodelfunc(folder_path, output_file_path):
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
    #output_folder = os.path.join(script_directory, r'Image')
    #folder_path = os.path.join(output_folder, 'Mass')

    #output = io.StringIO(newline='')
    #csv_writer = csv.writer(output)

    header = ['Filename', 'Predicted Class'] + [f'Class {i} Probability' for i in range(4)]
    #csv_writer.writerow(header)

    with open(output_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        #_segment_0_spectrogram.png
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                new_filename = filename.replace("_segment_0_spectrogram.png", ".mp3")
                result = classify_image(file_path)
                row = [new_filename, result['predicted_class']] + result['class_probabilities']
                csv_writer.writerow(row)
                
    return output_file_path
