import torch
from torchvision import transforms
from PIL import Image
from ThesisModeCNN import SimpleCNN
from ThesisModeCNN import SimpleCNN
import os
import sys
import csv

script_directory = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
def applyindivmodel(image_path):
    model = SimpleCNN(num_classes=4)
    pth_path = os.path.join(script_directory, 'Mimical_Model.pth')
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
        
    prediction_results_csv_path = f'{script_directory}/solo_history.csv'        
    header = ['FilePath', 'PredictedClass'] + [f'Class_{i}_Prob' for i in range(4)]
    file_exists = os.path.exists(prediction_results_csv_path)
    with open(prediction_results_csv_path, 'a', newline='', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(header)
        new_filename = image_path.replace("_segment_0_spectrogram.png", ".mp3")
        row = [new_filename, predicted_class.item(), *probabilities.squeeze().tolist()]
        csv_writer.writerow(row)
    
    return