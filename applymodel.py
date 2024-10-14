import torch
from torchvision import transforms
from PIL import Image
from ThesisModeCNN import SimpleCNN
from ThesisModeCNN import SimpleCNN
import os
import sys
import csv

script_directory = os.path.dirname(os.path.abspath(__file__))
csv_directory = os.path.join(script_directory, r'CSV')
prediction_results_path = f'{csv_directory}/prediction_results.txt'
prediction_results_csv_path = f'{csv_directory}/solo_history.csv'     

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
def applyindivmodel(image_path, file_path):
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

    with open(prediction_results_path, 'w') as f:
        print(f'Predicted Class: {predicted_class.item()}', file=f)
        print(f'Class Probabilities: {probabilities.squeeze().tolist()}', file=f)
        
    header = ['FilePath', 'PredictedClass'] + [f'Class_{i}_Prob' for i in range(4)]
    file_exists = os.path.exists(prediction_results_csv_path)
    with open(prediction_results_csv_path, 'a', newline='', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(header)
        #audio_filename = image_path
        new_filename = os.path.basename(file_path)
        #new_filename = image_path.replace("_spectrogram.png", ".mp3")
        row = [new_filename, predicted_class.item(), *probabilities.squeeze().tolist()]
        csv_writer.writerow(row)
    
    audiofilename = os.path.basename(new_filename)
    returned_values = (audiofilename, predicted_class.item(), probabilities.squeeze().tolist())
    return returned_values