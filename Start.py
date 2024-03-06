import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from scipy.signal import spectrogram
from sklearn.preprocessing import StandardScaler


def open_file_dialog():
    global file_path, filename
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[
            ("Audio files", "*.mp3;*.wav;*.flac"),
            ("MP3 files", "*.mp3"),
            ("WAV files", "*.wav"),
            ("FLAC files", "*.flac")
        ]
    )
    if file_path:
        transform_button.place(x=190, y=40)
        filename = os.path.basename(file_path)
        file_name.config(text=f"Selected Audio: {filename}")

def generate_spectrogram():
    if file_path:
        spectrogram_save_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_spectrogram.png")
        mp3_to_spectrogram(file_path, spectrogram_save_path)
        open_image(spectrogram_save_path)
        transform_button.place(x=465, y=95)
        os.system(f"python applymodel.py \"{spectrogram_save_path}\"")
        display_prediction()
    else:
        file_name.config(text="No audio file selected.")

# Transforms audio file into a normalized log-spectrogram image
def mp3_to_spectrogram(file_path, save_path=None):
    y, sr = librosa.load(file_path, sr=None)

    f, t, Zxx = spectrogram(
        y, fs=sr, window='hamming', nperseg=int(sr * 0.108), noverlap=int(sr * 0.01)
    )

    log_Zxx = np.log1p(np.abs(Zxx))

    scaler = StandardScaler()
    z_normalized_log_Zxx = scaler.fit_transform(log_Zxx.T).T

    max_db_value = 11.0
    z_normalized_log_Zxx = np.clip(z_normalized_log_Zxx, -np.inf, max_db_value)

    plt.figure(figsize=(8, 6))
    plt.imshow(z_normalized_log_Zxx, aspect='auto', origin='lower', cmap='viridis', vmax=max_db_value)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def open_image(image_path):
    original_image = Image.open(image_path)
    resized_image = original_image.resize((250, 250))

    tk_image = ImageTk.PhotoImage(resized_image)
    image_label.config(image=tk_image)
    image_label.image = tk_image
    root.geometry("750x340")

script_directory = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(script_directory, r'Image')
os.makedirs(output_folder, exist_ok=True)

def display_prediction():
    with open(output_file_path, 'r') as output_file:
        lines = output_file.readlines()

    predicted_class_line = lines[0].strip()
    class_probabilities_line = lines[1].strip()

    predicted_class = int(predicted_class_line.split(":")[1].strip())
    class_probabilities_str = class_probabilities_line.split(":")[1].strip()
    class_probabilities = [round(float(value) * 100, 2) for value in class_probabilities_str.strip("[]").split(",")]

    if predicted_class == 0:
        type_prediction.config(text=f'The audio file is LEGITIMATE')
    elif predicted_class == 1:
        type_prediction.config(text=f'The audio file is MODIFIED')
    
    guess_probability.config(text=f'Confidence Level: {class_probabilities[predicted_class]}%')

root = tk.Tk()
root.title("Audio DeepFake Detector")
root.geometry("750x150")
root.resizable(False, False)

audio_button = tk.Button(root, text="Open Audio File", command=open_file_dialog, width=20, height=1)
history_button = tk.Button(root, text="View Checking History", width=20, height=1)
transform_button = tk.Button(root, text="Perform Prediction", command=generate_spectrogram, width=20, height=1)
api_button = tk.Button(root, text="API Settings", width=20, height=1)
documentation_button = tk.Button(root, text="View Documentation", width=20, height=1)

file_name = tk.Label(root, text="")
image_label = tk.Label(root)
guess_probability = tk.Label(root, text="")
type_prediction = tk.Label(root, text="")

audio_button.grid(row=0, column=0, padx=10, pady=5)
history_button.grid(row=1, column=0, padx=10, pady=5)
api_button.grid(row=2, column=0, padx=10, pady=5)
documentation_button.grid(row=3, column=0, padx=10, pady=5)
#transform_button.grid(row=2, column=0, padx=10, pady=5)
#transform_button.place(x=465, y=95)
transform_button.grid_remove()
history_button.configure(state='disabled')
api_button.configure(state='disabled')
documentation_button.configure(state='disabled')

file_name.place(x=170, y=7)
image_label.place(x=180, y=45)
guess_probability.place(x=475, y=65)

type_prediction.place(x=475, y=45)

root.columnconfigure(0, minsize=0)
root.columnconfigure(1, minsize=225)
root.columnconfigure(2, minsize=0)

output_file_path = "prediction_results.txt"  # Replace with the actual path

root.mainloop()