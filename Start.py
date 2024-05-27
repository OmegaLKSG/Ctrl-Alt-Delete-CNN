import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from scipy.signal import spectrogram
from sklearn.preprocessing import StandardScaler
import sys

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
        reset_prediction_display()
        transform_button.grid(row=1, column=1, padx=10, pady=5)
        filename = os.path.basename(file_path)
        file_name.config(text=f"Selected Audio: {filename}")

def reset_prediction_display():
    image_label.config(image='')
    guess_probability.config(text='')
    type_prediction.config(text='')
    method_prediction.config(text='')

def generate_spectrogram():
    if file_path:
        spectrogram_save_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_spectrogram.png")
        mp3_to_spectrogram(file_path, spectrogram_save_path)
        open_image(spectrogram_save_path)
        transform_button.grid(row=4, column=0, padx=10, pady=5)

        applymodel_path = os.path.join(script_directory, "applymodel.py")
        if os.path.exists(applymodel_path):
            os.system(f"python \"{applymodel_path}\" \"{spectrogram_save_path}\"")
            display_prediction()
        else:
            file_name.config(text="applymodel.py not found.")
            
        display_prediction()
        transform_button.grid_remove()
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
print(f"Script directory: {script_directory}")

# Add the script's directory to the system path
sys.path.append(script_directory)
print(f"System path: {sys.path}")

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
    elif predicted_class != 0:
        type_prediction.config(text=f'The audio file is MODIFIED')
    
    match predicted_class:
        case 0:
            method_prediction.config(text=f'Modification Type: Unmodified')
        case 1:
            method_prediction.config(text=f'Modification Type: Voice Synthesis')
        case 2:
            method_prediction.config(text=f'Modification Type: Voice Changer')
        case 3:
            method_prediction.config(text=f'Modification Type: Voice Splicing')
        
    guess_probability.config(text=f'Confidence Level: {class_probabilities[predicted_class]}%')

    with open(history_file_path, "a") as history_file:
        history_file.write(f"Filename: {filename}\n")
        history_file.write(f"Type: {'LEGITIMATE' if predicted_class == 0 else 'MODIFIED'}\n")
        history_file.write(f"Confidence Level: {class_probabilities[predicted_class]}%\n\n")

def display_history():
    history_window = tk.Toplevel(root)
    history_window.title("Checking History")
    history_window.geometry("600x400")
    history_window.resizable(False, False)

    frame = tk.Frame(history_window)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    history_text = tk.Text(frame, wrap=tk.WORD, font=("Arial", 10))
    history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scroll_bar = tk.Scrollbar(frame, command=history_text.yview)
    scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)
    history_text.config(yscrollcommand=scroll_bar.set)

    try:
        with open(history_file_path, "r") as history_file:
            history_data = history_file.read()

        if history_data:
            history_text.insert(tk.END, history_data)
        else:
            history_text.insert(tk.END, "No history found.")
    except FileNotFoundError:
        history_text.insert(tk.END, "No history found.")

    history_text.config(state=tk.DISABLED)

def display_documentation():
    documentation_window = tk.Toplevel(root)
    documentation_window.title("Documentation")
    documentation_window.geometry("600x400")
    documentation_window.resizable(False, False)

    frame = tk.Frame(documentation_window)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    documentation_text = tk.Text(frame, wrap=tk.WORD, font=("Arial", 10))
    documentation_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scroll_bar = tk.Scrollbar(frame, command=documentation_text.yview)
    scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)
    documentation_text.config(yscrollcommand=scroll_bar.set)

    # Example documentation text
    documentation = """
    Audio DeepFake Detector Documentation

    This application is designed to detect audio deepfakes. It processes audio files, 
    generates spectrograms, and uses a pre-trained model to classify the audio as legitimate 
    or modified. The modifications can include various techniques such as voice synthesis, 
    voice changers, voice modulation, and voice splicing.

    Usage:
    1. Click "Open Audio File" to select an audio file.
    2. Click "Perform Prediction" to generate the spectrogram and classify the audio.
    3. View the prediction results displayed on the main window.
    4. Click "View Checking History" to see past predictions.

    Note:
    The application currently supports MP3 and WAV audio formats.
    """

    documentation_text.insert(tk.END, documentation)
    documentation_text.config(state=tk.DISABLED)

    return_button = tk.Button(documentation_window, text="Return to Startup", command=documentation_window.destroy)
    return_button.pack(side=tk.BOTTOM, anchor='w', padx=10, pady=10)


root = tk.Tk()
root.title("Audio DeepFake Detector")
root.geometry("750x250")
root.resizable(False, False)

history_label = tk.Label(root, text="", wraplength=600, justify="left")

audio_button = tk.Button(root, text="Open Audio File", command=open_file_dialog, width=20, height=1)
history_button = tk.Button(root, text="View Checking History", command=display_history, width=20, height=1)
transform_button = tk.Button(root, text="Perform Prediction", command=generate_spectrogram, width=20, height=1)
api_button = tk.Button(root, text="API Settings", width=20, height=1)
documentation_button = tk.Button(root, text="View Documentation", command=display_documentation, width=20, height=1)

file_name = tk.Label(root, text="")
image_label = tk.Label(root)
guess_probability = tk.Label(root, text="")
type_prediction = tk.Label(root, text="")
method_prediction = tk.Label(root, text="")

audio_button.grid(row=0, column=0, padx=10, pady=5)
history_button.grid(row=1, column=0, padx=10, pady=5)
api_button.grid(row=2, column=0, padx=10, pady=5)
documentation_button.grid(row=3, column=0, padx=10, pady=5)
#transform_button.grid(row=2, column=0, padx=10, pady=5)
#transform_button.place(x=465, y=95)
transform_button.grid_remove()

api_button.configure(state='disabled')

file_name.place(x=170, y=7)
image_label.place(x=180, y=45)
guess_probability.place(x=475, y=65)
method_prediction.place(x=475, y=85)

type_prediction.place(x=475, y=45)

root.columnconfigure(0, minsize=0)
root.columnconfigure(1, minsize=225)
root.columnconfigure(2, minsize=0)

output_file_path = os.path.join(script_directory, "prediction_results.txt")

# Define the directory for storing history files
history_directory = os.path.join(script_directory, 'History')
os.makedirs(history_directory, exist_ok=True)

# Define the path for the history file
history_file_path = os.path.join(history_directory, 'history.txt')

root.mainloop()
