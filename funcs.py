import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from applymodel import applyindivmodel
from massapplymodel import massapplymodelfunc
from tkinter import filedialog
from tkinter import ttk
from scipy.signal import spectrogram
from sklearn.preprocessing import StandardScaler
import csv
import math
import threading
from PIL import Image # as img

script_directory = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(script_directory, r'Image')
output_folder_path = ''
#print(output_folder)
#print(script_directory)
mass_applymodel_path = os.path.join(script_directory, "massapplymodel.py")
output_folder_path = ''
output_csv_path = ''
os.makedirs(output_folder, exist_ok=True)

VALID_EXTENSIONS = ('.mp3', '.wav', '.flac')

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
        print(file_path)
        return file_path

def generate_spectrogram_threaded():
    def task():
        if file_path:
            spectrogram_save_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_spectrogram.png")
            print(f'\n\n{spectrogram_save_path}\n\n')
            mp3_to_spectrogram(file_path, spectrogram_save_path)
            applymodel_path = os.path.join(script_directory, "applymodel.py")
            if os.path.exists(applymodel_path):
                applyindivmodel(spectrogram_save_path)
                return spectrogram_save_path
            else:
                print("ERROR: applymodel.py not found.")
                pass
        else:
            print("ERROR Occured.")
            return "ERROR"
    thread = threading.Thread(target=task)
    thread.start()
    
def mp3_to_spectrogram_threaded(file_path, save_path=None):
    y, sr = librosa.load(file_path, sr=16000, duration=5.0)
    f, t, Zxx = spectrogram(
        y, fs=sr, window='hamming', nperseg=int(sr * 0.108), noverlap=int(sr * 0.01)
    )
    log_Zxx = np.log1p(np.abs(Zxx))
    scaler = StandardScaler()
    z_normalized_log_Zxx = scaler.fit_transform(log_Zxx.T).T
    max_db_value = 11.0
    z_normalized_log_Zxx = np.clip(z_normalized_log_Zxx, -np.inf, max_db_value)
    z_normalized_log_Zxx = (z_normalized_log_Zxx - z_normalized_log_Zxx.min()) / (z_normalized_log_Zxx.max() - z_normalized_log_Zxx.min())
    z_normalized_log_Zxx = (z_normalized_log_Zxx * 255).astype(np.uint8)
    image_array = z_normalized_log_Zxx.astype(np.uint8)
    
    image = Image.fromarray(image_array)
    image = image.convert('RGB')

    if save_path:
        image.save(save_path)
    else:
        print("ERROR Occured in mp3_to_spec")
        return

def generate_spectrogram():
    if file_path:
        spectrogram_save_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_spectrogram.png")
        mp3_to_spectrogram(file_path, spectrogram_save_path)
        applymodel_path = 0
        applymodel_path = os.path.join(script_directory, "applymodel.py")
        if os.path.exists(applymodel_path):
            applyindivmodel(spectrogram_save_path)
            return spectrogram_save_path
        else:
            print("ERROR: applymodel.py not found.")
            pass
    else:
        return

def mp3_to_spectrogram(file_path, save_path=None):
    y, sr = librosa.load(file_path, sr=16000, duration=5.0)
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

# Mass Predictions
def reset_prediction_display():
    """image_label.config(image='')
    guess_probability.config(text='')
    type_prediction.config(text='')
    method_prediction.config(text='')"""
    pass
    
def open_folder_dialog():
    global folder_path, foldername
    folder_path = filedialog.askdirectory(
        title="Select Folder"
    )
    if folder_path:
        foldername = os.path.basename(folder_path)
        return foldername, folder_path

def mass_generate_spectrogram(folder_path):
    if not folder_path: # Receives folder of audio files
        print("ERROR NO FOLDER SELECTED")
        return "ERROR"
    global output_folder_path
    output_folder_path = os.path.join(output_folder, 'Mass') # \Image\Mass for spectrogram folder
    if not os.path.exists(output_folder_path): # Create \Image\Mass if it doesn't exist
        
        print("CREATED FOLDER")
        os.makedirs(output_folder_path, exist_ok=True)
    for filename in os.listdir(folder_path): # For filenames in audio file folder
        file_path = os.path.join(folder_path, filename) # gets individual file paths
        if os.path.isfile(file_path) and filename.lower().endswith(VALID_EXTENSIONS): # if file is valid filetype
            if os.path.isfile(file_path): # if file_path exists
                try:
                    y, sr = librosa.load(file_path, sr=16000, duration=5.0)
                    f, t, Zxx = spectrogram(
                        y, fs=sr, window='hamming', nperseg=int(sr * 0.108), noverlap=int(sr * 0.01)
                    )
                    """log_Zxx = np.log1p(np.abs(Zxx))
                    scaler = StandardScaler()
                    z_normalized_log_Zxx = scaler.fit_transform(log_Zxx.T).T
                    max_db_value = 11.0
                    z_normalized_log_Zxx = np.clip(z_normalized_log_Zxx, -np.inf, max_db_value)
                    num_segments = int(np.ceil(y.shape[0] / (sr * 5.0)))
                    for i in range(num_segments):
                        start_time = i * 5.0
                        end_time = min((i + 1) * 5.0, y.shape[0] / sr)
                        start_index = int(start_time * sr)
                        end_index = int(end_time * sr)
                        plt.figure(figsize=(8, 6))
                        plt.imshow(z_normalized_log_Zxx[:, start_index:end_index], aspect='auto', origin='lower', cmap='viridis', vmax=max_db_value)
                        plt.axis('off')"""
                    log_Zxx = np.log1p(np.abs(Zxx))
                    scaler = StandardScaler()
                    z_normalized_log_Zxx = scaler.fit_transform(log_Zxx.T).T
                    max_db_value = 11.0
                    z_normalized_log_Zxx = np.clip(z_normalized_log_Zxx, -np.inf, max_db_value)
                    
                    plt.figure(figsize=(8, 6))
                    plt.imshow(z_normalized_log_Zxx, aspect='auto', origin='lower', cmap='viridis', vmax=max_db_value)
                    plt.axis('off')
                    segment_output_path = os.path.join(output_folder_path, f"{os.path.splitext(os.path.basename(file_path))[0]}_spectrogram.png")
                    plt.savefig(segment_output_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    """
                    y, sr = librosa.load(file_path, sr=16000, duration=5.0)
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
                    plt.axis('off')"""
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    massapplyoutputcsv = classify_and_save_predictions_to_csv(folder_path)
    return massapplyoutputcsv

def classify_and_save_predictions_to_csv(folder_path):
    if not os.path.exists(mass_applymodel_path):
        return
    #mass_applymodel_path = os.path.join(script_directory, "massapplymodel.py")
    foldername = os.path.basename(folder_path)
    massapplyoutputcsv = os.path.join(script_directory, f'{foldername}_results_output.csv')
    print(output_folder_path, massapplyoutputcsv)
    result = massapplymodelfunc(output_folder_path, massapplyoutputcsv)
    
    print("Type: ", str(type(result)))
    print(result)
    return massapplyoutputcsv
    #display_predictions_csv(output_csv_path, output_folder_path)
    #file_name.config(text=f"Prediction results have been printed to {output_csv_path}.")

def create_treeview(csv_window, mass_output_file_path):
    frame = tk.Frame(csv_window)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    tree = ttk.Treeview(frame, columns=('Filename', 'Type', 'Confidence Level'), show='headings')
    tree.heading('Filename', text='Filename')
    tree.heading('Type', text='Type')
    tree.heading('Confidence Level', text='Confidence Level')

    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    vsb.pack(side='right', fill='y')
    tree.configure(yscrollcommand=vsb.set)
    tree.configure(height=5)
    tree.pack(fill=tk.BOTH, expand=True)

    with open(mass_output_file_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None) 
        for row in csv_reader:
            if not any(row):
                continue
            float_values = [float(value) for value in row[2:6]]
            max_value = max(float_values) * 100
            #formatted_max_value = "{:.4f}".format(max_value)

            last_insert = row[:2]
            
            if last_insert[1] == '1':
                last_insert[1] = 'Voice Synthesized'
            elif last_insert[1] == '2':
                last_insert[1] = 'Voice Changed'
            elif last_insert[1] == '3':
                last_insert[1] = 'Voice Spliced'
            elif last_insert[1] == '0':
                last_insert[1] = 'Unmodified'
            e = math.e
            k = 10
            x123 = max_value * 0.01
            funnum = (1/(1+e**(-k*(x123-0.3))))*100
            funnum = "{:.4f}".format(funnum)
            last_insert.append(funnum)
            tree.insert('', 'end', values=last_insert)
    return tree

def create_table_solo(scroll_frame, csv_filepath):
    with open(csv_filepath, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)
        if not any(row for row in csv_reader if any(row)):
            return 0
        
    frame = tk.Frame(scroll_frame)
    frame.grid(row=0, column=0, padx=0, pady=(10, 10))
    
    style = ttk.Style(frame)
    style.theme_use("default")
    style.configure("Treeview",
                    background="#2a2d2e",
                    foreground="white",
                    rowheight=25,
                    fieldbackground="#343638",
                    bordercolor="#343638",
                    borderwidth=0)
    style.map('Treeview', background=[('selected', '#22559b')])
    style.configure("Treeview.Heading",
                    background="#565b5e",
                    foreground="white",
                    relief="flat")
    style.map("Treeview.Heading",
                background=[('active', '#3484F0')])
    style.configure("Vertical.TScrollbar",
                background="#343638", 
                troughcolor="#2a2d2e", 
                bordercolor="#2a2d2e",  
                arrowcolor="white",     
                relief="flat",          
                borderwidth=0)
    style.map("Vertical.TScrollbar",
            background=[('active', '#565b5e'), ('!disabled', '#343638')])

    tree = ttk.Treeview(frame, columns=('Filename', 'Type', 'Confidence'), show='headings')
    tree.heading('Filename', text='Filename')
    tree.heading('Type', text='Type')
    tree.heading('Confidence', text='Confidence')
    
    tree.column('Filename', width=200, anchor='w')
    tree.column('Type', width=100, anchor='center')
    tree.column('Confidence', width=100, anchor='center') 

    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    vsb.pack(side='right', fill='y')
    tree.configure(yscrollcommand=vsb.set)
    tree.configure(height=5)
    tree.pack(fill=tk.BOTH, expand=True)
    
    with open(csv_filepath, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None) 
        for row in csv_reader:
            if not any(row):
                continue
            float_values = [float(value) for value in row[2:6]]
            max_value = max(float_values) * 100
            filename = os.path.basename(row[0])
            last_insert = [filename, row[1]]
            #last_insert = row[:2]
            if last_insert[1] == '1':
                last_insert[1] = 'Voice Synthesized'
            elif last_insert[1] == '2':
                last_insert[1] = 'Voice Changed'
            elif last_insert[1] == '3':
                last_insert[1] = 'Voice Spliced'
            elif last_insert[1] == '0':
                last_insert[1] = 'Unmodified'
            e = math.e
            k = 10
            x123 = max_value * 0.01
            funnum = (1/(1+e**(-k*(x123-0.3))))*100
            funnum = "{:.2f}".format(funnum)
            last_insert.append(funnum)
            tree.insert('', 'end', values=last_insert)
            
    return tree

def create_table_mass(scroll_frame, csv_filepath):
    with open(csv_filepath, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)
        if not any(row for row in csv_reader if any(row)):
            return 0
        
    frame = tk.Frame(scroll_frame)
    frame.grid(row=0, column=0, padx=0, pady=(10, 10))
    
    style = ttk.Style(frame)
    style.theme_use("default")
    style.configure("Treeview",
                    background="#2a2d2e",
                    foreground="white",
                    rowheight=25,
                    fieldbackground="#343638",
                    bordercolor="#343638",
                    borderwidth=0)
    style.map('Treeview', background=[('selected', '#22559b')])
    style.configure("Treeview.Heading",
                    background="#565b5e",
                    foreground="white",
                    relief="flat")
    style.map("Treeview.Heading",
                background=[('active', '#3484F0')])
    style.configure("Vertical.TScrollbar",
                background="#343638", 
                troughcolor="#2a2d2e", 
                bordercolor="#2a2d2e",  
                arrowcolor="white",     
                relief="flat",          
                borderwidth=0)
    style.map("Vertical.TScrollbar",
            background=[('active', '#565b5e'), ('!disabled', '#343638')])

    tree = ttk.Treeview(frame, columns=('Filename', 'Type', 'Confidence'), show='headings')
    tree.heading('Filename', text='Filename')
    tree.heading('Type', text='Type')
    tree.heading('Confidence', text='Confidence')
    
    tree.column('Filename', width=200, anchor='w')
    tree.column('Type', width=100, anchor='center')
    tree.column('Confidence', width=100, anchor='center') 

    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    vsb.pack(side='right', fill='y')
    tree.configure(yscrollcommand=vsb.set)
    tree.configure(height=5)
    tree.pack(fill=tk.BOTH, expand=True)
    
    with open(csv_filepath, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None) 
        for row in csv_reader:
            if not any(row):
                continue
            float_values = [float(value) for value in row[2:6]]
            max_value = max(float_values) * 100
            filename = os.path.basename(row[0])
            last_insert = [filename, row[1]]
            #last_insert = row[:2]
            if last_insert[1] == '1':
                last_insert[1] = 'Voice Synthesized'
            elif last_insert[1] == '2':
                last_insert[1] = 'Voice Changed'
            elif last_insert[1] == '3':
                last_insert[1] = 'Voice Spliced'
            elif last_insert[1] == '0':
                last_insert[1] = 'Unmodified'
            e = math.e
            k = 10
            x123 = max_value * 0.01
            funnum = (1/(1+e**(-k*(x123-0.3))))*100
            funnum = "{:.2f}".format(funnum)
            last_insert.append(funnum)
            tree.insert('', 'end', values=last_insert)
            
    return tree

def display_predictions_csv(mass_output_file_path, output_folder_path):
    def on_window_close():
        combined_command(csv_window, output_folder_path)

    csv_window = tk.Toplevel()
    csv_window.title("Predictions")
    csv_window.geometry("1100x500")
    csv_window.resizable(False, False)
    csv_window.protocol("WM_DELETE_WINDOW", on_window_close)

    tree = create_treeview(csv_window, mass_output_file_path)

    return_button = tk.Button(csv_window, text="Close", command=lambda: combined_command(csv_window, output_folder_path))
    return_button.pack(side=tk.BOTTOM, anchor='w', padx=10, pady=10)
    #mass_history_button.config(state=tk.NORMAL)

def redisplay_predictions():
    def on_window_close():
        combined_command(csv_window, output_folder_path)
    csv_window = tk.Toplevel()
    csv_window.title("Predictions")
    csv_window.geometry("1100x500")
    csv_window.resizable(False, False)
    csv_window.protocol("WM_DELETE_WINDOW", on_window_close)

    tree = create_treeview(csv_window, output_csv_path)
    return_button = tk.Button(csv_window, text="Close", command=lambda: combined_command(csv_window, output_folder_path))
    return_button.pack(side=tk.BOTTOM, anchor='w', padx=10, pady=10)

def combined_command(csv_window, folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    csv_window.destroy()
    
def extract_txt_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    predicted_class = int(lines[0].split(":")[1].strip())
    class_probabilities = list(map(float, lines[1].split(":")[1].strip()[1:-1].split(',')))
    highest_prob = max(class_probabilities)
    return predicted_class, highest_prob, class_probabilities

