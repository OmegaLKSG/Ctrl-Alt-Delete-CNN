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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

script_directory = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(script_directory, r'Image')
csv_directory = os.path.join(script_directory, r'CSV')
output_folder_path = ''
#print(output_folder)
#print(script_directory)
mass_applymodel_path = os.path.join(script_directory, "massapplymodel.py")
output_folder_path = ''
output_csv_path = ''

os.makedirs(output_folder, exist_ok=True)
os.makedirs(csv_directory, exist_ok=True)

VALID_EXTENSIONS = ('.mp3', '.wav', '.flac')

def open_file_dialog():
    global file_path#, filename
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
            audiofilename, predictedclass, probabilities = applyindivmodel(spectrogram_save_path, file_path)
            #audiofilename, predictedclass, probabilities = model_result
            #print(f'try: {spectrogram_save_path}, {audiofilename}, {predictedclass}, {probabilities}')
            return spectrogram_save_path, audiofilename, predictedclass, probabilities 
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
        os.makedirs(output_folder_path, exist_ok=True)
        
    audiofilenames = []
    for filename in os.listdir(folder_path): # For filenames in audio file folder
        file_path = os.path.join(folder_path, filename) # gets individual file paths
        if os.path.isfile(file_path) and filename.lower().endswith(VALID_EXTENSIONS): # if file is valid filetype
            audiofilenames.append(filename)
            #print(audiofilenames)
            if os.path.isfile(file_path): # if file_path exists
                try:
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
                    segment_output_path = os.path.join(output_folder_path, f"{os.path.splitext(os.path.basename(file_path))[0]}_spectrogram.png")
                    plt.savefig(segment_output_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    massapplyoutputcsv = classify_and_save_predictions_to_csv(folder_path, audiofilenames)
    return massapplyoutputcsv

def classify_and_save_predictions_to_csv(folder_path, audiofilenames):
    if not os.path.exists(mass_applymodel_path):
        return
    #mass_applymodel_path = os.path.join(script_directory, "massapplymodel.py")
    foldername = os.path.basename(folder_path)
    massapplyoutputcsv = os.path.join(csv_directory, f'{foldername}_results_output.csv')
    print(output_folder_path, massapplyoutputcsv)
    result = massapplymodelfunc(output_folder_path, massapplyoutputcsv, audiofilenames)
    
    #print("Type: ", str(type(result)))
    print(result)
    return massapplyoutputcsv
    #display_predictions_csv(output_csv_path, output_folder_path)
    #file_name.config(text=f"Prediction results have been printed to {output_csv_path}.")
    
def calc_confidence(probability):
    e = math.e
    k = 10
    x123 = probability
    funnum = (1/(1+e**(-k*(x123-0.3))))*100
    funnum = "{:.2f}".format(funnum)
    return funnum
    
def format_confidence(probabilities, predictedclass):
    #print(f'pred: {predictedclass}')
    verdict_prob = max(probabilities)
    other_prob = [prob for prob in probabilities if prob != verdict_prob]
    verdict_prob_text = ''
    if predictedclass == 1:
        verdict_prob_text = 'Voice Synthesized'
    elif predictedclass == 2:
        verdict_prob_text = 'Voice Changed'
    elif predictedclass == 3:
        verdict_prob_text = 'Voice Spliced'
    elif predictedclass == 0:
        verdict_prob_text = 'Unmodified'
    #print(verdict_prob_text)

    funnum = float(calc_confidence(verdict_prob))
    fun_otherprob = []
    
    for prob in other_prob:
        fun_otherprob.append((float(calc_confidence(prob)))/3)
    #print(fun_otherprob)
    #probabilities = [0.5, 0.2, 0.3, 0.4]
    prob_dict = {}
    for i, prob in enumerate(probabilities):
        prob_dict[i] = prob
        #print(prob_dict)
    fun_otherprob = [round(num, 2) for num in fun_otherprob]
    return funnum, fun_otherprob, verdict_prob_text

def create_table_solo(scroll_frame, csv_filepath):
    with open(csv_filepath, mode='r', encoding="utf8") as csv_file:
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
    
    with open(csv_filepath, mode='r', encoding="utf8") as csv_file:
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
    with open(csv_filepath, mode='r', encoding="utf8") as csv_file:
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
    
    with open(csv_filepath, mode='r', encoding="utf8") as csv_file:
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

def create_detailed_table(scroll_frame, csv_filepath):
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

    tree = ttk.Treeview(frame, columns=('Filepath', 'Type', 'Confidence', 'Probability 1', 'Probability 2', 'Probability 3', 'Probability 4'), show='headings')
    tree.heading('Filepath', text='Filename')
    tree.heading('Type', text='Type')
    tree.heading('Confidence', text='Confidence')
    tree.heading('Probability 1', text='Probability 1')
    tree.heading('Probability 2', text='Probability 2')
    tree.heading('Probability 3', text='Probability 3')
    tree.heading('Probability 4', text='Probability 4')
    
    tree.column('Filepath', anchor='w')
    tree.column('Type', anchor='center')
    tree.column('Confidence', anchor='center') 
    tree.column('Probability 1', anchor='center') 
    tree.column('Probability 2', anchor='center') 
    tree.column('Probability 3', anchor='center') 
    tree.column('Probability 4', anchor='center') 
    
    tree.column('Filepath', width=320, stretch=True)
    tree.column('Type', width=120, stretch=True)
    tree.column('Confidence', width=120, stretch=True)
    tree.column('Probability 1', width=110, stretch=True)
    tree.column('Probability 2', width=110, stretch=True)
    tree.column('Probability 3', width=110, stretch=True)
    tree.column('Probability 4', width=110, stretch=True)

    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    vsb.pack(side='right', fill='y')
    tree.configure(yscrollcommand=vsb.set)
    #tree.configure(height=5)
    tree.pack(fill=tk.BOTH, expand=True)
    
    with open(csv_filepath, mode='r', encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None) 
        if not any(row for row in csv_reader if any(row)):
            return 0
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
            #print(row[2:6])
            for i in range(2, 6):
                last_insert.append(row[i])
            tree.insert('', 'end', values=last_insert)
    return tree

def create_mass_summary_graph(frame, csv_filepath):
    class_tally = {
        'Unmodified': 0,
        'Voice Synthesis': 0,
        'Voice Changer': 0,
        'Voice Spliced': 0
    }
    with open(csv_filepath, mode='r', encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)
        for row in csv_reader:
            if not any(row):
                continue
            float_values = [float(value) for value in row[2:6]]
            max_index = float_values.index(max(float_values))
            class_names = ['Unmodified', 'Voice Synthesis', 'Voice Changer', 'Voice Spliced']
            class_name = class_names[max_index]
            class_tally[class_name] += 1
    create_pie_chart(frame, class_tally)
    return
            
def resize_pie_chart(event, fig, canvas):
    width = event.width / 600
    height = event.height / 600
    fig.set_size_inches(width, height)
    canvas.draw()

def create_pie_chart(frame, class_tally):
    labels = list(class_tally.keys())
    sizes = list(class_tally.values())
    colors = ['#005eff', '#002d7a', '#000b61', '#3b007a']
    explode = (0, 0, 0, 0)
    fig, ax = plt.subplots(figsize=(2.9, 2), facecolor='#333333')
    ax.set_facecolor('#333333')
    wedges, texts, autotexts = ax.pie(sizes,
                                      explode=explode,
                                      labels=labels,
                                      colors=colors,
                                      autopct='%1.1f%%',
                                      shadow=True,
                                      startangle=140,
                                      textprops={'color': 'white'})
    for autotext in autotexts:
        autotext.set_color('white')
    ax.axis('equal')
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
    canvas.get_tk_widget().update_idletasks()
    canvas.draw() 
    frame.bind("<Configure>", lambda event: resize_pie_chart(event, fig, canvas))

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

