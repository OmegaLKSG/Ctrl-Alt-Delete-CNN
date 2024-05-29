import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from scipy.signal import spectrogram
from sklearn.preprocessing import StandardScaler
import sys
import subprocess
import csv
import io

if __name__ == "__main__":
    root = tk.Tk()
    
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
        
    def open_folder_dialog():
        global folder_path, foldername
        folder_path = filedialog.askdirectory(
            title="Select Folder"
        )
        if folder_path:
            reset_prediction_display()
            mass_transform_button.grid(row=1, column=1, padx=10, pady=5)
            foldername = os.path.basename(folder_path)
            file_name.config(text=f"Selected Folder: {foldername}")

    def mass_generate_spectrogram():    
        if not folder_path:
            file_name.config(text="No folder selected.")
            return
        output_folder_path = os.path.join(output_folder, 'Mass')
        os.makedirs(output_folder_path, exist_ok=True)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                try:
                    y, sr = librosa.load(file_path, sr=16000, duration=5.0)

                    # Compute the short-time Fourier transform (STFT)
                    f, t, Zxx = spectrogram(
                        y, fs=sr, window='hamming', nperseg=int(sr * 0.108), noverlap=int(sr * 0.01)
                    )

                    log_Zxx = np.log1p(np.abs(Zxx))

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
                        plt.axis('off')
                        
                        segment_output_path = os.path.join(output_folder_path, f"{os.path.splitext(filename)[0]}_segment_{i}_spectrogram.png")
                        
                        plt.savefig(segment_output_path, bbox_inches='tight', pad_inches=0)
                        plt.close()
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        classify_and_save_predictions_to_csv()

    def classify_and_save_predictions_to_csv():

        mass_applymodel_path = os.path.join(script_directory, "massapplymodel.py")

        if not os.path.exists(mass_applymodel_path):
            file_name.config(text="massapplymodel.py not found.")
            return

        result = subprocess.run(["python", mass_applymodel_path, output_folder_path], capture_output=True, text=True)

        try:
            csv_output = io.StringIO(result.stdout)
            csv_reader = csv.reader(csv_output)
            header = next(csv_reader, None)
            if header is not None:
                with open(output_csv_path, 'w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(header)
                    for row in csv_reader:
                        csv_writer.writerow(row)
            else:
                with open(output_csv_path, 'w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(header)
                    for row in csv_reader:
                        csv_writer.writerow(row)
        except Exception as e:
            print(f"Error reading output file: {e}")
            import traceback
            traceback.print_exc()
        display_predictions_csv(output_csv_path, output_folder_path)
        file_name.config(text=f"Prediction results have been printed to {output_csv_path}.")

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
        tree.pack(fill=tk.BOTH, expand=True)

        with open(mass_output_file_path, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None) 
            for row in csv_reader:
                if not any(row):
                    continue
                float_values = [float(value) for value in row[2:6]]
                max_value = max(float_values) * 100
                formatted_max_value = "{:.4f}".format(max_value)
                last_insert = row[:2]
                last_insert.append(formatted_max_value)
                tree.insert('', 'end', values=last_insert)

        return tree

    def display_predictions_csv(mass_output_file_path, output_folder_path):
        def on_window_close():
            combined_command(csv_window, output_folder_path)

        csv_window = tk.Toplevel(root)
        csv_window.title("Predictions")
        csv_window.geometry("1100x500")
        csv_window.resizable(False, False)
        csv_window.protocol("WM_DELETE_WINDOW", on_window_close)

        tree = create_treeview(csv_window, mass_output_file_path)

        return_button = tk.Button(csv_window, text="Close", command=lambda: combined_command(csv_window, output_folder_path))
        return_button.pack(side=tk.BOTTOM, anchor='w', padx=10, pady=10)
        mass_history_button.config(state=tk.NORMAL)

    def redisplay_predictions():
        def on_window_close():
            combined_command(csv_window, output_folder_path)
        csv_window = tk.Toplevel(root)
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
    output_csv_path = os.path.join(script_directory, 'mass_prediction_results.csv')
    print(f"Script directory: {script_directory}")

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
        
        del_path = os.path.join(output_folder, filename[:-4] + "_spectrogram.png")

        try:
            os.remove(del_path)
        except Exception as e:
            print(f"Error deleting {del_path}: {e}")
        
        with open(history_file_path, "a") as history_file:
            history_file.write(f"Filename: {filename}\n")
            history_file.write(f"Type: {'LEGITIMATE' if predicted_class == 0 else 'MODIFIED'}\n")
            history_file.write(f"Confidence Level: {class_probabilities[predicted_class]}%\n\n")

    def display_history():
        def clear_history():
            try:
                with open(history_file_path, "r+") as history_file:
                    history_file.truncate(0)
                history_text.config(state=tk.NORMAL)
                history_text.delete(1.0, tk.END)
                history_text.insert(tk.END, "History cleared.")
                history_text.config(state=tk.DISABLED)
                
            except FileNotFoundError:
                history_text.config(state=tk.NORMAL)
                history_text.delete(1.0, tk.END)
                history_text.insert(tk.END, "No history found.")
                history_text.config(state=tk.DISABLED)

        history_window = tk.Toplevel(root)
        history_window.title("Checking History")
        history_window.geometry("620x520")
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

        clear_button = tk.Button(history_window, text="Clear History", command=clear_history)
        clear_button.pack()

    def display_documentation():
        documentation_window = tk.Toplevel(root)
        documentation_window.title("Documentation")
        documentation_window.geometry("620x400")
        documentation_window.resizable(False, False)

        frame = tk.Frame(documentation_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        documentation_text = tk.Text(frame, wrap=tk.WORD, font=("Arial", 10))
        documentation_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll_bar = tk.Scrollbar(frame, command=documentation_text.yview)
        scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)
        documentation_text.config(yscrollcommand=scroll_bar.set)

        documentation = """
        Audio DeepFake Detector Documentation

        This application is designed to detect audio deepfakes. It processes audio files, transforms 
        them into spectrograms, and uses a Convolutional Neural Network called ResEfficient 
        CNN to classify the audio as legitimate or modified. The modifications can include various 
        techniques such as voice synthesis, voice changers, and voice splicing.
        
        Instructions:
        
        For Individual File Checking:
        1. Click "Open Audio File" to select an audio file.
        2. Click "Perform Prediction" to generate the spectrogram and classify the audio.
        3. View the prediction results displayed on the main window.
        4. Click "View Checking History" to see past predictions.

        To Check Multiple Files at Once:
        1. Click "API Settings" to open the API menu.
        2. Click "Mass Prediction" to select a folder with audio files to process.
        3. Do note that depending on the number of files, processing may take a while.
        4. View the prediction results displayed on the new window.
        5. The results will be output to a csv file in the installation folder.
        
        The system is capable of detecting four types of voice audio indicated by number:
            0 Corresponds to real unmodified voice audio.
            1 Corresponds to synthetic or text-to-speech voice audio.
            2 Corresponds to audio modified using a voice changer.
            3 Corresponds to voice audio that has been cut and spliced together.
            
        This system works by transforming audio files into spectrogram images, and then using those
        spectrogram images to classify the audio and determine whether the voice audio is modified,
        and what kind of modification that voice audio has undergone.
        
        To test the capabilities of the system, several files are provided in the "Test Audio" folder
        for the user's convenience. These audio files consist of real voices, synthetic voices, voice 
        changed voices, and voice spliced voices.
        """

        documentation_text.insert(tk.END, documentation)
        documentation_text.config(state=tk.DISABLED)

        return_button = tk.Button(documentation_window, text="Return to Startup", command=documentation_window.destroy)
        return_button.pack(side=tk.BOTTOM, anchor='w', padx=10, pady=10)

    def api_settings():
        for widget in root.winfo_children():
            widget.grid_remove()
        return_button = tk.Button(root, text="Return", command=show_initial_menu, width=20, height=1)
        folder_button.grid(row=0, column=0, padx=10, pady=5)
        mass_history_button.grid(row=1, column=0, padx=10, pady=5)
        mass_history_button.config(state=tk.DISABLED)
        file_name.config(text=" ")
        return_button.place(x=10, y=215)

    def show_initial_menu():
        for widget in root.winfo_children():
            widget.place_forget()
        root.geometry("750x250")
        audio_button.grid(row=0, column=0, padx=10, pady=5)
        history_button.grid(row=1, column=0, padx=10, pady=5)
        api_button.grid(row=2, column=0, padx=10, pady=5)
        documentation_button.grid(row=3, column=0, padx=10, pady=5)
        transform_button.grid_remove()
        folder_button.grid_remove()
        mass_history_button.grid_remove()
        mass_transform_button.grid_remove()
        file_name.config(text=" ")
        file_name.place(x=170, y=7)
        image_label.place(x=180, y=45)
        guess_probability.place(x=475, y=65)
        method_prediction.place(x=475, y=85)
        type_prediction.place(x=475, y=45)

    root.title("Audio DeepFake Detector")
    root.geometry("750x250")
    root.resizable(False, False)

    history_label = tk.Label(root, text="", wraplength=600, justify="left")

    audio_button = tk.Button(root, text="Open Audio File", command=open_file_dialog, width=20, height=1)
    folder_button = tk.Button(root, text="Mass Prediction", command=open_folder_dialog, width=20, height=1)
    history_button = tk.Button(root, text="View Checking History", command=display_history, width=20, height=1)
    mass_history_button = tk.Button(root, text="Review Mass Prediction", command=redisplay_predictions, width=20, height=1)
    transform_button = tk.Button(root, text="Perform Prediction", command=generate_spectrogram, width=20, height=1)
    mass_transform_button = tk.Button(root, text="Perform Mass Prediction", command=mass_generate_spectrogram, width=20, height=1)
    api_button = tk.Button(root, text="API Settings", command=api_settings, width=20, height=1)
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
    transform_button.grid_remove()

    file_name.place(x=170, y=7)
    image_label.place(x=180, y=45)
    guess_probability.place(x=475, y=65)
    method_prediction.place(x=475, y=85)
    type_prediction.place(x=475, y=45)

    root.columnconfigure(0, minsize=0)
    root.columnconfigure(1, minsize=225)
    root.columnconfigure(2, minsize=0)

    output_file_path = os.path.join(script_directory, "prediction_results.txt")
    output_folder_path = os.path.join(output_folder, 'Mass')

    history_directory = os.path.join(script_directory, 'History')
    os.makedirs(history_directory, exist_ok=True)

    history_file_path = os.path.join(history_directory, 'history.txt')

    root.mainloop()
