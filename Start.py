import tkinter
import os
import tkinter.messagebox
import customtkinter
from PIL import Image as img  # Import for handling images
#import threading
import subprocess

from funcs import *
from audioplayer import *

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

VALID_EXTENSIONS = ('.mp3', '.wav', '.flac')

script_directory = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(script_directory, r'Image')
csv_directory = os.path.join(script_directory, r'CSV')
#results_txt_filepath = f'{csv_directory}/prediction_results.txt'
results_csv_filepath = f'{csv_directory}/solo_history.csv'
output_folder_path = ''
output_csv_path = ''

os.makedirs(output_folder, exist_ok=True)
os.makedirs(csv_directory, exist_ok=True)

class detailed_window(customtkinter.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("1100x600")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0), weight=1)
        self.scrollable_frame = customtkinter.CTkScrollableFrame(self, label_text="Results")
        self.scrollable_frame.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        if "solo_history.csv" in os.listdir(csv_directory):
            csv_filepath = os.path.join(csv_directory, "solo_history.csv")
            label = customtkinter.CTkLabel(self.scrollable_frame, text="solo_history.csv")
            label.pack(pady=(10, 0), anchor="center")  # Center the label
            table_frame = customtkinter.CTkFrame(self.scrollable_frame)
            table_frame.pack(pady=(0, 20), fill="x", expand=True)  # Center the table frame
            create_detailed_table(table_frame, csv_filepath)

        for filename in os.listdir(csv_directory):
            if filename.endswith("_results_output.csv"):
                csv_filepath = os.path.join(csv_directory, filename)
                label = customtkinter.CTkLabel(self.scrollable_frame, text=filename)
                label.pack(pady=(10, 0), anchor="center")  # Center the label
                table_frame = customtkinter.CTkFrame(self.scrollable_frame)
                table_frame.pack(pady=(0, 20), fill="x", expand=True)  # Center the table frame
                create_detailed_table(table_frame, csv_filepath)
                # table_frame.bind("<MouseWheel>", lambda event, frame=table_frame: self.on_mouse_wheel(event, frame))

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # setting values
        self.solo_filename = None
        self.mass_foldername = None
        self.mass_folder_path = None
        self.detailed_window = None
        
        #self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.icon_folder_path = f'{script_directory}\icons'
        self.dark_icon_path = f'{script_directory}\icons\mimicallogo_white.ico'
        self.light_icon_path = f'{script_directory}\icons\mimicallogo.ico'
        # configure window
        self.iconbitmap(self.dark_icon_path)
        self.title("Mimical - Detect Fake Audio")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Mimical", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_1_event)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_2_event)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_3_event)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))

        # create main entry and button
        self.footer = customtkinter.CTkFrame(self)
        self.footer.grid(row=3, column=1, columnspan=2, padx=(20, 10), pady=(10, 0), sticky="nsew")
        self.loadingBar = customtkinter.CTkProgressBar(self.footer)
        self.loadingBar.grid(row=0, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.loadingStatusLabel = customtkinter.CTkLabel(self.footer)
        self.loadingStatusLabel.grid(row=0, column=1, padx=(20, 10), pady=(10, 10), sticky="ew")

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, width=250)
        self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        
        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=2, columnspan=1, padx=(20, 10), pady=(20, 0), sticky="nsew")
        self.tabview.add("Control")
        self.tabview.add("Details")
        self.tabview.add("Mass Details")
        
        self.tabview.tab("Control").grid_columnconfigure(0, weight=1, minsize=150)
        self.tabview.tab("Control").grid_columnconfigure(1, weight=1, minsize=150)
        
        self.tabview.tab("Details").grid_columnconfigure(0, weight=1, minsize=300)
        self.tabview.tab("Mass Details").grid_columnconfigure(0, weight=1, minsize=300)
        #self.tabview.tab("Mass Details").configure(state='disabled')

        self.selection_label_tab_1 = customtkinter.CTkLabel(self.tabview.tab("Control"), text="Select a file or folder to start audio checking")
        self.selection_label_tab_1.grid(row=0, column=0, padx=0, pady=20, sticky="ew", columnspan=2)
        self.perform_prediction_button = customtkinter.CTkButton(self.tabview.tab("Control"), text="Perform Predictions", command=self.perform_prediction, state="disabled", fg_color="#cc7a00", hover_color="#ab4d00")
        self.perform_prediction_button.grid(row=2, column=0, padx=20, pady=(10, 10), sticky="nsew")
        self.perform_massprediction_button = customtkinter.CTkButton(self.tabview.tab("Control"), text="Perform Mass Predictions", command=self.perform_mass_prediction, state="normal", fg_color="#cc7a00", hover_color="#ab4d00")
        self.detailed_history_button = customtkinter.CTkButton(self.tabview.tab("Control"), text="View Detailed Descriptions", command=self.open_detailed_table, state="normal")
        self.detailed_history_button.grid(row=2, column=1, padx=20, pady=(10, 10), sticky="nsew")
        self.result_directory_button = customtkinter.CTkButton(self.tabview.tab("Control"), text="Open CSV File Directory", command=self.open_csv_directory, state="normal")
        self.result_directory_button.grid(row=3, column=0, padx=20, pady=(10, 10), sticky="nsew")
        self.image_directory_button = customtkinter.CTkButton(self.tabview.tab("Control"), text="Open Image Directory", command=self.open_image_directory, state="normal")
        self.image_directory_button.grid(row=3, column=1, padx=20, pady=(10, 10), sticky="nsew")
        
        self.label_tab_2 = customtkinter.CTkLabel(self.tabview.tab("Details"), text="Perform a Solo Audio Check to see results here.")
        self.label_tab_2.grid(row=0, column=0, padx=20, pady=(20,0))
        self.filename_tab_2 = customtkinter.CTkLabel(self.tabview.tab("Details"), text="re.")
        self.verdict_tab_2 = customtkinter.CTkLabel(self.tabview.tab("Details"), text="here.")
        self.confidence_tab_2 = customtkinter.CTkLabel(self.tabview.tab("Details"), text="here.")
        self.confidence1_tab_2 = customtkinter.CTkLabel(self.tabview.tab("Details"), text="here.")
        self.confidence2_tab_2 = customtkinter.CTkLabel(self.tabview.tab("Details"), text="here.")
        self.confidence3_tab_2 = customtkinter.CTkLabel(self.tabview.tab("Details"), text="here.")
        #self.filename_tab_2.grid(row=1, column=0, padx=20, pady=0)
        #self.verdict_tab_2.grid(row=2, column=0, padx=20, pady=0)
        #self.confidence_tab_2.grid(row=3, column=0, padx=20, pady=0)
        
        self.label_tab_3 = customtkinter.CTkLabel(self.tabview.tab("Mass Details"), text="Perform a Mass Audio Checker to see summary here.")
        self.label_tab_3.grid(row=0, column=0, padx=20, pady=20)

        # create slider and progressbar frame
        self.slider_progressbar_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.slider_progressbar_frame.grid(row=1, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.slider_progressbar_frame.grid_columnconfigure(0, weight=1)
        self.slider_progressbar_frame.grid_rowconfigure(4, weight=1)
        self.audio_label = customtkinter.CTkLabel(self.slider_progressbar_frame, text="No Audio Selected")
        self.audio_label.grid(row=0, column=0, padx=(0, 0), pady=(10, 5), sticky="ew")

        self.audio_slider = customtkinter.CTkSlider(self.slider_progressbar_frame, from_=0, to=1)
        self.audio_slider.set(0)
        self.audio_slider.grid(row=2, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        
        self.play_button_img = os.path.join(self.icon_folder_path, "play_button.png") 
        self.pause_button_img = os.path.join(self.icon_folder_path, "pause_button.png") 
        self.play_image = customtkinter.CTkImage(img.open(self.play_button_img), size=(20, 20))
        self.pause_image = customtkinter.CTkImage(img.open(self.pause_button_img), size=(20, 20))
        
        self.play_button = customtkinter.CTkButton(self.slider_progressbar_frame, text="Play", command=lambda: play_music(self.audio_slider))
        self.play_button.grid(row=3, column=0, padx=5, sticky="ew")
        self.play_button.configure(state='disabled', image=self.play_image)
        
        self.volume_slider = customtkinter.CTkSlider(self.slider_progressbar_frame, orientation="vertical", from_=0.0, to=1.0, height=100, command=set_volume)
        self.volume_slider.grid(row=2, column=1, rowspan=1, padx=(10, 10), pady=(10, 10), sticky="ns")
        self.volume_label = customtkinter.CTkLabel(self.slider_progressbar_frame, text="Volume")
        self.volume_label.grid(row=3, column=1, padx=(10, 10), pady=(10, 10), sticky="ew")
        
        self.result_tabview = customtkinter.CTkTabview(self, width=250)
        self.result_tabview.grid(row=1, column=2, columnspan=1, padx=(20, 10), pady=(20, 0), sticky="nsew")
        self.result_tabview.add("Solo Results")
        self.result_tabview.add("Mass Results")
        
        self.result_tabview.tab("Solo Results").grid_columnconfigure(0, weight=1, minsize=300)
        self.result_tabview.tab("Mass Results").grid_columnconfigure(0, weight=1, minsize=300)
        
        self.text_label_tab1 = customtkinter.CTkLabel(self.result_tabview.tab("Solo Results"))
        self.text_label_tab1.grid(row=0, column=0, padx=(20, 10), pady=(0, 0), sticky="ew")
        self.text_label_tab1.configure(text='No History Yet.')
        
        self.text_label_tab2 = customtkinter.CTkLabel(self.result_tabview.tab("Mass Results"))
        self.text_label_tab2.grid(row=0, column=0, padx=(20, 10), pady=(0, 0), sticky="ew")
        self.text_label_tab2.configure(text='No History Yet.')
        
        self.sidebar_button_3_event()

        # set default values
        self.textbox.insert("0.0", "Mimical\n\n"+"""Press "Solo Audio Checking" or "Mass Audio Checking" to start
You can see previous history in the bottom right.
Click "View Documentation to see full instructions.\n\n""")
        self.textbox.configure(state='disabled')
        self.sidebar_button_1.configure(state="normal", text="Sole Audio Checking")
        self.sidebar_button_2.configure(state="normal", text="Mass Audio Checking")
        self.sidebar_button_3.configure(state="normal", text="View Previous History")
        self.appearance_mode_optionemenu.set("Dark")
    
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)
        current_mode = customtkinter.get_appearance_mode()
        if current_mode == "Dark":
            self.iconbitmap(self.dark_icon_path)
        else:
            self.iconbitmap(self.light_icon_path)

    def perform_prediction(self):
        spectrogram_file_path, audiofilename, predictedclass, probabilities = generate_spectrogram()
        self.spectrogram_image_path = os.path.join(os.path.dirname(__file__), spectrogram_file_path)
        self.photo = customtkinter.CTkImage(img.open(self.spectrogram_image_path), size=(400, 320))
        self.image_label = customtkinter.CTkLabel(self, text='', image=self.photo)
        self.image_label.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.perform_prediction_button.configure(state='disabled')
        
        self.label_tab_2.configure(text='')
        self.label_tab_2.forget()
        self.filename_tab_2.grid(row=0, column=0, padx=20, pady=(20,0), sticky="nw")
        self.verdict_tab_2.grid(row=1, column=0, padx=20, pady=0, sticky="nw")
        self.confidence_tab_2.grid(row=2, column=0, padx=20, pady=0, sticky="nw")
        self.confidence1_tab_2.grid(row=3, column=0, padx=20, pady=0, sticky="nw")
        self.confidence2_tab_2.grid(row=4, column=0, padx=20, pady=0, sticky="nw")
        self.confidence3_tab_2.grid(row=5, column=0, padx=20, pady=0, sticky="nw")
        self.filename_tab_2.configure(text=f'Filename: {audiofilename}')
        funnum, other_prob, verdict_prob_text = format_confidence(probabilities, predictedclass)
        a, b, c = other_prob
        self.verdict_tab_2.configure(text=f'Verdict: {verdict_prob_text}')
        self.confidence_tab_2.configure(text=f'Confidence: {funnum}%')
        other_prob_str = ', '.join(f'{value}%' for value in other_prob)
        conf1tab2text = f'Other probabilities: {other_prob_str}'
        self.confidence1_tab_2.configure(text=conf1tab2text)
        #self.confidence1_tab_2.configure(text=f'Other probabilities: {max(other_prob)}%')
        #self.confidence2_tab_2.configure(text=f'Other prob: {b}%')
        #self.confidence3_tab_2.configure(text=f'Other prob: {c}%')
        self.confidence2_tab_2.configure(text=f'')
        self.confidence3_tab_2.configure(text=f'')
        
        self.sidebar_button_3_event()
        self.tabview.set("Details")
        
    def perform_mass_prediction(self):
        mass_csv_output_path = mass_generate_spectrogram(self.mass_folder_path)
        if create_table_mass(self.result_tabview.tab("Mass Results"), mass_csv_output_path):
            self.text_label_tab2.forget()
            create_mass_summary_graph(self.tabview.tab("Mass Details"), mass_csv_output_path)
        self.tabview.set("Mass Details")
        self.result_tabview.set("Mass Results")
        self.perform_massprediction_button.configure(state='disabled')

    def delete_perform_button(self):
        self.perform_prediction_button.grid_remove()
        self.perform_massprediction_button.grid_remove()
    
    def sidebar_button_1_event(self):
        self.solo_filename = open_file_dialog()
        self.delete_perform_button()
        self.perform_prediction_button.grid(row=2, column=0, padx=20, pady=(10, 10))
        if not self.solo_filename:
            self.perform_prediction_button.configure(state='disabled')
            self.selection_label_tab_1.configure(text=f'No File Selected')
            self.audio_label.configure(text=f'No File Selected')
            return
        else:
            self.perform_prediction_button.configure(state='normal')
            self.selection_label_tab_1.configure(text=f'Selected file: {os.path.basename(self.solo_filename)}')
            song_length = load_song(self.solo_filename)
            self.audio_slider = customtkinter.CTkSlider(self.slider_progressbar_frame, from_=0, to=song_length, command=lambda x: slide(x, self.audio_slider))
            self.audio_slider.grid(row=2, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
            self.audio_slider.set(0)
            self.audio_label.configure(text=f'Playing: {os.path.basename(self.solo_filename)}')
            self.play_button.configure(state='normal')
            self.tabview.set("Control")
            
    def sidebar_button_2_event(self):
        print("Mass Audio Checking click")
        self.mass_foldername, self.mass_folder_path = open_folder_dialog()
        self.delete_perform_button()
        self.perform_massprediction_button.grid(row=2, column=0, padx=20, pady=(10, 10), sticky="nsew")
        if not self.mass_foldername:
            self.perform_massprediction_button.configure(state='disabled')
            self.selection_label_tab_1.configure(text=f'No Folder Selected')
            return
        else:
            self.perform_massprediction_button.configure(state='normal')
        self.selection_label_tab_1.configure(text=f'Selected folder: {(self.mass_foldername)}')
        
    def sidebar_button_3_event(self):
        header = ['FilePath', 'PredictedClass'] + [f'Class_{i}_Prob' for i in range(4)]
        file_exists = os.path.exists(results_csv_filepath)
        with open(results_csv_filepath, 'a', newline='', encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            if not file_exists:
                csv_writer.writerow(header)
        if create_table_solo(self.result_tabview.tab("Solo Results"), results_csv_filepath):
            self.text_label_tab1.forget()
        
    def open_detailed_table(self):
        if not hasattr(self, 'detailed_table') or self.detailed_table is None or not self.detailed_table.winfo_exists():
            header = ['FilePath', 'PredictedClass'] + [f'Class_{i}_Prob' for i in range(4)]
            file_exists = os.path.exists(results_csv_filepath)
            with open(results_csv_filepath, 'a', newline='', encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                if not file_exists:
                    csv_writer.writerow(header)
            self.detailed_table = detailed_window(self)
            self.detailed_table.focus()
        else:
            self.detailed_table.focus()
    
    def open_csv_directory(self):
        os.system(f'start {os.path.realpath(csv_directory)}')
        
    def open_image_directory(self):
        os.system(f'start {os.path.realpath(output_folder)}')
    
    def open_directory(self, path):
        subprocess.Popen(f'explorer /select,"{path}\a"')
        
if __name__ == "__main__":
    app = App()
    app.mainloop()