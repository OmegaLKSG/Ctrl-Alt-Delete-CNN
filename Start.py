import tkinter
import os
import tkinter.messagebox
import customtkinter
from PIL import Image as img  # Import for handling images
import threading

from funcs import *
from audioplayer import *

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

VALID_EXTENSIONS = ('.mp3', '.wav', '.flac')

script_directory = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(script_directory, r'Image')
results_txt_filepath = f'{script_directory}/prediction_results.txt'
results_csv_filepath = f'{script_directory}/solo_history.csv'
output_folder_path = ''
output_csv_path = ''

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # setting values
        self.solo_filename = None
        self.mass_foldername = None
        self.mass_folder_path = None
        
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
        """
        self.image_path = os.path.join(os.path.dirname(__file__), "spectrogram.png")  # Adjust the path if necessary
        self.photo = customtkinter.CTkImage(PIL.Image.open(self.image_path), size=(400, 320))
        self.image_label = customtkinter.CTkLabel(self, text='', image=self.photo)
        self.image_label.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        """
        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=2, columnspan=1, padx=(20, 10), pady=(20, 0), sticky="nsew")
        self.tabview.add("Control")
        self.tabview.add("Details")
        self.tabview.add("Tab 3")
        
        self.tabview.tab("Control").grid_columnconfigure(0, weight=1, minsize=150)
        self.tabview.tab("Control").grid_columnconfigure(1, weight=1, minsize=150)
        self.tabview.tab("Details").grid_columnconfigure(0, weight=1, minsize=300)
        self.tabview.tab("Tab 3").grid_columnconfigure(0, weight=1, minsize=300)
        #self.tabview.tab("Tab 3").configure(state='disabled')

        self.selection_label_tab_1 = customtkinter.CTkLabel(self.tabview.tab("Control"), text="Select a file or folder to start audio checking")
        self.selection_label_tab_1.grid(row=0, column=0, padx=0, pady=20, sticky="ew", columnspan=2)
        self.perform_prediction_button = customtkinter.CTkButton(self.tabview.tab("Control"), text="Perform Predictions", command=self.perform_prediction, state="disabled", fg_color="#cc7a00", hover_color="#ab4d00")
        self.perform_prediction_button.grid(row=2, column=0, padx=20, pady=(10, 10), sticky="nsew")
        self.perform_massprediction_button = customtkinter.CTkButton(self.tabview.tab("Control"), text="Perform Mass Predictions", command=self.perform_mass_prediction, state="normal", fg_color="#cc7a00", hover_color="#ab4d00")
        self.detailed_history_button = customtkinter.CTkButton(self.tabview.tab("Control"), text="Detailed History", command=self.perform_prediction, state="disabled")
        self.detailed_history_button.grid(row=2, column=1, padx=20, pady=(10, 10), sticky="nsew")
        self.result_directory_button = customtkinter.CTkButton(self.tabview.tab("Control"), text="Open CSV File Directory", command=self.perform_prediction, state="disabled")
        self.result_directory_button.grid(row=3, column=0, padx=20, pady=(10, 10), sticky="nsew")
        self.image_directory_button = customtkinter.CTkButton(self.tabview.tab("Control"), text="Open Image Directory", command=self.perform_prediction, state="disabled")
        self.image_directory_button.grid(row=3, column=1, padx=20, pady=(10, 10), sticky="nsew")
        
        self.label_tab_2 = customtkinter.CTkLabel(self.tabview.tab("Details"), text="CTkLabel on Tab 2")
        self.label_tab_2.grid(row=0, column=0, padx=20, pady=20)
        
        self.label_tab_3 = customtkinter.CTkLabel(self.tabview.tab("Tab 3"), text="CTkLabel on Tab 3")
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

        # create scrollable frame
        """self.scrollable_frame = customtkinter.CTkScrollableFrame(self, label_text="Results")
        self.scrollable_frame.grid(row=1, column=2, padx=(20, 10), pady=(20, 0), sticky="nsew")
        self.scrollable_frame.grid_columnconfigure(0, weight=1)"""
        
        self.result_tabview = customtkinter.CTkTabview(self, width=250)
        self.result_tabview.grid(row=1, column=2, columnspan=1, padx=(20, 10), pady=(20, 0), sticky="nsew")
        self.result_tabview.add("Solo Results")
        self.result_tabview.add("Mass Results")
        
        self.result_tabview.tab("Solo Results").grid_columnconfigure(0, weight=1, minsize=300)
        self.result_tabview.tab("Mass Results").grid_columnconfigure(0, weight=1, minsize=300)
        
        #self.selection_label_tab_1 = customtkinter.CTkLabel(self.tabview.tab("Control"), text="Select a file or folder to start audio checking")
        #self.selection_label_tab_1.grid(row=0, column=0, padx=0, pady=20, sticky="ew")

        #self.label_tab_2 = customtkinter.CTkLabel(self.tabview.tab("Details"), text="CTkLabel on Tab 2")
        #self.label_tab_2.grid(row=0, column=0, padx=20, pady=20)
        
        """self.result_frame = customtkinter.CTkFrame(self)
        self.result_frame.grid(row=1, column=2, padx=(20, 10), pady=(20, 0), sticky="nsew")
        self.result_frame.grid_columnconfigure(0, weight=1)"""
        
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

    """def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)"""

    def perform_prediction(self):
        #event = threading.Event()
        spectrogram_file_path = generate_spectrogram()
        #event.wait()
        
        print(spectrogram_file_path)
        self.spectrogram_image_path = os.path.join(os.path.dirname(__file__), spectrogram_file_path)
        self.photo = customtkinter.CTkImage(img.open(self.spectrogram_image_path), size=(400, 320))
        self.image_label = customtkinter.CTkLabel(self, text='', image=self.photo)
        self.image_label.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.sidebar_button_3_event()
        self.tabview.set("Details")
        
    def perform_mass_prediction(self):
        #event = threading.Event()
        mass_csv_output_path = mass_generate_spectrogram(self.mass_folder_path)
        #event.wait()
        if create_table_mass(self.result_tabview.tab("Mass Results"), mass_csv_output_path):
            self.text_label_tab2.forget()
        """print(spectrogram_folder_path)
        self.spectrogram_image_path = os.path.join(os.path.dirname(__file__), spectrogram_folder_path)
        self.photo = customtkinter.CTkImage(img.open(self.spectrogram_image_path), size=(400, 320))
        self.image_label = customtkinter.CTkLabel(self, text='', image=self.photo)
        self.image_label.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")"""
        self.sidebar_button_3_event()
        self.tabview.set("Details")

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

if __name__ == "__main__":
    app = App()
    app.mainloop()

"""
style = ttk.Style()
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
                background=[('active', '#3484F0')])`
"""