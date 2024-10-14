from tkinter import *
from tkinter import filedialog
import pygame
import time
from mutagen.mp3 import MP3
import tkinter.ttk as ttk
import os

is_playing = False
paused = False
current_time = 0
filepath = ''
song_length = 0

def play_music(slider):
    global is_playing, paused
    if is_playing:
        if pygame.mixer.music.get_busy() == False:
            stop_music(slider)
            is_playing = False
            play_music(slider)
            return
        if paused:
            unpause_music(slider)
            return
        else: 
            pause_music()
            return
    else:
        is_playing = True
        paused = False
        #print("NEW", current_time)
        play_time(slider)
        if current_time > 0:
            pygame.mixer.music.play(loops=0, start=current_time)
        else:
            pygame.mixer.music.play(loops=0)
        
def slide(x, slider):
    pygame.mixer.music.set_pos(int(slider.get()))
    cur_slider = slider.get()
    slider.set(cur_slider)
    global current_time
    current_time = slider.get()

def pause_music():
    global paused
    pygame.mixer.music.pause()
    paused = True

def unpause_music(slider):
    global paused
    pygame.mixer.music.unpause()
    paused = False
    play_time(slider)

def stop_music(slider):
    global is_playing
    pygame.mixer.music.stop()
    is_playing = False
    slider.set(0)
    
def load_song(file_path):
    global song_length, filepath
    filepath = file_path
    file_ext = os.path.splitext(file_path)[1]
    if file_ext == ".mp3":
        pygame.mixer.music.load(file_path)
        song_mut = MP3(file_path)
        song_length = song_mut.info.length
    elif file_ext in [".wav", ".ogg"]:
        pygame.mixer.music.load(file_path)
        sound = pygame.mixer.Sound(file_path)
        song_length = sound.get_length()
    return song_length

def update_audio_status():
    pass

def set_volume(x):
    pygame.mixer.music.set_volume(x)
    return
    
def play_time(slider):
    if not is_playing:
        return
    if paused:
        return

    global song_length, current_time, filepath
    converted_current_time = time.strftime('%M:%S', time.gmtime(current_time))

    file_ext = os.path.splitext(filepath)[1]

    if file_ext == ".mp3":
        song_mut = MP3(filepath)
        song_length = song_mut.info.length
    else:
        pass

    converted_song_length = time.strftime('%M:%S', time.gmtime(song_length))

    current_time += 1

    if int(slider.get()) == int(song_length):
        stop_music(slider)
        current_time = 0
        next_time = 0
        return
    elif paused:
        return
    else:
        converted_current_time = time.strftime('%M:%S', time.gmtime(int(slider.get())))
        next_time = int(current_time) + 1
        slider.set(next_time)

    global task_id
    if is_playing and not paused:
        try:
            task_id = slider.after(1000, lambda: play_time(slider))
        except Exception as e:
            print(f"Error: {e}")

pygame.mixer.init()
