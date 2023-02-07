!pip install patool

from google.colab import drive
drive.mount('/content/drive')

!pip install pydub

!pip install tensorflow

import os
#from pydub import AudioSegment
# for audio processing
#import librosa.display

import soundfile as sf
import zipfile as zf
import patoolib
import numpy as np
import pandas as pd
import librosa
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers

# for autoencoder
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# ONE TIME RUNNING
# extracting background noises dataset's zip folder
files = zf.ZipFile('/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/background-noises-dataset.zip', 'r')
files.extractall('/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/background-noises-dataset')
files.close()

# ONE TIME RUNNING
# extracting audio fake files dataset's zip folder (for training)
files = zf.ZipFile('/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/fake.zip', 'r')
files.extractall('/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset')
files.close()

# ONE TIME RUNNING
# extracting audio real files dataset's zip folder (for training)
files = zf.ZipFile('/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/real.zip', 'r')
files.extractall('/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset')
files.close()

# set the paths to the folders containing the audio files
audios_real_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset/real/'
audios_fake_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset/fake/'
audios_noises_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/background-noises-dataset/background-noises-dataset/'

# extracting the folders' files to lists
real_audios = os.listdir(audios_real_folder)
fake_audios = os.listdir(audios_fake_folder)
noises_audios = os.listdir(audios_noises_folder)


# ONE IME RUN - AND SAVE TO A FOLDER
# mixing the fake audio files with background noises randomly
for filename in fake_audios:
    if filename.endswith(".wav"):
        fake_audio_path = os.path.join(audios_fake_folder, filename)
        fake_audio, sr = librosa.load(fake_audio_path)
        
        # Choose a random background noise sample from the background noise dataset
        random_background_noise_path = os.path.join(audios_noises_folder, np.random.choice(noises_audios))
        background_noise, sr = librosa.load(random_background_noise_path, sr=sr)
        
        if fake_audio.shape[0] > background_noise.shape[0]:
            background_noise = np.tile(background_noise, int(np.ceil(fake_audio.shape[0] / background_noise.shape[0])))
            background_noise = background_noise[:fake_audio.shape[0]]

        else:
            background_noise = background_noise[:fake_audio.shape[0]]

        mixed_audio = fake_audio + background_noise
        
        mixed_audio_path = os.path.join('/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset/mixed-fake', filename)
        sf.write(mixed_audio_path, mixed_audio, sr)


# ONE IME RUN - AND SAVE TO A FOLDER
# mixing the real audio files with background noises randomly
for filename in real_audios:
    if filename.endswith(".wav"):
        real_audio_path = os.path.join(audios_real_folder, filename)
        real_audio, sr = librosa.load(real_audio_path)
        
        # Choose a random background noise sample from the background noise dataset
        random_background_noise_path = os.path.join(audios_noises_folder, np.random.choice(noises_audios))
        background_noise, sr = librosa.load(random_background_noise_path, sr=sr)
        
        if real_audio.shape[0] > background_noise.shape[0]:
            background_noise = np.tile(background_noise, int(np.ceil(real_audio.shape[0] / background_noise.shape[0])))
            background_noise = background_noise[:real_audio.shape[0]]

        else:
            background_noise = background_noise[:real_audio.shape[0]]

        mixed_audio = real_audio + background_noise
        
        mixed_audio_path = os.path.join('/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset/mixed-real', filename)
        sf.write(mixed_audio_path, mixed_audio, sr)

# set the paths to the folders containing the fake/real audio files which mixed with background noises
real_audios_with_noises_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset/mixed-real'
fake_audios_with_noises_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset/mixed-fake'

# extracting the fake/real mixed folders' files to lists
real_audios_with_noises = os.listdir(real_audios_with_noises_folder)
fake_audios_with_noises = os.listdir(fake_audios_with_noises_folder)

preprocessed_real_audio_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset/mixed-real-preprocessed'

preprocessed_fake_audio_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset/mixed-fake-preprocessed'

# defining a method for pre-process the audios using spectral subtraction method
def spectral_subtraction(audio):
    # Perform Spectral Subtraction on the audio signal
    magnitude_spectrogram = np.abs(librosa.stft(audio))
    power_spectrogram = magnitude_spectrogram**2
    avg_power = np.mean(power_spectrogram, axis=1)
    power_spectrogram = np.transpose(np.transpose(power_spectrogram)/avg_power)
    power_spectrogram[power_spectrogram <= 1.0] = 1.0
    magnitude_spectrogram = np.sqrt(power_spectrogram)
    enhanced_audio = librosa.istft(magnitude_spectrogram * np.exp(1j * np.angle(librosa.stft(audio))))
    return enhanced_audio

# performing spectral subtraction on real pre-processed audios
for filename in (real_audios_with_noises):
    if filename.endswith(".wav"):
        file_path = os.path.join(real_audios_with_noises_folder, filename)
        audio, sr = librosa.load(file_path, sr=None)
        processed_audio = spectral_subtraction(audio)
        processed_file_path = os.path.join(preprocessed_real_audio_folder, filename)
        sf.write(processed_file_path, processed_audio, sr)

# performing spectral subtraction on fake pre-processed audios
for filename in (fake_audios_with_noises):
    if filename.endswith(".wav"):
        file_path = os.path.join(fake_audios_with_noises_folder, filename)
        audio, sr = librosa.load(file_path, sr=None)
        processed_audio = spectral_subtraction(audio)
        processed_file_path = os.path.join(preprocessed_fake_audio_folder, filename)
        sf.write(processed_file_path, processed_audio, sr)


# create the final dataset

# loading all real preprocessed audios from the folder
real_preprocessed_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset/mixed-real-preprocessed'
real_preprocessed_files = [os.path.join(real_preprocessed_folder, f) for f in os.listdir(real_preprocessed_folder) if f.endswith('.wav')]

# real preprocessed files storing to a list and labeling them as 'real'
real_preprocessed_audios_data = []
for file in real_preprocessed_files:
    audio, sr = librosa.load(file, sr=None)
    real_preprocessed_audios_data.append((audio, sr, 'real'))

# loading all fake preprocessed audios from the folder
fake_preprocessed_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset/mixed-fake-preprocessed'
fake_preprocessed_files = [os.path.join(fake_preprocessed_folder, f) for f in os.listdir(fake_preprocessed_folder) if f.endswith('.wav')]

# fake preprocessed files storing to a list and labelling them as 'fake'
fake_preprocessed_audios_data = []
for file in fake_preprocessed_files:
    audio, sr = librosa.load(file, sr=None)
    fake_preprocessed_audios_data.append((audio, sr, 'fake'))

all_audio_data = real_preprocessed_audios_data + fake_preprocessed_audios_data

# split the dataset into training and testing dataset
X = [(audio, sr) for (audio, sr, label) in all_audio_data]
y = [label for (audio, sr, label) in all_audio_data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
