# ----- pre-processing ------ #
!pip install patool
from google.colab import drive
drive.mount('/content/drive')
!pip install pydub
!pip install tensorflow

import os
#from pydub import AudioSegment
# for audio processing
#import librosa.display
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras_preprocessing.sequence import pad_sequences

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

# for RNN model
from tensorflow.keras.layers import Input, Dense, GRU
from tensorflow.keras.models import Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Embedding, Flatten, Reshape

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

preprocessed_real_audio_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset simple/mixed-real-preprocessed'

preprocessed_fake_audio_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset simple/mixed-fake-preprocessed'

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

# ONE TIME RUNING
# performing spectral subtraction on real pre-processed audios
for filename in (real_audios_with_noises):
    if filename.endswith(".wav"):
        file_path = os.path.join(real_audios_with_noises_folder, filename)
        audio, sr = librosa.load(file_path, sr=None)
        processed_audio = spectral_subtraction(audio)
        processed_file_path = os.path.join(preprocessed_real_audio_folder, filename)
        sf.write(processed_file_path, processed_audio, sr)

# ONE TIME RUNNING
# performing spectral subtraction on fake pre-processed audios
for filename in (fake_audios_with_noises):
    if filename.endswith(".wav"):
        file_path = os.path.join(fake_audios_with_noises_folder, filename)
        audio, sr = librosa.load(file_path, sr=None)
        processed_audio = spectral_subtraction(audio)
        processed_file_path = os.path.join(preprocessed_fake_audio_folder, filename)
        sf.write(processed_file_path, processed_audio, sr)

# ------ creating the final pre-processed dataset ------ #
# loading all real preprocessed audios from the folder
real_preprocessed_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset simple/mixed-real-preprocessed'
real_preprocessed_files = [os.path.join(real_preprocessed_folder, f) for f in os.listdir(real_preprocessed_folder) if f.endswith('.wav')]

# real preprocessed files storing to a list and labeling them as 'real'
real_preprocessed_audios_data = []
for file in real_preprocessed_files:
    audio, sr = librosa.load(file, sr=None)
    real_preprocessed_audios_data.append((audio, sr, 'real'))

# loading all fake preprocessed audios from the folder
fake_preprocessed_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset simple/mixed-fake-preprocessed'
fake_preprocessed_files = [os.path.join(fake_preprocessed_folder, f) for f in os.listdir(fake_preprocessed_folder) if f.endswith('.wav')]

# fake preprocessed files storing to a list and labelling them as 'fake'
fake_preprocessed_audios_data = []
for file in fake_preprocessed_files:
    audio, sr = librosa.load(file, sr=None)
    fake_preprocessed_audios_data.append((audio, sr, 'fake'))

all_audio_data = real_preprocessed_audios_data + fake_preprocessed_audios_data

# Convert the list of audio data into a numpy array
all_audio_data = np.array(all_audio_data)

# Divide the dataset into training (80%) and testing (20%) datasets
train_data = all_audio_data[:int(0.8 * len(all_audio_data))]
test_data = all_audio_data[int(0.8 * len(all_audio_data)):]

# Separate the audio and labels into different arrays
train_audios = np.array([x[0] for x in train_data])
train_labels = np.array([1 if x[2] == 'fake' else 0 for x in train_data])
test_audios = np.array([x[0] for x in test_data])
test_labels = np.array([1 if x[2] == 'fake' else 0 for x in test_data])


train_audios = np.array([x[0] for x in train_data])
train_labels = np.array([x[2] for x in train_data])

# find the length of the longest audio sample
max_length = max([len(audio) for audio in train_audios])

# pad shorter audio samples with zeros to make all audio samples have the same length
train_audios = np.array([np.pad(audio, (0, max_length - len(audio)), 'constant') for audio in train_audios])

# 1- => 'train_audios.shape[0]'
train_audios = train_audios.reshape(train_audios.shape[0], max_length, 1)


test_audios = np.array([x[0] for x in test_data])
test_labels = np.array([x[2] for x in test_data])

# find the length of the longest audio sample
max_length = max([len(audio) for audio in test_audios])

# pad shorter audio samples with zeros to make all audio samples have the same length
test_audios = np.array([np.pad(audio, (0, max_length - len(audio)), 'constant') for audio in test_audios])

# 1- => 'test_audios.shape[0]'
test_audios = test_audios.reshape(test_audios.shape[0], max_length, 1)

# ------- autoencoder model design -------- #
#len(train_audios[0]) =>'210944'
#max_length => 210944
inputs = tf.keras.layers.Input(shape=(210944, 1))
encoded = tf.keras.layers.Dense(32, activation='relu')(inputs)

# len(train_audios[0]) => 1
#max_length => len(train_audios[0])
decoded = tf.keras.layers.Dense(max_length, activation='sigmoid')(encoded)
autoencoder = tf.keras.models.Model(inputs, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# Find the length of the longest audio sample
max_length = max([audio.shape[0] for audio in train_audios])

# Pad the audio signals with zeros to make them all have the same length
train_audios_padded = np.zeros((train_audios.shape[0], max_length, 1))
for i, audio in enumerate(train_audios):
    audio_padded = np.pad(audio, [(0, max_length - audio.shape[0]), (0, 0)], 'constant')
    train_audios_padded[i, :, :] = audio_padded



# Find the length of the longest audio sample
max_length = max([audio.shape[0] for audio in test_audios])

# Pad the audio signals with zeros to make them all have the same length
test_audios_padded = np.zeros((test_audios.shape[0], max_length, 1))
for i, audio in enumerate(test_audios):
    audio_padded = np.pad(audio, [(0, max_length - audio.shape[0]), (0, 0)], 'constant')
    test_audios_padded[i, :, :] = audio_padded

# Train the autoencoder model
autoencoder.fit(train_audios_padded, train_audios_padded,
                epochs=100,
                batch_size=32,
                shuffle=True,
                validation_data=(test_audios_padded, test_audios_padded))