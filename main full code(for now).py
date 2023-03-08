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
#from keras_preprocessing.sequence import pad_sequences

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

# converting to images
import matplotlib.pyplot as plt

from skimage.transform import resize

# to convert labels to hot-encoded
from keras.utils import to_categorical

# 'real' and 'fake' into (0,1)
from sklearn.preprocessing import LabelEncoder

# for pre-trained model VGG16
#import keras
#from keras.applications import VGG16
#from keras.models import Model
#from keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# ---------------------- data pre-processing part ------------------- #
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


# create the final dataset
# loading all real preprocessed audios from the folder
real_preprocessed_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset 500/mixed-real-preprocessed'
real_preprocessed_files = [os.path.join(real_preprocessed_folder, f) for f in os.listdir(real_preprocessed_folder) if f.endswith('.wav')]

# real preprocessed files storing to a list and labeling them as 'real'
real_preprocessed_audios_data = []
for file in real_preprocessed_files:
    audio, sr = librosa.load(file, sr=None)
    real_preprocessed_audios_data.append((audio, sr, 'real'))

# loading all fake preprocessed audios from the folder
fake_preprocessed_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset 500/mixed-fake-preprocessed'
fake_preprocessed_files = [os.path.join(fake_preprocessed_folder, f) for f in os.listdir(fake_preprocessed_folder) if f.endswith('.wav')]

# fake preprocessed files storing to a list and labelling them as 'fake'
fake_preprocessed_audios_data = []
for file in fake_preprocessed_files:
    audio, sr = librosa.load(file, sr=None)
    fake_preprocessed_audios_data.append((audio, sr, 'fake'))

all_audio_data = real_preprocessed_audios_data + fake_preprocessed_audios_data

# ----------- model building starts -------------#
spectrogram_images = []
labels = []

for audio, sr, label in all_audio_data:
    spectrogram = librosa.feature.melspectrogram(audio, sr=sr)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = np.array(spectrogram, dtype=np.float32)
    spectrogram_images.append(spectrogram)
    labels.append(label)

spectrogram_images_resized = []

for spectrogram_image in spectrogram_images:
    spectrogram_image_resized = resize(spectrogram_image, (128, 128))
    spectrogram_images_resized.append(spectrogram_image_resized)
spectrogram_images = np.array(spectrogram_images_resized)

le = LabelEncoder()
labels_integer = le.fit_transform(labels)
labels_one_hot = to_categorical(labels_integer, num_classes=2)

# deviding the dataset into train and test
train_images, test_images, train_labels, test_labels = train_test_split(spectrogram_images_resized, labels_one_hot, test_size=0.2, random_state=42)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# new
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# new
train_images = np.repeat(train_images, 3, axis=-1)
test_images = np.repeat(test_images, 3, axis=-1)

# base model MobileNetV2
base_model = MobileNetV2(input_shape=(128, 128, 3),include_top=False, weights='imagenet')

# new
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(2, activation='softmax')
])

# compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# new
# Train the model
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))