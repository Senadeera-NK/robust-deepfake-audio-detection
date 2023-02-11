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

# create the final dataset

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

train_images, test_images, train_labels, test_labels = train_test_split(spectrogram_images_resized, labels_one_hot, test_size=0.2, random_state=42)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

train_labels = keras.utils.to_categorical(train_labels, num_classes=2)
test_labels = keras.utils.to_categorical(test_labels, num_classes=2)

# Define the model architecture
height, width = spectrogram_images_resized[0].shape[0], spectrogram_images_resized[0].shape[1]
num_classes = 2

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    #keras.layers.Dense(2, activation='softmax')
    keras.layers.Dense(2 * num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))