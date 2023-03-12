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

# for early stopping
from keras.callbacks import EarlyStopping
# for dropout to reduce overfitting
from keras.layers import Dropout

# data augmentation for overfitting
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# create the final dataset

# loading all real preprocessed audios from the folder
real_preprocessed_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset 1000/mixed-real-preprocessed'
real_preprocessed_files = [os.path.join(real_preprocessed_folder, f) for f in os.listdir(real_preprocessed_folder) if f.endswith('.wav')]

# real preprocessed files storing to a list and labeling them as 'real'
real_preprocessed_audios_data = []
for file in real_preprocessed_files:
    audio, sr = librosa.load(file, sr=None)
    real_preprocessed_audios_data.append((audio, sr, 'real'))

# loading all fake preprocessed audios from the folder
fake_preprocessed_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset 1000/mixed-fake-preprocessed'
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

print(np.shape(labels))
print(np.shape(spectrogram_images_resized))

le = LabelEncoder()
labels_integer = le.fit_transform(labels)
labels_one_hot = to_categorical(labels_integer, num_classes=2)

print(np.shape(labels_one_hot))
print(np.shape(spectrogram_images_resized))

train_images, test_images, train_labels, test_labels = train_test_split(spectrogram_images_resized, labels_one_hot, test_size=0.2, random_state=42)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

print(train_labels.shape)
print(test_labels.shape)

# new
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# new
train_images = np.repeat(train_images, 3, axis=-1)
test_images = np.repeat(test_images, 3, axis=-1)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# new
# define data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True, 
    fill_mode='nearest'
)

# new
# fit data augmentation generator to training data
datagen.fit(train_images)

# new
base_model = MobileNetV2(input_shape=(128, 128, 3),include_top=False, weights='imagenet')

# add dropout layer
dropout_rate = 0.2
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(dropout_rate)(x)
predictions = keras.layers.Dense(2, activation='softmax')(x)

model = keras.models.Model(inputs=base_model.input, outputs=predictions)

# new
# fine-tune the pre-trained model
for layer in base_model.layers[:-4]:
    layer.trainable = False

for layer in base_model.layers[-4:]:
    layer.trainable = True

# new
#model = keras.models.Model(inputs=model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#new
#fit the model
epochs = 20
batch_size = 32
#new
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

#New
history = model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(test_images, test_labels),
                    callbacks=[early_stopping],
                    verbose=1)