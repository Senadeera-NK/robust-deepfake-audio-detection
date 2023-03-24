!pip install patool

# no need in here
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

#global average pooling layer
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense, BatchNormalization, Conv2D

# for adam optimizer
from tensorflow.keras.optimizers import Adam

# converting to images
import matplotlib.pyplot as plt

from skimage.transform import resize

# to convert labels to hot-encoded
from keras.utils import to_categorical

# 'real' and 'fake' into (0,1)
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# for early stopping
from keras.callbacks import EarlyStopping
# for dropout to reduce overfitting
from keras.layers import Dropout

# data augmentation for overfitting
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# define base model with dropout layer and kernel regularization
from tensorflow.keras.regularizers import l2

#to add more callbacks, using a learning rate scheduler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# for the plot to show the performance, accuracy of the model
import matplotlib.pyplot as plt

# for cross-validation
from sklearn.model_selection import KFold

# create the final dataset

# loading all real preprocessed audios from the folder
real_preprocessed_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset new 1000/mixed-real-preprocessed'
real_preprocessed_files = [os.path.join(real_preprocessed_folder, f) for f in os.listdir(real_preprocessed_folder) if f.endswith('.wav')]

# real preprocessed files storing to a list and labeling them as 'real'
real_preprocessed_audios_data = []
for file in real_preprocessed_files:
    audio, sr = librosa.load(file, sr=None)
    real_preprocessed_audios_data.append((audio, sr, 'real'))

# loading all fake preprocessed audios from the folder
fake_preprocessed_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset new 1000/mixed-fake-preprocessed'
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
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
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

# new 
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# new
train_images = np.repeat(train_images, 3, axis=-1)
test_images = np.repeat(test_images, 3, axis=-1)

#new- Normalize image data
train_images = train_images / 255.0
test_images = test_images / 255.0

# new
# define data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# new
# fit data augmentation generator to training data
datagen.fit(train_images)

# new
# DONT CHANGE (it wont be compatible with the data type) - (128,128,3)
base_model = MobileNetV2(input_shape=(128, 128, 3),include_top=False, weights='imagenet')


# add dropout layer
dropout_rate = 0.3
kernel_regularizer = l2(0.001)

x = base_model.output
# Batch normalization is a technique for improving the speed, performance, and stability of deep neural network
x = BatchNormalization()(x)
#new
x = Dropout(dropout_rate)(x)
x = Conv2D(256, (3,3), padding='same', activation='relu', kernel_regularizer=kernel_regularizer)(x)
x = Dropout(dropout_rate)(x)
x = Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=kernel_regularizer)(x)
x = GlobalAveragePooling2D()(x)
#new
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)

#DONT CHANGE THIS (ERRORS HAPPENED)
predictions = Dense(2, activation='softmax', kernel_regularizer=kernel_regularizer)(x)

model = keras.models.Model(inputs=base_model.input, outputs=predictions)

# new
# fine-tune the pre-trained model
for layer in base_model.layers[:-10]:
    layer.trainable = False

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy')
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
callbacks = [early_stopping, model_checkpoint, lr_scheduler]

#NEW - TO IMPROVE GENERALIZATION
# define k-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

epochs = 30
batch_size = 64

#NEW - TO IMPROVE THE GENERALIZATION
# perform k-fold cross-validation
histories = []
for i, (train_idx, val_idx) in enumerate(kf.split(train_images, train_labels)):
    print(f'Fold {i+1}/{k}')
    # split data into training and validation sets
    x_train, y_train = train_images[train_idx], train_labels[train_idx]
    x_val, y_val = train_images[val_idx], train_labels[val_idx]
    # train model on this fold
    history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stopping, model_checkpoint, lr_scheduler])
