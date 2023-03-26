import os
import numpy as np
import pandas as pd
import random
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from skimage.transform import resize
from sklearn.model_selection import KFold
import cv2

# create the final dataset

# loading all real preprocessed audios from the folder
real_preprocessed_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset new 1000/mixed-real-preprocessed'
real_preprocessed_files = [os.path.join(real_preprocessed_folder, f) for f in os.listdir(real_preprocessed_folder) if f.endswith('.wav')]

num_files = len(real_preprocessed_files)
print("Number of real files in folder:", num_files)

# real preprocessed files storing to a list and labeling them as 'real'
real_preprocessed_audios_data = []
for file in real_preprocessed_files:
    audio, sr = librosa.load(file, sr=None)
    real_preprocessed_audios_data.append((audio, sr, 'real'))

# loading all fake preprocessed audios from the folder
fake_preprocessed_folder = '/content/drive/MyDrive/FINAL YEAR/FYP/FINAL PROJECT/audio training dataset new 1000/mixed-fake-preprocessed'
fake_preprocessed_files = [os.path.join(fake_preprocessed_folder, f) for f in os.listdir(fake_preprocessed_folder) if f.endswith('.wav')]

num_files = len(fake_preprocessed_files)
print("Number of fake files in folder:", num_files)

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


# model.py functions
# define the model architecture
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(256, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model

# compile the model
def compile_model(model, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# train the model
def train_model(model, train_images, train_labels, val_images, val_labels, batch_size=32, epochs=50, callbacks=None):
    history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(val_images, val_labels), callbacks=callbacks)
    return history

# evaluate the model
def evaluate_model(model, test_images, test_labels):
    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

# save the model
def save_model(model, filepath):
    model.save(filepath)

# load the model
def load_model(filepath):
    model = keras.models.load_model(filepath)
    return model

