import os
import random
import cv2
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from keras.utils import to_categorical
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, GRU, Concatenate, Embedding, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from sklearn.metrics import accuracy_score
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
# data augmentation for overfitting
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# convert audio files to spectrogram images
def preprocess_audio(real_preprocessed_folder, fake_preprocessed_folder):
    real_preprocessed_files = [os.path.join(real_preprocessed_folder, f) for f in os.listdir(real_preprocessed_folder) if f.endswith('.wav')]

    num_files = len(real_preprocessed_files)
    print("Number of real files in folder:", num_files)

    real_preprocessed_audios_data = []
    for file in real_preprocessed_files:
        audio, sr = librosa.load(file, sr=None)
        real_preprocessed_audios_data.append((audio, sr, 'real'))

    fake_preprocessed_files = [os.path.join(fake_preprocessed_folder, f) for f in os.listdir(fake_preprocessed_folder) if f.endswith('.wav')]

    num_files = len(fake_preprocessed_files)
    print("Number of fake files in folder:", num_files)

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

    # apply denoising filter to train and test images
    train_images = [cv2.GaussianBlur(img, (5, 5), 0) for img in train_images]
    test_images = [cv2.GaussianBlur(img, (5, 5), 0) for img in test_images]

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

    datagen.fit(train_images)

    return train_images, test_images, train_labels, test_labels



def build_model():
    base_model = MobileNetV2(input_shape=(128, 128, 3),include_top=False, weights='imagenet')
    dropout_rate = 0.3
    kernel_regularizer = l2(0.005)
    models = []

    for i in range(3):
        print(f'Creating model {i+1}')
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(256, activation='relu', kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(128, activation='relu', kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        predictions = Dense(2, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        optimizer = SGD(learning_rate=0.01, momentum=0.9)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        models.append(model)

    return models


models= build_model()

epochs = 200
batch_size = 218

if len(train_images) < 1000:
    batch_size = 16
    epochs = 50
elif len(train_images) < 5000:
    batch_size = 32
    epochs = 75
else:
    batch_size = 128
    epochs = 200

# Train ensemble models using KFold cross-validation
n_splits = 5
if n_splits > len(models):
    n_splits = len(models)
print(f'Training {n_splits} models using KFold cross-validation')
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
for i, (train_idx, val_idx) in enumerate(kfold.split(train_images, train_labels)):
    print(f'Training model {i+1}')
    # Define callbacks for early stopping and saving the best model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(f'best_model_{i+1}.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)

    # Train model
    history = models[i].fit(
        datagen.flow(train_images[train_idx], train_labels[train_idx], batch_size=batch_size),
        validation_data=(test_images, test_labels),
        epochs=epochs,
        callbacks=[es, mc, reduce_lr]
    )

# Evaluate models on test data and calculate ensemble accuracy
ensemble_predictions = np.zeros_like(test_labels)
for model in models:
    predictions = model.predict(test_images)
    ensemble_predictions += predictions
ensemble_predictions /= n_splits
ensemble_predictions = np.argmax(ensemble_predictions, axis=-1)
test_accuracy = np.mean(ensemble_predictions == np.argmax(test_labels, axis=-1))
print(f'Ensemble test accuracy: {test_accuracy}')



