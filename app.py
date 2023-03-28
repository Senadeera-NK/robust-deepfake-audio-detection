# user should be able to upload audio files
# user should be able to upload background noises

# trained model should be used to classify the deeepfake/real audios

# user should be able to download deepfake audios and real audios seperately

from flask import Flask, render_template, request,redirect, url_for, jsonify
from werkzeug.utils import secure_filename

# to load the model
from tensorflow.keras.models import load_model

import numpy as np
import tensorflow as tf
import os
import sys

# for preprocessing the audio file
from skimage.transform import resize
#import cv2
# instead of cv2, using guassian_filter here
from scipy.ndimage.filters import gaussian_filter

import librosa


#sys.path.append('D:\\CS FINAL YEAR\\Final Project\\my project\\robust-deepfake-audio-detection\\venv\\Lib\\site-packages')

# to get the current directory
current_dir = os.getcwd()
model_file = 'best_model_3.h5'

model_path = os.path.join(current_dir, model_file)

# load the saved model
model = load_model(model_path)

#define a function to preprocess the audio file before feeding it to the model
def preprocess_audio(audio_path):
   # # load the audio file
   # signal, sr = librosa.load(audio_path, sr=16000)

   # #trim the silence from the start and end of the audio
   # signal,_ = librosa.effects.trim(signal)

   # # extract features using mel spectogram
   # spectogram = librosa.feature.melspectrogram(signal,sr=8000,n_mels=128)
   # log_mel_spectrogram = librosa.amplitude_to_db(spectogram, ref=np.max)

   # # normalize the spectogram
   # normalized_spectogram = (log_mel_spectrogram+80) / 8.0

   # add an additional dimension to the spectogram for the model input
   #return normalized_spectogram.reshape(1,128,173,1)


   #audio_file = [os.path.join(audio_path, f) for f in os.listdir(audio_path) if f.endswith('.wav')]
   audio, sr = librosa.load(audio_path, sr=None)
   spectrogram = librosa.feature.melspectrogram(y=audio,sr=sr)
   spectrogram = librosa.power_to_db(spectrogram)
   spectrogram = np.array(spectrogram, dtype=np.float32)
   spectrogram_resized = resize(spectrogram, (128,128))
   spectrogram_resized = gaussian_filter(spectrogram_resized,sigma=(5,5), mode='constant')
   spectrogram_resized = np.repeat(spectrogram_resized[:,:,np.newaxis], 3, axis=-1)
   spectrogram_resized = spectrogram_resized / 255.0
   return spectrogram_resized

# FLASK APPLICATION BEGINS HERE
app = Flask(__name__)

# function to upload the audio file/files
@app.route('/')
def home():
   return render_template('upload.html')


@app.route('/upload-audio', methods = ['POST'])
def upload_audio():
   audio_files = request.files.getlist('audio-file')
   filepaths = []

   for audio_file in audio_files:
      filename = secure_filename(audio_file.filename)
      if audio_file.filename == '':
         return 'No selected file'
      if audio_file:
         # save the audio file
         filepath = current_dir + '/audios/' + filename
         audio_file.save(filepath)
         filepaths.append(filepath)
   #redirect to classification route
   print('hi clsasify')
   return redirect(url_for('classify_audio', filepaths=filepaths))

@app.route('/classify-audio', methods = ['GET'])
def classify_audio():
    filepaths = request.args.getlist('filepaths')
    results = []
    for filepath in filepaths:
        # preprocess the audio file
        preprocessed_audio = preprocess_audio(filepath)

        # run the model to detect if the audio is a deepfake or not
        prediction = model.predict(preprocessed_audio)

        # if the model predicts a label of 0, the audio is a deepfake
        # if the model predicts a label of a 1, the audio is a real
        if prediction[0][0] < 0.5:
            result = 'real audio'
        else:
            result = 'deepfake audio'
        results.append(result)
    print('hi')
    print(results)
    return jsonify(results)

@app.route('/loading')
def show_loading():
   return render_template('loading.html')

@app.route('/results')
def show_results():
   results = request.args.getlist('results')
   if not results:
      return redirect(url_for('home'))
   return render_template('result.html', results=results)

#running the app
if __name__ == '__main__':
   app.run(debug = True)