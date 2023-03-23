# user should be able to upload audio files
# user should be able to upload background noises

# trained model should be used to classify the deeepfake/real audios

# user should be able to download deepfake audios and real audios seperately

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import librosa
import numpy as np
import tensorflow as tf
import os

# load the saved model
model = tf.keras.models.load_model('model.h5')

# define a function to preprocess the audio file before feeding it to the model
def preprocess_audio(audio_path):
   # load the audio file
   signal, sr = librosa.load(audio_path, sr=16000)

   #trim the silence from the start and end of the audio
   signal,_ = librosa.effects.trim(signal)

   # extract features using mel spectogram
   spectogram = librosa.feature.melspectrogram(signal,sr=8000,n_mels=128)
   log_mel_spectrogram = librosa.amplitude_to_db(spectogram, ref=np.max)

   # normalize the spectogram
   normalized_spectogram = (log_mel_spectrogram+80) / 8.0

   # add an additional dimension to the spectogram for the model input
   return normalized_spectogram.reshape(1,128,173,1)

# FLASK APPLICATION BEGINS HERE
app = Flask(__name__)

# function to upload the audio file/files
@app.route('/')
def home():
   return render_template('upload.html')


@app.route('/upload-audio', methods = ['POST'])
def upload_audio():
   audio_file = request.files['audio-file']
   filename = secure_filename(audio_file.filename)

   # to get the current directory
   current_dir = os.getcwd()

   if audio_file.filename == '':
      return 'No selected file'
   if audio_file:
      #save the audio and background noise files to the current directory
      file_path = current_dir+'/audios/'+filename
      audio_file.save(file_path)

      # preprocess the audio file
      preprocessed_audio = preprocess_audio(file_path)

      # Run the model to detect if the audio is a deepfake or not
      prediction = model.predict(preprocessed_audio)

      # If the model predicts a label of 0, the audio is a real audio
      # If the model predicts a label of 1, the audio is a deepfake
      if prediction[0][0] < 0.5:
        result = 'Real audio'
      else:
        result = 'Deepfake audio'

   return result

#running the app
if __name__ == '__main__':
   app.run(debug = True)