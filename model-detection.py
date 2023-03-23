# user should be able to upload audio files
# user should be able to upload background noises

# trained model should be used to classify the deeepfake/real audios

# user should be able to download deepfake audios and real audios seperately

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# function to upload the audio file/files
@app.route('/')
def home():
   return render_template('upload.html')

# to get the current directory
current_dir = os.getcwd()


@app.route('/upload-audio', methods = ['POST'])
def upload_audio():
   audio_file = request.files['audio-file']
   filename = secure_filename(audio_file.filename)
   if audio_file.filename == '':
      return 'No selected file'
   if audio_file:
      #save the audio and background noise files to the current directory
      audio_file.save(current_dir+'/audios/'+filename)
      return 'files uploaded successfully'

#running the app
if __name__ == '__main__':
   app.run(debug = True)