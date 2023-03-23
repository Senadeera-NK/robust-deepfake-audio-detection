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


@app.route('/audios-uploader', methods = ['POST'])
def upload_audios():
    audio_file = request.files['file']
    # Save the audio file to a folder
    audio_file.save('<path_to_folder>/audio_file.wav')
    return 'Audio file uploaded successfully'

@app.route('/background-noises-uploader', methods = ['POST'])
def upload_background_noises():
    bg_noise_file = request.files['file']
    # Save the background noise file to a folder
    bg_noise_file.save('<path_to_folder>/bg_noise_file.wav')
    return 'Background noise file uploaded successfully'

#running the app
if __name__ == '__main__':
   app.run(debug = True)