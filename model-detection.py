# user should be able to upload audio files
# user should be able to upload background noises

# trained model should be used to classify the deeepfake/real audios

# user should be able to download deepfake audios and real audios seperately

from flask import Flask, render_template, request
from werkzeug import secure_filename
app = Flask(__name__)

@app.route('/upload')
def upload_file():
   return render_template('upload.html')

@app.route('/audios-uploader', methods = ['GET', 'POST'])
def upload_audios():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'audio files uploaded successfully'

@app.route('/background-noises-uploader', methods = ['GET', 'POST'])
def upload_background_noises():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'background noises files uploaded successfully'

if __name__ == '__main__':
   app.run(debug = True)