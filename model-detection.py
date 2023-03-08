# user should be able to upload audio files
# user should be able to upload background noises

# trained model should be used to classify the deeepfake/real audios

# user should be able to download deepfake audios and real audios seperately

from flask import Flask
app = Flask(__name__)

@app.route('/')
def upload_audios():
  'upload audio files'

def upload_background_noises():
  'upload background noises'

if __name__ == '__main__':
   app.run()