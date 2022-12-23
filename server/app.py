import os
import io
import json
import numpy as np
# from utils import *
from PIL import Image
from flask import Flask,request
from flask import Flask, render_template
from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileAllowed, FileRequired
from werkzeug.utils import secure_filename
import librosa
from keras.models import load_model

SECRET_KEY = os.urandom(32)

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
Bootstrap(app)

model = load_model("basemodel.h5")

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)
    result = np.expand_dims(result, axis=1)
    result = np.expand_dims(result, axis=0)
    return result

# with Flask-WTF, each web form is represented by a class
# "NameForm" can change; "(FlaskForm)" cannot
# see the route for "/" and "index.html" to see how this is used
class NameForm(FlaskForm):
    file = FileField("Submit the audio below for sentiment analysis:")
    submit = SubmitField('Submit')

# unet_model = DynamicUNet([16,32,64,128,256])
# unet_classifier = BrainTumorClassifier(unet_model,'cpu')
# unet_classifier.restore_model(os.path.join('./',f"brain_tumor_segmentor.pt"))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# @app.route('/')
# def index():
#     return 'Web App with Python Flask! Hey'
@app.route('/', methods=['GET', 'POST'])
def home():
    form = NameForm(meta={'csrf': False})
    if form.validate_on_submit():
        filename = secure_filename(form.file.data.filename)
        print(filename)
        form.file.data.save('uploads/' + filename)
        features = get_features('uploads/' + filename)
        print(features.shape)
        x = model.predict(features)
        d = np.argmax(x)
        result = { "0": "angry", 
 "1": "disgust", 
 "2": "fear", 
 "3": "happy", 
 "4": "neutral", 
 "5": "sad", 
 "6": "surprise" }
        alld = [[result[str(i)], j ]for i,j in enumerate(x)]
        test = result[str(d)]
        return render_template('index.html', form=form, message = test, alld = alld)
    
    return render_template('index.html', form=form)

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         file = request.files['file']
#         file = np.array(Image.open(io.BytesIO(file.read())))
#         data =  {"image": file }
#         output= unet_classifier.predict(data,  0.65)
#         return json.dumps({'mask':output}, cls=NumpyEncoder)

if __name__ == '__main__':
    app.run()