from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from gtts import gTTS
from googletrans import Translator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['AUDIO_FOLDER'] = 'static'

# Load model
model = tf.keras.models.load_model('traffic_classifier.h5')

# Class labels
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing vehicles over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles',
    16: 'Vehicles > 3.5 tons prohibited', 17: 'No entry', 18: 'General caution',
    19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians',
    28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Beware of ice/snow',
    31: 'Wild animals crossing', 32: 'End speed + passing limits',
    33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only',
    36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right',
    39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}

# Language options
language_options = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German",
    "it": "Italian", "zh-cn": "Chinese", "ja": "Japanese", "ko": "Korean",
    "ru": "Russian", "pt": "Portuguese", "ar": "Arabic", "nl": "Dutch", "hi": "Hindi"
}

def predict_image(image_path):
    img = Image.open(image_path)
    img = img.resize((30, 30))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    prediction = model.predict(input_img)
    predicted_class = np.argmax(prediction)
    return classes[predicted_class]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_lang = request.form['language']
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Predict the class
        label = predict_image(file_path)

        # Translate & convert to speech
        translator = Translator()
        translated_text = translator.translate(f"It is a {label}.", dest=selected_lang).text
        tts = gTTS(text=translated_text, lang=selected_lang)
        audio_path = os.path.join(app.config['AUDIO_FOLDER'], 'output.mp3')
        tts.save(audio_path)

        return render_template('index.html', label=label, image_path=file_path,
                               audio_path=audio_path, selected_language=language_options[selected_lang],
                               languages=language_options)
    
    return render_template('index.html', label=None, languages=language_options)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
