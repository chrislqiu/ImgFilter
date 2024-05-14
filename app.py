from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the pre-trained model
model = tf.keras.models.load_model('models/your_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (416, 416))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the image
            image = preprocess_image(file_path)
            predictions = model.predict(image)
            # Filter logic based on predictions
            if 'alcoholic_drink' in predictions:
                return "Alcoholic drink detected. Image filtered out."
            else:
                return "No alcoholic drinks detected. Image is safe."

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
