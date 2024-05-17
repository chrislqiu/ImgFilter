import warnings
from flask import Flask, request, jsonify, send_from_directory, render_template
import tensorflow as tf
import os

warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

app = Flask(__name__, static_folder='static', template_folder='template')

# Load the model
model = tf.keras.models.load_model('models/model.keras')

# Serve the HTML file
@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure an image file is provided
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Preprocess the image
    img = tf.keras.preprocessing.image.load_img(file, target_size=(416, 416))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Predict using the model
    predictions = model.predict(img_array)
    score = predictions[0][0]

    # Return the result
    return jsonify({"score": float(score)})

if __name__ == '__main__':
    app.run(debug=True)
