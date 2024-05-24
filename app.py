import os
import zipfile
import tensorflow as tf
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
from PIL import Image  # Ensure Pillow is installed
import numpy as np

app = Flask(__name__, static_folder='static', template_folder='template')
UPLOAD_FOLDER = 'uploads'
FILTERED_FOLDER = 'filtered'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FILTERED_FOLDER'] = FILTERED_FOLDER

# Load your pre-trained model
model = tf.keras.models.load_model('models/model.keras')

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def filter_image(file_path):
    try:
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(416, 416))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

        # Preprocess the image as your model expects
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        predictions = model.predict(img_array)
        score = predictions[0][0]  # Assuming binary classification with sigmoid output
        print(f"Prediction score for {file_path}: {score}")
        
        # Adjust threshold if necessary
        threshold = 0.5
        print(f"Threshold: {threshold}")
        return score < threshold  # Assuming that a score < 0.5 means "no alcohol"
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/filter', methods=['POST'])
def filter_images():
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No selected files"}), 400

    filtered_images = []
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['FILTERED_FOLDER'], exist_ok=True)

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(f"Saved {filename} to {file_path}")

            if filter_image(file_path):  # Apply the model for filtering
                filtered_path = os.path.join(app.config['FILTERED_FOLDER'], filename)
                os.rename(file_path, filtered_path)
                filtered_images.append(filtered_path)
                print(f"Filtered {filename}")
            else:
                os.remove(file_path)  # Delete the image if it doesn't pass the filter
                print(f"Removed {filename}")

    if not filtered_images:
        return jsonify({"error": "No images passed the filtering"}), 400

    zip_filename = "filtered_images.zip"
    zip_path = os.path.join(app.config['FILTERED_FOLDER'], zip_filename)

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for image in filtered_images:
            arcname = os.path.join('filtered_images', os.path.basename(image))
            zipf.write(image, arcname)
            print(f"Added {image} to zip")

    return jsonify({"download_url": f"/download/{zip_filename}"}), 200

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(app.config['FILTERED_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
