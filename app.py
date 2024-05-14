from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('models/your_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['file']

    # Preprocess the image and make predictions
    # (Implementation depends on your model's input requirements)

    # Example: preprocess the image and predict using the model
    # prediction = model.predict(preprocess_image(file))

    # Return the prediction
    return jsonify({'prediction': 'alcoholic' if prediction > 0.5 else 'non-alcoholic'})

if __name__ == '__main__':
    app.run(debug=True)
