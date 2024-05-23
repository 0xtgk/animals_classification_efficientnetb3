from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('eff.h5')

# Define a function to preprocess the image
def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

# Define the index route
@app.route('/')
def index():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file from the request
    file = request.files['file']
    
    # Save the file to disk
    img_path = 'static/images/' + file.filename
    file.save(img_path)
    
    # Preprocess the image
    img = preprocess_img(img_path)
    
    # Make a prediction
    pred = model.predict(img)
    pred_class = np.argmax(pred)
    
    # Define a dictionary of class names
    class_names = {
        0: 'dog',
        1: 'horse',
        2: 'elephant',
        3: 'butterfly',
        4: 'cock',
        5: 'cat',
        6: 'cow',
        7: 'sheep',
        8: 'spider',
        9: 'squirrel'
    }
    
    # Get the predicted class name and the uploaded file name
    pred_name = class_names[pred_class]
    file_name = file.filename
    
    # Return the predicted class name and the uploaded file name to the user
    return render_template('predict.html', pred_name=pred_name, file_name=file_name)

if __name__ == '__main__':
    app.run(debug=True)
