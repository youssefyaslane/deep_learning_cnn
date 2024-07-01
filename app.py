from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model_path = r'C:\Users\user\Desktop\Project_diaaa\model\model.keras'
model = load_model(model_path)

def prepare_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    return img_input

def predict(image_path):
    img_input = prepare_image(image_path)
    predictions = model.predict(img_input)
    probability = predictions[0][0]
    if probability > 0.5:
        result = "Good"
    else:
        result = "Defective"
    return result, probability

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Ensure the uploads directory exists
            uploads_dir = 'uploads'
            os.makedirs(uploads_dir, exist_ok=True)
            file_path = os.path.join(uploads_dir, file.filename)
            file.save(file_path)
            result, probability = predict(file_path)
            return render_template('result.html', result=result, probability=probability)
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)
