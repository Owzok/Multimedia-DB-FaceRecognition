from flask import Flask, render_template, request, send_from_directory
from flaskapp import app
from flaskapp.facerec import detect_faces, knn
import os
import base64

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/dataset/<path:filename>')
def serve_image(filename):
    dataset_dir = os.path.join(os.getcwd(), 'dataset')
    return send_from_directory(dataset_dir, filename)

@app.route('/upload', methods=['GET', 'POST'])
def upload_pic():
    if 'file' not in request.files:
        return 'No file uploaded.'
    
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return 'No file selected.'

        img = file.read()
        img_base64 = base64.b64encode(img).decode('utf-8')

        try:
            numeric_value = int(request.form.get('numeric', 5))
        except ValueError:
            numeric_value = 5

        results = knn(img, numeric_value)
        return render_template('results.html', images=results, original_image=img_base64)