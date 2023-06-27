from flask import Flask, render_template, request, send_from_directory
from flaskapp import app
from flaskapp.facerec import detect_faces, knn
import os
import base64
import time

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

        start_time = time.time()
        results, names, similarity_scores = knn(img, numeric_value)
        zipped_data = zip(results, names, similarity_scores)
        return render_template('results.html', data=zipped_data, original_image=img_base64, time=(time.time()-start_time))