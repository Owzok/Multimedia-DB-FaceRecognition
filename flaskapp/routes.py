from flask import Flask, render_template, request, send_from_directory, Response, redirect
from flaskapp import app
from flaskapp.facerec import detect_faces, faiss_knn, rindex_knn, unindexed_knn
import os                                       # send images that aren't in static       
import base64                                   # encode uploaded images
import cv2
import face_recognition
import pickle
import numpy as np
import time


@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_pic():
    if 'file' not in request.files:
        return 'No file uploaded.'
    
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return 'No file selected.'

        img = file.read()

        method = request.form.get('method') # one of 
                                            # 'faiss', 'unindexed', 'rtree'

        try:
            numeric_value = int(request.form.get('numeric', 8))
        except ValueError:
            numeric_value = 8

        start_time = time.perf_counter()
        results, names, similarity_scores, best_result = (None,None,None,None)
        if (method == 'faiss'):
            results, names, similarity_scores, best_result = faiss_knn(img, numeric_value)
        elif(method == 'rtree'):
            results, names, similarity_scores, best_result = rindex_knn(img, numeric_value)
        else:
            results, names, similarity_scores, best_result = unindexed_knn(img, numeric_value)
        total_time = time.perf_counter() - start_time
        zipped_data = zip(results, names, similarity_scores)

        img_base64 = base64.b64encode(img).decode('utf-8')
        return render_template('results.html', data=zipped_data, best_result_image=best_result, original_image=img_base64, time=(total_time))

# Retrieve an image outside the static folder for the frontend
@app.route('/dataset/<path:filename>')
def serve_image(filename):
    dataset_dir = os.path.join(os.getcwd(), 'dataset')
    return send_from_directory(dataset_dir, filename)


def detect_faces_realtime():
    # Cargar los datos de codificación de rostros almacenados en un archivo llamado "encodings.pickle"
    data = pickle.loads(open("encodings.pickle", "rb").read())
    global video_capture
    # Activar la cámara para capturar fotos
    global video_capture
    video_capture = cv2.VideoCapture(0)

    current_name = None  # Nombre de la imagen actualmente mostrada
    current_distance = None  # Distancia euclidiana actual
    photo_window = None  # Ventana de la foto actual

    start_time = time.time()
    elapsed_time = 0

    show_photo = False  # Variable para controlar si se muestra la foto

    while elapsed_time < 10:  # Ejecutar durante 10 segundos
        # Leer un fotograma del video
        ret, frame = video_capture.read()

        # Voltear el fotograma horizontalmente
        frame = cv2.flip(frame, 1)

        # Convertir el fotograma de BGR a RGB (OpenCV utiliza el formato BGR por defecto)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar las ubicaciones de los rostros en el fotograma
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")

        # Si se detecta al menos un rostro, realizar el reconocimiento
        if face_locations:
            # Calcular las codificaciones faciales de los rostros detectados
            encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Inicializar una lista para almacenar los nombres de los rostros detectados
            names = []

            for encoding in encodings:
                # Comparar las codificaciones faciales con los datos cargados previamente
                matches = face_recognition.compare_faces(data["encodings"], encoding)
                name = "Unknown"
                distance = None

                # Si se encuentra una coincidencia, asignar el nombre y la distancia correspondiente
                if True in matches:
                    matched_indexes = [i for (i, b) in enumerate(matches) if b]
                    distances = face_recognition.face_distance(data["encodings"], encoding)
                    min_distance_index = np.argmin(distances)
                    name = data["names"][min_distance_index]
                    distance = distances[min_distance_index]

                names.append(name)

                # Mostrar la distancia euclidiana en la ventana del video
                cv2.putText(frame, f"Distancia: {distance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Actualizar la foto y la distancia mínima actual
                if current_name is None or (distance is not None and distance < current_distance):
                    current_name = name
                    current_distance = distance

            # Dibujar los recuadros y nombres de los rostros detectados en el fotograma
            for (top, right, bottom, left), name in zip(face_locations, names):
                # Calcular las coordenadas para la posición arriba a la derecha del recuadro
                x = left
                y = top

                # Dibujar el recuadro y el nombre en el fotograma
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Mostrar la foto de la persona si se encuentra una coincidencia
                results = []
                testdir = "./dataset"
                for f in os.listdir(testdir):
                    results.append(f)

                if current_name in results:
                    if photo_window is None or current_distance < distance:
                        # Si es la primera vez que se muestra la foto o si la distancia es menor que la foto actualmente mostrada
                        photo_path = os.path.join(testdir, current_name, current_name+'_0001.jpg')  # Ruta de la foto de la persona

                        # Cargar la nueva foto
                        new_photo = cv2.imread(photo_path)

                        # Mostrar la nueva foto en una ventana separada
                        if photo_window is None:
                            photo_window = "Foto de " + current_name
                            cv2.imshow(photo_window, new_photo)
                        elif current_name != photo_window[8:]:  # Comprobar si la foto mostrada es diferente a la nueva foto
                            cv2.destroyWindow(photo_window)
                            photo_window = "Foto de " + current_name
                            cv2.imshow(photo_window, new_photo)

                            # Activar la visualización de la foto
                            show_photo = True

        # Mostrar el fotograma con los resultados del reconocimiento facial
        cv2.imshow('Video', frame)

        # Mostrar el nombre de la foto con la mejor distancia y su distancia mínima actual debajo del video
        if current_name is not None and current_distance is not None:
            text = f"Foto más parecida: {current_name} - Distancia mínima: {current_distance:.2f}"
            cv2.putText(frame, text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Detener la ejecución si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elapsed_time = time.time() - start_time

    # Cerrar la ventana del video
    cv2.destroyWindow('Video')

    # Mostrar la mejor distancia y el nombre de la foto en la consola
    if current_name is not None and current_distance is not None:
        print(f"Mejor distancia: {current_distance:.2f}")
        print(f"Nombre de la foto: {current_name}")

        # Mostrar la foto obtenida después de los 10 segundos
        photo_path = os.path.join(testdir, current_name, current_name + '_0001.jpg')  # Ruta de la foto de la persona
        new_photo = cv2.imread(photo_path)
        cv2.imshow("Mejor Foto", new_photo)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Cerrar la ventana de la foto mostrada si no se presiona una tecla
        if not show_photo:
            cv2.destroyWindow("Mejor Foto")


@app.route('/camerart')
def index():
    return render_template('newindex.html', variable = '0')

@app.route('/start',methods=['POST'])
def start():
    return render_template('newindex.html')

@app.route('/stop',methods=['POST'])
def stop():
    if video_capture.isOpened():
        video_capture.release()
    return render_template('stop.html')

@app.route('/video_capture')
def video_capture():
    return Response(detect_faces_realtime(), mimetype='multipart/x-mixed-replace; boundary=frame')

