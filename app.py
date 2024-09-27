import face_recognition
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import base64
import os  # Add this line to import the os module

app = Flask(__name__)

# Load known faces and their encodings
known_face_encodings = []
known_face_names = []

def load_known_faces():
    global known_face_encodings, known_face_names
    # Load images from the known_faces directory
    for filename in os.listdir('known_faces'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = face_recognition.load_image_file(f'known_faces/{filename}')
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(filename.split('.')[0])  # Use filename as the name (without extension)

load_known_faces()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    image_data = data['image']
    
    # Decode the image
    image_data = image_data.split(',')[1]
    image_data = base64.b64decode(image_data)
    
    # Convert the image data to a numpy array
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find all the faces in the image
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    recognized_faces = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the closest distance
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        recognized_faces.append({
            "name": name,
            "location": face_location  # face_location is a tuple (top, right, bottom, left)
        })

    return jsonify(recognized_faces)

if __name__ == '__main__':
    app.run(debug=True)
