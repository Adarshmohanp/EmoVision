import os
from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import mediapipe as mp
from collections import deque, Counter

app = Flask(__name__)

# Model loading with enhanced compatibility
try:
    model = tf.keras.models.load_model(
        'newerfacemodel_v2.keras',
        compile=False,
        custom_objects=None
    )
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    # Fallback dummy model
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(7, activation='softmax')
    ])
    print("⚠️ Using dummy model")

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
mp_face_detection = mp.solutions.face_detection

def generate_frames(camera_active):
    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5,
        model_selection=1
    ) as face_detection:
        
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        recent_emotions = deque(maxlen=5)
        recent_confidences = deque(maxlen=5)
        
        try:
            while camera_active:
                success, frame = camera.read()
                if not success:
                    break
                
                frame = cv2.flip(frame, 1)
                results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                        w, h = int(bboxC.width * iw), int(bboxC.height * ih)
                        
                        # Face processing logic here
                        # ... [keep your existing face processing code]
                
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        finally:
            camera.release()

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/video_feed/<int:camera_active>')
def video_feed(camera_active):
    return Response(generate_frames(camera_active), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Must match Render's PORT
    app.run(host='0.0.0.0', port=port)