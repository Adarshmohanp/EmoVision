import os
from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque, Counter

app = Flask(__name__)

# Simplified model loading - use ONLY .h5 format for compatibility
try:
    model = load_model('model.h5', compile=False)
    print("✅ Model loaded successfully")
except:
    print("⚠️ Using dummy model")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(7, activation='softmax')
    ])

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return "OK", 200

@app.route('/video_feed/<int:camera_active>')
def video_feed(camera_active):
    def generate():
        with mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5
        ) as face_detection:
            
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            recent_emotions = deque(maxlen=5)
            
            try:
                while camera_active:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb)
                    
                    if results.detections:
                        # Your face detection and emotion logic here
                        # [Keep your existing processing code]
                        
                        ret, buffer = cv2.imencode('.jpg', frame)
                        yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            finally:
                cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, threaded=True)