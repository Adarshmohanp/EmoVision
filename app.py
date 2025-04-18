import os
from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque, Counter

app = Flask(__name__)

# Load model - use ONLY .h5 format
try:
    model = load_model('model.h5')
    print("✅ Model loaded successfully")
except:
    print("⚠️ Using dummy model")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    model = Sequential([Flatten(), Dense(7, activation='softmax')])

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
        cap = cv2.VideoCapture(0)
        while camera_active:
            ret, frame = cap.read()
            if not ret: break
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)