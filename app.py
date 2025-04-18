import os
from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque, Counter

app = Flask(__name__)

# Load model with error handling and compatibility fixes
try:
    # Try loading with Keras 3.0+ compatibility settings
    model = tf.keras.models.load_model('newerfacemodel.keras', compile=False)
    print("✅ Model loaded successfully in .keras format")
except Exception as e:
    print(f"⚠️ Error loading .keras model: {e}")
    try:
        # Fallback to H5 format if available
        model = tf.keras.models.load_model('newerfacemodel.h5', compile=False)
        print("✅ Model loaded successfully in .h5 format")
    except Exception as e:
        print(f"⚠️ Error loading .h5 model: {e}")
        # Create minimal dummy model if both fail
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(7, activation='softmax')
        ])
        print("⚠️ Created dummy model for testing - real emotions won't be detected")

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# MediaPipe setup (module reference only)
mp_face_detection = mp.solutions.face_detection

def generate_frames(camera_active):
    # Initialize MediaPipe for this session
    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5,
        model_selection=1  # 0=short-range, 1=full-range
    ) as face_detection:
        
        # Camera setup
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Emotion smoothing buffers
        recent_emotions = deque(maxlen=5)
        recent_confidences = deque(maxlen=5)
        
        try:
            while camera_active:
                success, frame = camera.read()
                if not success:
                    break
                
                # Mirror the frame
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                results = face_detection.process(rgb_frame)
                ih, iw, _ = frame.shape

                if results.detections:
                    for detection in results.detections:
                        # Get face bounding box
                        bboxC = detection.location_data.relative_bounding_box
                        x = int(bboxC.xmin * iw)
                        y = int(bboxC.ymin * ih)
                        w = int(bboxC.width * iw)
                        h = int(bboxC.height * ih)

                        # Add 10% padding
                        pad_x, pad_y = int(0.1 * w), int(0.1 * h)
                        x1, y1 = max(0, x-pad_x), max(0, y-pad_y)
                        x2, y2 = min(iw, x+w+pad_x), min(ih, y+h+pad_y)
                        face_roi = frame[y1:y2, x1:x2]

                        if face_roi.size == 0:
                            continue

                        try:
                            # Preprocess face for emotion detection
                            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                            resized_face = cv2.resize(gray_face, (48, 48))
                            normalized_face = resized_face / 255.0
                            input_face = np.expand_dims(normalized_face, axis=(0, -1))

                            # Predict emotion
                            emotion_predictions = model.predict(input_face, verbose=0)
                            emotion_index = np.argmax(emotion_predictions)
                            emotion = emotion_labels[emotion_index]
                            confidence = emotion_predictions[0][emotion_index] * 100

                            # Update smoothing buffers
                            recent_emotions.append(emotion)
                            recent_confidences.append(confidence)

                            # Get smoothed result
                            common_emotion = Counter(recent_emotions).most_common(1)[0][0]
                            avg_confidence = np.mean([
                                conf for emo, conf 
                                in zip(recent_emotions, recent_confidences) 
                                if emo == common_emotion
                            ])

                            # Draw UI elements
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            y_offset = y1 - 10 if y1 - 10 > 10 else y2 + 30
                            cv2.putText(
                                frame, f"{common_emotion}: {avg_confidence:.1f}%",
                                (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2
                            )

                        except Exception as e:
                            print(f"⚠️ Face processing error: {e}")
                
                # Encode frame for streaming
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        finally:
            camera.release()
            print("🚀 Camera resources released")

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/video_feed/<int:camera_active>')
def video_feed(camera_active):
    return Response(
        generate_frames(camera_active),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if $PORT not set
    app.run(host='0.0.0.0', port=port)