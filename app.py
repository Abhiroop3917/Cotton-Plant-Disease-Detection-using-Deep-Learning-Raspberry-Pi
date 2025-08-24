import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, Response
import subprocess

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('cotton disease.h5')

# Define class labels
class_labels = [
    "diseased cotton leaf",
    "diseased cotton plant",
    "fresh cotton leaf",
    "fresh cotton plant"
]

# Function to check if a plant is in the frame (green detection)
def is_plant_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])  
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask)
    return green_pixels > 500  

# Function to preprocess the image for the model
def preprocess_image(frame, target_size=(224, 224)):
    resized = cv2.resize(frame, target_size)
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

# Function to capture an image using libcamera
def capture_image():
    subprocess.run([
        "libcamera-still",
        "--output", "captured_image.jpg",
        "--width", "640",
        "--height", "480",
        "--timeout", "1"
    ])
    image = cv2.imread('captured_image.jpg')
    return image

# Function to generate frames for live streaming
def generate_frames():
    while True:
        frame = capture_image()
        if frame is None:
            continue

        if is_plant_frame(frame):
            preprocessed_frame = preprocess_image(frame)
            predictions = model.predict(preprocessed_frame)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)
            label = f"{class_labels[predicted_class]} ({confidence:.2f})"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        else:
            label = "No plant detected"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
