


# 🌱 Cotton Plant Disease Detection using Deep Learning & Raspberry Pi

## 📌 Project Overview

This project uses a **Convolutional Neural Network (CNN)** model trained with **TensorFlow/Keras** to detect whether cotton plants are healthy or diseased. The system is deployed on a **Raspberry Pi** with a camera module, enabling **real-time disease detection in the field**. A **Flask web application** streams live video with prediction overlays, making it easy to monitor plant health remotely.

---

## ⚙️ Tech Stack

* **Languages & Frameworks**: Python, TensorFlow, Keras, Flask
* **Computer Vision**: OpenCV, NumPy
* **Hardware**: Raspberry Pi (with camera, libcamera support)
* **Deployment**: Flask-based web streaming, edge AI on Raspberry Pi

---

## 🚀 Features

* Real-time plant disease detection from live camera feed
* Classification into four categories: *Diseased Cotton Leaf, Diseased Cotton Plant, Fresh Cotton Leaf, Fresh Cotton Plant*
* Lightweight deployment on Raspberry Pi for **edge AI applications**
* Flask web server to stream predictions via browser

---

## 🖼️ Demo

*(Add screenshots or GIFs of your app in action here — predictions shown on live video feed)*

---

## 📂 Project Structure

```
├── cotton disease.h5       # Trained CNN model
├── app.py                  # Flask application with live detection
├── requirements.txt        # Dependencies
├── static/                 # Static files (if any)
└── README.md               # Project documentation
```

---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/cotton-disease-detection.git
cd cotton-disease-detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask App

```bash
python app.py
```

### 4️⃣ Access in Browser

Open:

```
http://<raspberry-pi-ip>:5000/video_feed
```
