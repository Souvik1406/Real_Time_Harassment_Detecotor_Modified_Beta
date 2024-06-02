import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input
from deepface import DeepFace

# Custom CSS for Streamlit app
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    .title {
        color: #4CAF50;
        text-align: center;
        margin-bottom: 20px;
    }
    .instructions {
        font-size: 18px;
        margin-bottom: 20px;
    }
    .video-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #4CAF50;
        color: white;
        text-align: center;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the pre-trained harassment detection model
harassment_model = load_model('harassment_detection_model.h5')

# Function to preprocess frames for VGG19
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    return frame

# Function to get predictions from the harassment model
def get_harassment_prediction(frame):
    preprocessed_frame = preprocess_frame(frame)
    prediction = harassment_model.predict(preprocessed_frame)
    return np.argmax(prediction), np.max(prediction)

# Function to get emotion prediction
def get_emotion_prediction(frame):
    results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    emotion = results[0]['dominant_emotion']
    return emotion

# Function to process the frame and detect perpetrator and victim
def detect_perpetrator_victim(frame):
    results = DeepFace.analyze(frame, actions=['gender'], enforce_detection=False)
    genders = results[0]['gender']
    dominant_gender = max(genders, key=genders.get)
    return "Perpetrator" if dominant_gender == "Man" else "Victim"

# Set up Streamlit app
st.markdown('<h1 class="title">Real-Time Harassment and Emotion Detection</h1>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="instructions">
        <p>This application uses a machine learning model to detect harassment in real-time via webcam feed. It also identifies the perpetrator and victim based on gender and detects the dominant emotion in the scene.</p>
        <ul>
            <li>Click "Start Webcam" to begin the live video feed.</li>
            <li>If harassment is detected, the webcam will automatically stop, and the results will be displayed.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Placeholder for the webcam feed
frame_placeholder = st.empty()

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

# Class names (replace these with your actual class names)
class_names = ['No Harassment', 'Harassment']

# Thresholds for harassment detection
harassment_threshold = 0.8
emotion_perpetrator = "anger"
emotion_victim = "fear"

# Track harassment instances
harassment_confirmed = False

# Start button
if st.button('Start Webcam'):
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture frame. Exiting...")
                break

            # Get predictions
            harassment_prediction, harassment_confidence = get_harassment_prediction(frame)
            harassment_label = class_names[harassment_prediction]

            # Get emotion prediction
            emotion_label = get_emotion_prediction(frame)

            # Check if harassment is detected and emotions match the threshold
            if harassment_label == 'Harassment' and harassment_confidence > harassment_threshold:
                perpetrator_victim_label = detect_perpetrator_victim(frame)
                if perpetrator_victim_label == "Perpetrator" and emotion_label == emotion_perpetrator:
                    harassment_confirmed = True
                elif perpetrator_victim_label == "Victim" and emotion_label == emotion_victim:
                    harassment_confirmed = True
                else:
                    harassment_confirmed = False

            # Combine predictions
            if harassment_confirmed:
                combined_label = f"Harassment Confirmed - {perpetrator_victim_label} with emotion: {emotion_label}"
            else:
                combined_label = f"{harassment_label} with emotion: {emotion_label}"

            # Display the label on the frame
            cv2.putText(frame, combined_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Convert the frame to RGB format for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in the Streamlit app
            frame_placeholder.image(frame, channels="RGB")

            # If harassment is confirmed, stop the video feed
            if harassment_confirmed:
                st.write("Harassment Confirmed! Stopping webcam...")
                break

    finally:
        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()

# Stop button
if st.button('Stop Webcam'):
    if cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()
        st.write("Webcam stopped.")

# Footer
st.markdown(
    """
    <div class="footer">
        Made by : Souvik Roy, RCCIIT College
    </div>
    """,
    unsafe_allow_html=True
)
