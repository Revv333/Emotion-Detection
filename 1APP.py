import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained Keras model
emotion_model = load_model('model.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

st.title("Real-Time Emotion Detection")
st.subheader("This tool detects faces and identifies emotions like happiness, sadness, anger, and more in real-time using state-of-the-art AI technology.")

st.sidebar.title("Options")
run = st.sidebar.checkbox('Run')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame_count = 0

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture video feed.")
        break
    
    # Increment the frame counter
    frame_count += 1

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract the region of interest (ROI) and preprocess it for emotion prediction
        roi_gray_frame = gray_frame[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        
        # Predict emotion using your custom model
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        
        # Get the corresponding emotion label from the dictionary
        predicted_emotion = emotion_dict[maxindex]
        
        # Display the predicted emotion on the video frame
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Resize frame for Streamlit display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    # Display the frame count on Streamlit
    st.sidebar.write(f"Frames captured: {frame_count}")

    # Stop the loop if "Run" is unchecked
    if not run:
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

