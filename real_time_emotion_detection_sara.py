# Real-time emotion detection with additional information
import cv2
import numpy as np
from keras.models import load_model

# Load the model and other resources
model = load_model('expression_model_sara.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize counters and performance metrics
emotion_counter = {label: 0 for label in expression_labels}
model_accuracy = 55.71  # Example accuracy from training
fps_start_time = 0
fps = 0

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / 255.0

        output = model.predict(face_roi)[0]
        expression_index = np.argmax(output)
        expression_label = expression_labels[expression_index]
        confidence = output[expression_index] * 100

        emotion_counter[expression_label] += 1
        cv2.putText(frame, f"{expression_label}: {confidence:.2f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display emotion counters
    y_offset = 50
    for emotion, count in emotion_counter.items():
        cv2.putText(frame, f"{emotion}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 20

    # Display model accuracy
    cv2.putText(frame, f"Model Accuracy: {model_accuracy:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Calculate and display FPS
    fps_end_time = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (fps_end_time - fps_start_time + 1e-6)
    fps_start_time = fps_end_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Real-Time Emotion Detection', frame)

    key = cv2.waitKey(25)
    if key == 27:  # ESC key to exit
        break

video_capture.release()
cv2.destroyAllWindows()
