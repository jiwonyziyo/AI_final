# Expression Recognition
import cv2
import dlib
import numpy as np
from keras.models import load_model

# Face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Return positions of eyes, nose, mouth, etc., for expression recognition
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Expression labels
expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Expression weight model
model = load_model('updated_emotion_model.h5')

# Start video capture
video_capture = cv2.VideoCapture(0)

prev_faces = []

while True:
    # Return ret and frame
    ret, frame = video_capture.read()
    
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    # A scaleFactor closer to 1 improves face detection accuracy; farther reduces it
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    #region Recognize expressions if faces are detected
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the face region
        face_roi = gray[y:y+h, x:x+w]

        # Resize face to match the size used in the expression dataset
        # If the dataset image size and the input face size are different, an error occurs
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / 255.0

        # Analyze the expression using the model
        output = model.predict(face_roi)[0]

        # Get the value of the predicted expression
        expression_index = np.argmax(output)

        # Save the label corresponding to the predicted expression
        expression_label = expression_labels[expression_index]
        # Print the expression label (optional)
        # print(expression_label, end=' ')
        # Display the expression label on the frame
        cv2.putText(frame, expression_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    #endregion
    
    # Display the output
    cv2.imshow('Expression Recognition', frame)

    # Exit when the ESC key is pressed
    key = cv2.waitKey(25)
    if key == 27:
        break

if video_capture.Opened():
    video_capture.release()
cv2.destroyAllWindows()
