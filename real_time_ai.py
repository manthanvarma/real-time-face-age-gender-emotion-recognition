import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the models
AGE_PROTO = "models/deploy_age.prototxt"
AGE_MODEL = "models/age_net.caffemodel"
GENDER_PROTO = "models/deploy_gender.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"
EMOTION_MODEL = "models/emotion_model.h5"

age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
emotion_model = load_model(EMOTION_MODEL, compile=False)

# Labels
AGE_CLASSES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', 
               '(38-43)', '(48-53)', '(60-100)']
GENDER_CLASSES = ['Male', 'Female']
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img_color = frame[y:y+h, x:x+w].copy()
        face_img_gray = gray[y:y+h, x:x+w].copy()

        # ---------- Emotion Detection ----------
        roi_gray_resized = cv2.resize(face_img_gray, (64, 64))  # Resize to 64x64
        roi = roi_gray_resized.astype("float") / 255.0
        roi = np.reshape(roi, (1, 64, 64, 1))  # Now the shape will be (1, 64, 64, 1)
        emotion_preds = emotion_model.predict(roi, verbose=0)
        emotion = EMOTION_CLASSES[np.argmax(emotion_preds)]

        # ---------- Age & Gender Detection ----------
        blob = cv2.dnn.blobFromImage(face_img_color, 1.0, (227, 227), 
                                     (78.426, 87.769, 114.896), swapRB=False)

        # Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_CLASSES[gender_preds[0].argmax()]

        # Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_CLASSES[age_preds[0].argmax()]

        # Combine info
        label = f'{gender}, {age}, {emotion}'
        
        # Draw the face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the combined label
        cv2.putText(frame, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # Show the final image with face and predictions
    cv2.imshow("Real-time Face | Age | Gender | Emotion Detection", frame)

    # Exit the loop on pressing ESC
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
