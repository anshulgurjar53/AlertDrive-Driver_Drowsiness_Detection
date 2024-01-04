import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
alarm_sound = mixer.Sound('buzzer.wav')

face_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
left_eye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

labels = ['Closed', 'Open']

eye_model = load_model('models/trmodel.h5')
current_path = os.getcwd()
video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
counter = 0
score = 0
thickness = 2
right_eye_predictions = [99]
left_eye_predictions = [99]

while True:
    ret, frame = video_capture.read()
    height, width = frame.shape[:2] 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_frame, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eyes = left_eye_cascade.detectMultiScale(gray_frame)
    right_eyes = right_eye_cascade.detectMultiScale(gray_frame)

    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eyes:
        right_eye_roi = frame[y:y+h, x:x+w]
        counter += 1
        right_eye_roi = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)
        right_eye_roi = cv2.resize(right_eye_roi, (24, 24))
        right_eye_roi = right_eye_roi / 255
        right_eye_roi = right_eye_roi.reshape(24, 24, -1)
        right_eye_roi = np.expand_dims(right_eye_roi, axis=0)
        right_eye_predictions = np.argmax(eye_model.predict(right_eye_roi), axis=-1)
        break

    for (x, y, w, h) in left_eyes:
        left_eye_roi = frame[y:y+h, x:x+w]
        counter += 1
        left_eye_roi = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)  
        left_eye_roi = cv2.resize(left_eye_roi, (24, 24))
        left_eye_roi = left_eye_roi / 255
        left_eye_roi = left_eye_roi.reshape(24, 24, -1)
        left_eye_roi = np.expand_dims(left_eye_roi, axis=0)
        left_eye_predictions = np.argmax(eye_model.predict(left_eye_roi), axis=-1)
        break

    if right_eye_predictions[0] == 0 and left_eye_predictions[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    #if score < 0:
    #    score = 0   
    #cv2.putText(frame, 'Score:' + str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 30:
        cv2.imwrite(os.path.join(current_path, 'image.jpg'), frame)
        try:
            alarm_sound.play()
        except:
            pass
        thickness = min(thickness + 2, 16) if thickness < 16 else max(thickness - 2, 2)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thickness) 

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
