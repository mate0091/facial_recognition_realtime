import numpy as np
import cv2
import os
import pickle

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("trainer.yml")

labels = {}
temp_labels = {}
with open("labels.dat", 'rb') as f:
    temp_labels = pickle.load(f)
    labels = {v: k for k, v in temp_labels.items()}

while(True):
    ret, frame = cap.read()

    #conv to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    if len(faces) != 0:
        x, y, w, h = faces[0]

        roi = gray[y: y + h, x: x + w]

        roi = cv2.resize(roi, (512, 512), interpolation=cv2.INTER_LINEAR)

        ind, conf = recognizer.predict(roi)

        if 45 <= conf:
            #print(labels[ind])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[ind]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y - 25), font, 1, color, stroke, cv2.LINE_AA)
            cv2.putText(frame, str(int(conf)), (x + 280, y - 25), font, 1, color, stroke, cv2.LINE_AA)

        color = (255, 255, 255)
        stroke = 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

    cv2.imshow('webcam', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
