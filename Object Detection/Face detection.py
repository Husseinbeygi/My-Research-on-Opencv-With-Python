### Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
# Add Cascade Classifier
face_cass = cv2.CascadeClassifier('DATA/haarcascade/haarcascade_frontalface_default.xml')

## Face Detection Function
def face_detection(img):
    # Make a Copy
    face_img = img.copy()
    # Use the detect Algoritm for Face Detection 
    # You Can Adjust scaleFactor & minNeighbors 
    # For Better Detection
    face_rects = face_cass.detectMultiScale(face_img,scaleFactor=1.1,minNeighbors=3)
    # Darw a rectangle around the face
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),8)
    return face_img


# Start the video Capture
cap = cv2.VideoCapture(0)

while True:
    # Get the frame
    ret,frame = cap.read(0)
    # Detect Face
    frame = face_detection(frame)
    #Show the Image
    cv2.imshow('TEST',frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()