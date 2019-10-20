import cv2
from datetime import datetime


cap = cv2.VideoCapture(2)

while True:

    ret,frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # ret,thresh = cv2.threshold(frame,(frame.max() /2),frame.max(),cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(frame,frame.max(),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)
    cv2.imshow('OBV',th2)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()