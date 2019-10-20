import cv2
from datetime import datetime

cap = cv2.VideoCapture(2)
orb = cv2.ORB_create()


Target = cv2.imread('DATA/socket.png')
Target = cv2.cvtColor(Target,cv2.COLOR_BGR2GRAY)
Target = cv2.adaptiveThreshold(Target,Target.max(),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)
#ret,Target = cv2.threshold(Target,(Target.max() /2),Target.max(),cv2.THRESH_BINARY)

kp1, des1 = orb.detectAndCompute(Target, None)

while True:

    ret,frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.adaptiveThreshold(frame,frame.max(),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)
    #ret,frame = cv2.threshold(frame,(frame.max() /2),frame.max(),cv2.THRESH_BINARY)

    kp2, des2 = orb.detectAndCompute(frame, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    bf_matches = cv2.drawMatches(frame, kp2, Target, kp1, matches[:25], None, flags=2)

    cv2.imshow('OBV',bf_matches)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()