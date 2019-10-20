import cv2

cap = cv2.VideoCapture(2)

while True:
    ret,frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(9,9),0)
    # frame = cv2.adaptiveThreshold(frame,frame.max(),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,8)
    _,frame = cv2.threshold(frame,(frame.max() /2),frame.max(),cv2.THRESH_BINARY)
    #contours, hierarchy = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   # cv2.drawContours(frame, contours, -1, (0,255,0), 3)

    cv2.imshow('Frame',frame)

    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()