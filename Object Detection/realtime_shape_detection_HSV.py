import cv2
import numpy as np



def nothing(x):
    # any operation
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("LH", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("LS", "Trackbars", 66, 255, nothing)
cv2.createTrackbar("LV", "Trackbars", 134, 255, nothing)
cv2.createTrackbar("UH", "Trackbars", 180, 180, nothing)
cv2.createTrackbar("US", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("UV", "Trackbars", 243, 255, nothing)

font = cv2.FONT_HERSHEY_COMPLEX

while True:
    # Read the Frame
    _, frame = cap.read()
    O_frame = frame.copy()
    # Add Gaussian Blur
    frame = cv2.GaussianBlur(frame,(9,9),0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Trackbars")
    l_s = cv2.getTrackbarPos("LS", "Trackbars")
    l_v = cv2.getTrackbarPos("LV", "Trackbars")
    u_h = cv2.getTrackbarPos("UH", "Trackbars")
    u_s = cv2.getTrackbarPos("US", "Trackbars")
    u_v = cv2.getTrackbarPos("UV", "Trackbars")

    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)

    image,contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # try catch error
    
    try: hierarchy = hierarchy[0]
    except: hierarchy = []

    # get the shape of the image
    height, width, _ = frame.shape

    # define the x,y and width and height from center to edge ----> the edge is 0
    min_x, min_y = width, height
    max_x = max_y = 0

    # Find the bigest Contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    if contour_sizes:
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    else:
        fs = frame.shape
        cv2.putText(O_frame,"No Object Detected",(fs[0]//3,fs[1]//2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),6)
    # computes the bounding box for the contour, and draws it on the frame, 
    # Here, bounding rectangle is drawn with minimum area, so it considers the rotation also.
    # The function used is cv2.minAreaRect(). It returns a Box2D structure which contains following detals 
    # (center (x,y), (width, height), angle of rotation ). But to draw this rectangle, we need 4 corners of the rectangle.
    # It is obtained by the function cv2.boxPoints()
    rect = cv2.minAreaRect(biggest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Find the Angle of object
    angle = round(rect[2],2) 
    X = rect[0]
    X_x = int(X[0])
    X_y = int(X[1])

    ### DRAW
    cv2.drawContours(O_frame,[box],0,(0,255,0),2)

    # Put Cricle Center of Frame
    cv2.circle(O_frame,(X_x,X_y),5,(255,0,0),-1)

    # print Angle , X, Y
    cv2.putText(O_frame,"Angle = " + str(angle),(40,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
    cv2.putText(O_frame,"X = " + str(X_x),(40,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
    cv2.putText(O_frame,"Y = " + str(X_y),(40,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)

    # Show the Frame
    cv2.imshow("What Oprator See", O_frame)

    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()