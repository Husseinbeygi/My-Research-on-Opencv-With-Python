import cv2
# CALL BACK FUNCTION
def draw_rectangle(event,x,y,flags,param):

    global pt1,pt2,topleft_clicked,bottomright_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        if topleft_clicked == True and bottomright_clicked == True:
            pt1 = 0
            pt2 = 0
            topleft_clicked = False
            bottomright_clicked = False


        if topleft_clicked == False:
            pt1 = (x,y)
            topleft_clicked = True
        
        elif bottomright_clicked == False:
            pt2 = (x,y)
            bottomright_clicked = True


# VARIABLES
pt1 = (0,0)
pt2 = (0,0)
topleft_clicked = False
bottomright_clicked = False
# CONNECT TO CALLBACK
cap = cv2.VideoCapture(0)

cv2.namedWindow('TEST')
cv2.setMouseCallback('TEST',draw_rectangle)

while True:
    ret,frame = cap.read()
    # DRAW OPRATION
    if topleft_clicked:
        cv2.circle(frame,pt1,3,(0,0,255),3)

    if topleft_clicked and bottomright_clicked:
        cv2.rectangle(frame,pt1,pt2,(0,0,255),3)    
    
    cv2.imshow('TEST',frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()