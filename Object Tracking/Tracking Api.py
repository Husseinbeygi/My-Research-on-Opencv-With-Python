import cv2
from datetime import datetime

def ask_for_tracker():
    print("0 --------> BOOSTING: ")
    print("1 --------> MIL: ")
    print("2 --------> KCF: ")
    print("3 --------> TLD: ")
    print("4 --------> MEDIANFLOW: ")
    choice = input("Please select your tracker ? ")

    if choice == '0':
        tracker  = cv2.TrackerBoosting_create()
    if choice == '1':
        tracker  = cv2.TrackerMIL_create()
    if choice == '2':
        tracker  = cv2.TrackerKCF_create()
    if choice == '3':
        tracker  = cv2.TrackerTLD_create()
    if choice == '4':
        tracker  = cv2.TrackerMedianFlow_create()

    return tracker

#tracker  = cv2.TrackerBoosting_create()
tracker = ask_for_tracker()
tracker_name = str(tracker).split()[0][1:]

cap = cv2.VideoCapture(2)

ret, frame = cap.read()

roi = cv2.selectROI(frame,False)

ret = tracker.init(frame,roi)

while True:

    ret,frame = cap.read()

    success, roi = tracker.update(frame)

    (x,y,w,h) = tuple(map(int,roi))

    if success:
        pt1 = (x,y)
        pt2 = (x+w,y+h)
        cpt1 = (x + (x+w)) // 2
        cpt2 = (y + (y+h)) // 2
        cv2.rectangle(frame,pt1,pt2,(0,255,0),3)
        cv2.putText(frame,'X = ' + str(cpt1) + 'Y = ' + str(cpt2),(200,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
        cv2.circle(frame,(cpt1,cpt2),1,(255,0,0),1)

    else:
        cv2.putText(frame,"Failure to Detect Tracking!!",(100,200),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)

    cv2.putText(frame,tracker_name,(20,400),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    cv2.putText(frame,time,(20,450),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    cv2.imshow(tracker_name,frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()