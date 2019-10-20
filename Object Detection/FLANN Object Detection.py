import cv2
from datetime import datetime

cap = cv2.VideoCapture(2)
orb = cv2.xfeatures2d.SIFT_create()


Target = cv2.imread('DATA/socket.png')
Target = cv2.cvtColor(Target,cv2.COLOR_BGR2GRAY)
Target = cv2.adaptiveThreshold(Target,Target.max(),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)
ret,Target = cv2.threshold(Target,(Target.max() /2),Target.max(),cv2.THRESH_BINARY)

kp1, des1 = orb.detectAndCompute(Target, None)

### Initial the flann prameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params,search_params)

while True:

    ret,frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.adaptiveThreshold(frame,frame.max(),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)
    ret,frame = cv2.threshold(frame,(frame.max() /2),frame.max(),cv2.THRESH_BINARY)

    kp2, des2 = orb.detectAndCompute(frame, None)

    ### Initial the flann prameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)


    # find matches
    matches = flann.knnMatch(des1, des2, k=2)

    # create match mask
    matchesMask = [[0,0] for i in range(len(matches))]


    ### RATIO MATCH1 < 75% MATCH 2

    for i,(match1, match2) in enumerate(matches):

        if match1.distance < 0.75 * match2.distance:
            matchesMask[i] = [1,0]

    # create mask for draw matches
    # the match point are green and non match point are red
    draw_params = dict(matchColor = (0,255,0),
                        singlePointColor=(255,0,0),
                        matchesMask=matchesMask,
                        flags=0)
    # draw falnn matches
    flann_matches = cv2.drawMatchesKnn(frame, kp2, Target, kp1, matches, None, **draw_params)

    
    cv2.imshow('Matches',flann_matches)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()