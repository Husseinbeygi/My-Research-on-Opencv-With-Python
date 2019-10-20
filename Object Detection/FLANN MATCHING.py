### Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np

### load Images
Original = cv2.imread('DATA/socket.png')
Target = cv2.imread('DATA/socket2.jpg')
Original = cv2.cvtColor(Original,cv2.COLOR_BGR2GRAY)
Target = cv2.cvtColor(Target,cv2.COLOR_BGR2GRAY)

# ret,thresh = cv2.threshold(frame,(frame.max() /2),frame.max(),cv2.THRESH_BINARY)
Original = cv2.adaptiveThreshold(Original,Original.max(),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)
Target = cv2.adaptiveThreshold(Target,Target.max(),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)
### Create SIFT Function
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(Original, None)
kp2, des2 = sift.detectAndCompute(Target, None)
### Initial the flann prameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params = dict(checks = 50)
### Create flann Matcher
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
flann_matches = cv2.drawMatchesKnn(Original, kp1, Target, kp2, matches, None, **draw_params)

plt.imshow(flann_matches)
plt.title('FLANN FEATURE MATCHING')
plt.show()
