### Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np

### load Images
Original = cv2.imread('../DATA/selfie.jpg')
Target = cv2.imread('../DATA/ellen.jpg')

###############################################
############### ORB FEATURE MATCHING #########
###############################################

### Create ORB Function
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(Original, None)
kp2, des2 = orb.detectAndCompute(Target, None)

### Create bf Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# find matches
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
bf_matches = cv2.drawMatches(Original, kp1, Target, kp2, matches[:25], None, flags=2)

plt.imshow(bf_matches)
plt.title('ORB FEATURE MATCHING')
plt.show()

###############################################
############### SIFT FEATURE MATCHING #########
###############################################

### Create SIFT Function
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(Original, None)
kp2, des2 = sift.detectAndCompute(Target, None)

### Create bf Matcher
bf = cv2.BFMatcher()
# find matches
matches = bf.knnMatch(des1, des2, k=2)

### RATIO MATCH1 < 75% MATCH 2
goodmatches = []

for match1, match2 in matches:

    if match1.distance < 0.75 * match2.distance:
        goodmatches.append([match1])

sift_matches = cv2.drawMatchesKnn(Original, kp1, Target, kp2, goodmatches, None, flags=2)

plt.imshow(sift_matches)
plt.title('SIFT FEATURE MATCHING')
plt.show()

###############################################
#### FLANN FEATURE MATCHING With Mask #########
###############################################

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
