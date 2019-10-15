### Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np


### load Images 
img = cv2.imread('DATA/con.png',0)

## FInd Contours on image 
## Note the MODE is on the full type RETR_CCOMP
image,contours,hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

## SHOW THE EXTERNAL CONTOURS

# Make a blank Page
external = np.zeros(image.shape)
# Draw the countors on blank page

# Go through contours 
for i in range(len(contours)):
    # find the one that is externals means equal to -1
    if hierarchy [0][i][3] == -1:
        # draw the contours
        cv2.drawContours(external,contours,i,255,-1) # TODO the external page is blank and donot have any od external hierachyes


## SHOW THE INTERNAL CONTOURS

# Make a blank Page
internal = np.zeros(image.shape)
# Draw the countors on blank page

# Go through contours 
for i in range(len(contours)):
    # find the one that is internal means equal to -1
    if hierarchy [0][i][3] != -1:
        # draw the contours
        cv2.drawContours(internal,contours,i,255,-1)



### Plot and show the result
    
# Make a subplot 1
plt.subplot(131)
plt.imshow(image,cmap='gray')
plt.title('Origianl Image')
# Make a subplot 2
plt.subplot(132)
plt.imshow(external,cmap = 'gray')
plt.title('External Contours')
# Make a subplot 3
plt.subplot(133)
plt.imshow(internal,cmap='gray')
plt.title('Internal Contours')

plt.show()