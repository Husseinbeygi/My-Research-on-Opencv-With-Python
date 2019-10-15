### Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np


### load Images 
full = cv2.imread('DATA/chess.jpg')

## Find Chessboad Corners 
found,corners = cv2.findChessboardCorners(full,(7,7))

## Draw on chess board
chessdraw = cv2.drawChessboardCorners(full,(7,7),corners,found)
## Show the image
plt.imshow(chessdraw)
plt.show()


## For dots detection use : 
# cv2.findCirclesGrid