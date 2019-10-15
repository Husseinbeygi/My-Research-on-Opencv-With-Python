### Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
### load Images
Original = cv2.imread('DATA/8.jpg')

# Make Copy of Image 
Copy_image = np.copy(Original)

# Make Maker for Seeding Data
marker_image = np.zeros(Copy_image.shape[:2],dtype=np.int32)
segements = np.zeros(Copy_image.shape,dtype=np.uint8)

# Create Color for Markers
# cm.tab10(0)
# (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0)
# So We get the three col and *255  as Array & tuple
def Create_RGB(i):
    return tuple(np.array(cm.tab10(i)[:3])*255)
 
colors = []
for i in range(10):
    colors.append(Create_RGB(i))



####
## Global Variables
# COLOR CHOISE
current_mrker = 1
# MARKERS UPDATED BY WATERSHED
marks_update = False
## Callback Function

def mouse_callback(event,x,y,flags,param):
    global marks_update

    if event == cv2.EVENT_LBUTTONDOWN:
        # Markerrr Passed to Watershed ALGO.
        cv2.circle(marker_image,(x,y),10,(current_mrker),-1)

        # User Sees On the reoad image 
        cv2.circle(Copy_image,(x,y),10,colors[current_mrker],-1)

        marks_update = True
cv2.namedWindow('Image')
cv2.setMouseCallback('Image',mouse_callback)


# WHILE TURE

while True:
    # show the image
    cv2.imshow('Watershed Segments',segements)
    cv2.imshow('Image',Copy_image)
    # Close all Windows
    k = cv2.waitKey(1)
    if k == 27:
        break
    # clear all Color by press c
    elif k == ord('c'):
        # Make Copy of Image 
        Copy_image = np.copy(Original)

        # Make Maker for Seeding Data
        marker_image = np.zeros(Copy_image.shape[:2],dtype=np.int32)
        segements = np.zeros(Copy_image.shape,dtype=np.uint8)


    # update color choice
    elif k > 0 and chr(k).isdigit():
        current_mrker = int(chr(k))
    # update the marking
    if marks_update:
        marker_image_copy = marker_image.copy()
        cv2.watershed(Original,marker_image_copy)

        segements = np.zeros(Original.shape,dtype=np.uint8)

        for colors_ind in range(10):
            segements[marker_image_copy == (colors_ind)] = colors[colors_ind]


cv2.destroyAllWindows()