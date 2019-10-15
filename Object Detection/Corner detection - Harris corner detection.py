### Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np

### load Images 
img = cv2.imread('DATA/chess.jpg')

# Make the picture gray scale 
_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Make the picture float
_img = np.float32(_img)

# Use the Harris Corner Detection Algoritm
# for More Information : https://en.wikipedia.org/wiki/Harris_Corner_Detector

HCD = cv2.cornerHarris(_img,blockSize=2,ksize=3,k=0.04)

# Morphological operations are a set of operations that process images based on shapes.
# They apply a structuring element to an input image and generate an output image.
HCDD = cv2.dilate(HCD,None)

# Make the Corners RED
img[HCDD > (0.01*HCDD.max())] = [255,0,0] #RGB Color Code

# Plot and show the result
    
#make a subplot 1
plt.subplot(131)
plt.imshow(img)
plt.title('the result of opration')

#make a subplot 2
plt.subplot(132)
plt.imshow(HCD)
plt.title('Harris C. D.')

#make a subplot 3
plt.subplot(133)
plt.imshow(HCDD)
plt.title('Harris C. D. with dilate')


#make Super title for know waht metods do we use
plt.suptitle('Harris Corner Detection')
plt.show()
