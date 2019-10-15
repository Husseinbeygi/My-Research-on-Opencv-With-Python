### Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np


### load Images 
full = cv2.imread('DATA/chess.jpg')



### Blur the Image for Reduce too much details in image 
img = cv2.blur(full,ksize=(6,6))



### Calculate the threshhold for canny Edge Detection
# Get the median Value of image 
med = np.median(full)
# Lower threshold to either 0 or 70% of the median value wichever is greater
thresh1 = int(max(0,(0.7 * med)))
# higher threshold to either 0 or 130% of the median value wichever is smaller
thresh2 = int(min(0,(1.3 * med)))
# The K variable is for make the edge detection better
K = 100
thresh2 = thresh2 + K


### Excute the Canny edge Detection
result = cv2.Canny(img,thresh1,thresh2) 




### Plot and show the result
    
# Make a subplot 1
plt.subplot(121)
plt.imshow(img)

# Make a subplot 2
plt.subplot(122)
plt.imshow(result)

plt.show()