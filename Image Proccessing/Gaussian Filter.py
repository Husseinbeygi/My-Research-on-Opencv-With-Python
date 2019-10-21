import cv2
import matplotlib.pyplot as plt
import numpy as np
src = cv2.imread("DATA/filter_sample2.jpg",0)
#########################
####### BLUR  ###########
#########################

##  kernel = (5,5)
"""
######Average############
filtername = "Average Blur"
gb = cv2.blur(src,(5,5),0)

plt.subplot(121)
plt.imshow(src)
plt.title("Before")

plt.subplot(122)
plt.imshow(gb)
plt.title("After")

plt.suptitle(filtername)

plt.show()

#####Gaussian Filter#####
filtername = "Gaussian Blur"
gb = cv2.GaussianBlur(src,(9,9),0)

plt.subplot(121)
plt.imshow(src)
plt.title("Before")

plt.subplot(122)
plt.imshow(gb)
plt.title("After")

plt.suptitle(filtername)

plt.show()

#####Median Filter#####
filtername = "Median Blur"
gb = cv2.medianBlur(src,5)
plt.subplot(121)
plt.imshow(src)
plt.title("Before")
plt.subplot(122)
plt.imshow(gb)
plt.title("After")
plt.suptitle(filtername)
plt.show()


#####Bilateral Filter#####
filtername = "Bilateral Blur"
gb = cv2.bilateralFilter(src,9,75,75)
plt.subplot(121)
plt.imshow(src)
plt.title("Before")
plt.subplot(122)
plt.imshow(gb)
plt.title("After")
plt.suptitle(filtername)
plt.show()

"""
# Image Tresholding

ret,thresh1 = cv2.threshold(src,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(src,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(src,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(src,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(src,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [src, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
