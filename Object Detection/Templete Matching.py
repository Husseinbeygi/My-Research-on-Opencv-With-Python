### Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np


### load Images 
full = cv2.imread('DATA/5.jpg')
tmpl = cv2.imread('DATA/5_face.jpg')


### Metods for Comparison in a list 
#Use eval() function
Metods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']


### Create a for loop to check Different Metods
for m in Metods:
    #create a copy of full image
    full_copy = full.copy()

    #use eval to use string as function
    metods = eval(m)

    #Use Tempelete Matching 
    result  = cv2.matchTemplate(full_copy,tmpl,metods)

    ### Draw a rectangle around the tmpl in full image
    
    # Get the values and Locations in Result
    _min_val,_max_val,_min_loc,_max_loc = cv2.minMaxLoc(result)

    # Because the SQDIFF Metods for Different the other we need to make 
    # sure that they get the minimum pixel location in full image not the max 

    if metods in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        topleft = _min_loc
    else:
        topleft = _max_loc

    # Get the shape of tmpl 
    height , width,channels = tmpl.shape

    # set value for second point BOTTOM_RIGHT
    # the formula is 
    # topleft_x + width , topleft_y + height

    bottom_right = (topleft[0] + width , topleft[1] + height)

    # draw the rectangle 
    cv2.rectangle(full_copy,topleft,bottom_right,(255,255,0),10)

    ### Plot and show the result
    
    #make a subplot 1
    plt.subplot(121)
    plt.imshow(result)
    plt.title('the result of opration')

    #make a subplot 1
    plt.subplot(122)
    plt.imshow(full_copy)
    plt.title('the result of opration on the Image')

    #make Super title for know waht metods do we use
    plt.suptitle(m)

    plt.show()
