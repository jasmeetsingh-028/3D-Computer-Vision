import cv2 as cv
import numpy as np

img = cv.imread('data/images/spidey.jpg')
blank_img = np.zeros((500,500,3), dtype='uint8')
cv.imshow('Image', blank_img)
# blank_img[:] = 255,0,0
# cv.imshow('Imagegreen', blank_img)
# blank_img[200:300, 300:400] = 0,0,255
# cv.imshow('Imagered', blank_img)
cv.rectangle(blank_img, (0,0), (blank_img.shape[0]//2,blank_img.shape[1]//2), (0,255,0), thickness = -1)
cv.imshow('Imagerect', blank_img)

##drawing circle
cv.circle(blank_img, (blank_img.shape[0]//2, blank_img.shape[1]//2), 40, (0,0,255), thickness = 3)
cv.imshow('circleImg', blank_img)

##Drawing line
cv.line(blank_img, (0,0), (blank_img.shape[0]//2, blank_img.shape[1]//2), (0,0,255), thickness = 3)
cv.imshow('lineImg', blank_img)

##write text
cv.putText(blank_img, 'Hello Blank Image', (225,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), thickness = 2)
cv.imshow('lineImg', blank_img)

cv.waitKey(0)