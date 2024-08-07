import cv2 as cv
import numpy as np
from rescale import rescale_frame

#READING IMAGES

# img = cv.imread('data/images/spidey.jpg')

# cv.imshow('Image', img)
# cv.waitKey(0) #waits for infinite amout of time for a keyboard key to be pressed

#READING VIDEOS

capture = cv.VideoCapture('data/videos/vid.mp4')  #0 for webcam capture

while True:
    isTure, frame = capture.read()
    cv.imshow('origVideo', frame)
    frame = rescale_frame(frame, scale = 0.25)
    cv.imshow('rescaleVideo', frame)
    
    if cv.waitKey(20) & 0xFF == ord('d'):  #if letter d is pressed
        break
capture.release()
cv.destroyAllWindows()

##resizing and rescaling

