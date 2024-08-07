import cv2

img = cv2.imread('data/images/spidey.jpg')

#changing image to grayscale

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

##gaussian blurr
#blurring an image
blur = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
cv2.imshow('Blur', blur)

#edge cascade - to find edges present in the image
canny = cv2.Canny(blur, 125, 175)
cv2.imshow('Canny', canny)

#dialation - dilate an image using structing element eg edges in this case
dilated = cv2.dilate(canny, (7, 7), iterations = 3)
cv2.imshow('Dilated', dilated)

#erosion
eroded = cv2.erode(dilated, (3,3), iterations  = 1)
cv2.imshow('erode', eroded)

cv2.waitKey(0)