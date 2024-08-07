import cv2 as cv

##resizing and rescaling
def rescale_frame(frame, scale = 0.75):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dim = (width, height)

    return cv.resize(frame, dim, interpolation = cv.INTER_AREA)