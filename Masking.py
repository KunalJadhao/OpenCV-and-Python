import cv2 as cv
import numpy as np

# Read the image
image = cv.imread('Photos/scribblerkunal.jpg')
if image is None:
    print("Error: Image not Found!")
    exit()

#cv.imshow('Original Image', image)

# Create a blank mask with the same height and width as the image
blank = np.zeros(image.shape[:2], dtype='uint8')

# Create a circle mask at the center of the image
circle = cv.circle(blank.copy(), (image.shape[1]//2, image.shape[0]//2), 200, 255, -1)

# Create a rectangle mask with properly adjusted coordinates
rectangle = cv.rectangle(blank.copy(), (400, 200), (900, 600), 255, -1)

# Perform bitwise AND to get the intersection (weird shape)
weird_shape = cv.bitwise_and(circle, rectangle)

# Display the masks and results
#cv.imshow('Circle Mask', circle)
#cv.imshow('Rectangle Mask', rectangle)
cv.imshow('Weird Shape', weird_shape)

cv.waitKey(0)
cv.destroyAllWindows()
