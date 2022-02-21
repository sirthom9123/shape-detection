import cv2
import numpy as np
from matplotlib import pyplot as plt

# read image
img = cv2.imread('stock.jpg', cv2.IMREAD_COLOR)

# converting image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# setting threshold of gray image
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(
	                    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# list for storing names of shapes
for obj in contours:
	
	# using drawContours() function to get the shape
	cv2.drawContours(img, [obj], -1, (100, 0, 255), 3)


# displaying the image
cv2.imshow('shape', img)

# plot the image on graph
plt.imshow(img)
plt.show()