import cv2
import numpy as np
from matplotlib import pyplot as plt

# read image
img = cv2.imread('shapes.jfif', cv2.IMREAD_COLOR)

# converting image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# setting threshold of gray image
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(
	                    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i=0

# list for storing names of shapes
for obj in contours:
	
	# using drawContours() function to get the shape
	cv2.drawContours(img, [obj], -1, (100, 0, 255), 3)

	# here we are ignoring first counter because
	# findcontour function detects whole image as shape
	if i == 0:
		i = 1
		continue

	# cv2.approxPloyDP() function to approximate the shape
	approx = cv2.approxPolyDP(
		obj, 0.01 * cv2.arcLength(obj, True), True)
	
	# using drawContours() function to get the shape
	cv2.drawContours(img, [obj], -1, (0, 255, 0), 3)

	# identify center of shape
	M = cv2.moments(obj)
	if M['m00'] != 0.0:
		x = int(M['m10']/M['m00'])
		y = int(M['m01']/M['m00'])

	# map text to image 
	if len(approx) == 3:
		cv2.putText(img, 'Triangle', (x, y),
					cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.6, (31, 33, 32), 2)

	elif len(approx) == 4:
		cv2.putText(img, 'Quadrilateral', (x, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (31, 33, 32), 2)

	elif len(approx) == 5:
		cv2.putText(img, 'Pentagon', (x, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (31, 33, 32), 2)

	elif len(approx) == 6:
		cv2.putText(img, 'Hexagon', (x, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (31, 33, 32), 2)

	else:
		cv2.putText(img, 'circle', (x, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (31, 33, 32), 2)

# displaying the image
cv2.imshow('shapes', img)

# plot the image on graph
plt.imshow(img)
plt.show()