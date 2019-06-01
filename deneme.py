import numpy as np
# import the necessary packages
import argparse
import glob
import cv2


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

# construct the argument parse and parse the arguments
# loop over the images
image = cv2.imread(r'‪C:\Users\Durkan\Desktop\durkan.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
tight = cv2.Canny(blurred, 300, 400)
auto = auto_canny(tight)

# show the images
cv2.imshow("Original", image)
cv2.imshow("Edges", np.hstack([tight, auto]))
cv2.waitKey(0)