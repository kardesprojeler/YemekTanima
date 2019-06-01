import cv2
import numpy as np

img = cv2.imread(r'C:\Users\BULUT\Desktop\indir.jpg', 0)
img = cv2.medianBlur(img, 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 3, 100, param1=100, param2=60, minRadius=10, maxRadius=100)

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 2)

cv2.imshow('detected circles', cimg)
cv2.imshow('res', img)
cv2.waitKey(0)
cv2.destroyAllWindows()