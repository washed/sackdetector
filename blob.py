# Standard imports
import cv2
import numpy as np
from time import sleep

def _detect(img, img_bg, detector: cv2.SimpleBlobDetector):
    diff = cv2.absdiff(img_bg, img)

    img_hsv = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)

    mask_red_upper = cv2.inRange(img_hsv, (0,50,100), (5,255,255))
    mask_red_lower = cv2.inRange(img_hsv, (150,50,100), (255,255,255))
    mask_red = cv2.bitwise_or(mask_red_upper, mask_red_lower)

    mask_blue = cv2.inRange(img_hsv, (90,120,100), (105,255,255))

    cropped_red = cv2.bitwise_and(img, diff, mask=mask_red)
    cropped_blue = cv2.bitwise_and(img, diff, mask=mask_blue)

    gray_red = cv2.cvtColor(cropped_red, cv2.COLOR_BGR2GRAY)
    blurred_red = cv2.GaussianBlur(gray_red, (5, 5), 5)
    thresh_red = cv2.threshold(blurred_red, 20, 255, cv2.THRESH_BINARY)[1]

    gray_blue = cv2.cvtColor(cropped_blue, cv2.COLOR_BGR2GRAY)
    blurred_blue = cv2.GaussianBlur(gray_blue, (5, 5), 5)
    thresh_blue = cv2.threshold(blurred_blue, 60, 255, cv2.THRESH_BINARY)[1]

    blobs_red = detector.detect(thresh_red)
    img_with_blobs_red = cv2.drawKeypoints(cropped_red, blobs_red, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    blobs_blue = detector.detect(thresh_blue)
    img_with_blobs_blue = cv2.drawKeypoints(cropped_blue, blobs_blue, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    img_with_blobs = cv2.drawKeypoints(img, blobs_red, np.array([]), (255,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_with_blobs = cv2.drawKeypoints(img_with_blobs, blobs_blue, np.array([]), (255,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    ## Display
    cv2.imshow("diff", diff)
    cv2.imshow("cropped_red", cropped_red)
    cv2.imshow("cropped_blue", cropped_blue)
    cv2.imshow("thresh_red", thresh_red)
    cv2.imshow("thresh_blue", thresh_blue)
    cv2.imshow("img_with_blobs", img_with_blobs)
    cv2.waitKey(1)


    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # img_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    # cv2.imshow("Keypoints", img_with_keypoints)
    # cv2.imwrite("img_with_keypoints.jpg", img_with_keypoints)
    # cv2.waitKey(0)


# Set up the detector with default parameters.
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
# params.maxThreshold = 200

# Filter by Area
params.filterByArea = True
params.minArea = 500
# params.maxArea = 5000

params.filterByColor = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.0
params.maxCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8
params.maxConvexity = 1.0

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01


detector = cv2.SimpleBlobDetector_create(params)


_img_state, img_bg = cam.read()

while True:
    img_state, img = cam.read()
    if not img_state:
        continue
    _detect(img, img_bg, detector)
