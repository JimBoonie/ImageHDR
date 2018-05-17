import cv2 
import numpy as np

ref_img_name = 'imgs/reference1.jpg'
ali_img_name = 'imgs/aligned1.jpg'

img = cv2.imread(ref_img_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)

img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('imgs/sift_keypoints_opencv.jpg', img)