import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

img = cv2.imread('images/OpenCV_logo.png')
img_crop = img[100:200, 200:300]

height, width = img.shape[:2]
center = (width // 2.0, height // 2.0)

M1 = np.float32([[1, 0, 10], [0, 1, 0]])
M2 = cv2.getRotationMatrix2D(center, 180, 1)
p1 = np.float32([[0, 0], [0, 50], [50, 0]])
p2 = np.float32([[10, 10], [20, 70], [40, 0]])
M3 = cv2.getAffineTransform(p1, p2)

img_moved = cv2.warpAffine(img, M1, (width, height))
img_rotated = cv2.warpAffine(img, M2, (width, height))
img_trans = cv2.warpAffine(img, M3, (width, height))


panel1 = np.zeros((300, 300), dtype='uint8')
panel2 = np.zeros((300, 300), dtype='uint8')
rect = cv2.rectangle(panel1, (25, 25), (275, 275), 255, -1)
circle = cv2.circle(panel2, (150, 150), 150, 255, -1)
img_and = cv2.bitwise_and(rect, circle)
img_or = cv2.bitwise_or(rect, circle)
img_xor = cv2.bitwise_xor(rect, circle)

(B, G, R) = cv2.split(img)
zeros = np.zeros(img.shape[:2], dtype='uint8')
img_merge = cv2.merge([zeros, G, zeros])

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

plt.imshow(img_lab)
plt.show()
