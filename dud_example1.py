import cv2
import numpy as np

# img = cv2.imread('pic224.jpg')
# cv2.imshow('res', img)
#
# cv2.waitKey(0)

# cap = cv2.VideoCapture('videoplayback.mp4')
# while True:
#     success, img = cap.read()
#     cv2.imshow('Result', img)
#
#     if cv2.waitKey(1) & 0xff == ord('q'): # задержка 1 мс и ожидание нажатия клавиши q
#         break

img = cv2.imread('img.jpg')
new_img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
new_img = cv2.GaussianBlur(new_img, (9, 9), 5)
new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
new_img = cv2.Canny(new_img, 45, 45)

kernel = np.ones((5, 5), np.uint8) #матрица 5*5
new_img = cv2.dilate(new_img, kernel, iterations=1) #увеличить жирность линии
new_img = cv2.erode(new_img, kernel, iterations=1) #уменьшить жирность линии

cv2.imshow('', new_img[0:250, 0:250])
cv2.waitKey(0)