import cv2
import  numpy as np

photo = np.zeros((450, 450, 3), dtype='uint8') #матрица (рисунок) 300*300 и 3 слоя (3 цвета)

photo[:] = 255, 0, 0
cv2.rectangle(photo, (50, 70), (200, 200), (100, 100, 100), thickness=5) # квадрат
cv2.circle(photo, (photo.shape[1]//2, photo.shape[0]//2), 100, (100, 100, 100), thickness=cv2.FILLED)

cv2.imshow('Photo', photo)
cv2.waitKey(0)