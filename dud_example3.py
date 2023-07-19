"""
import cv2

img = cv2.imread('images/face_2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cv2.CascadeClassifier('faces.xml') # вытягивает файл как натренерованную модель

results = faces.detectMultiScale(gray, scaleFactor=2, minNeighbors=4) # принимает картинки, на сколько отличается размер лица от размера в напренерованых изображения, сколько минимум лиц на картинки нужно обнаружить

for (x, y, w, h) in results: # в results будет храниться адрес где находится лицо
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=3) # рисуем кравдрат вокруг найденно лица

cv2.imshow('img', img)
cv2.waitKey(0)
"""
import cv2
import numpy as np
import imutils
#import easyocr
from matplotlib import pyplot as pl

img = cv2.imread('images/car_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_filter = cv2.bilateralFilter(gray, 11, 15, 15) # диаметр (насколько много пикселей будет охвачено), цветовое пространство (насколько много пикселей с примерно одинаковым цветом будет смешано), координатное пространство (тоже что и 2 только одинаковые по координатам)
img_edges = cv2.Canny(img_filter, 30, 200) # пороги

pl.imshow(cv2.cvtColor(img_edges, cv2.COLOR_BGR2RGB))
pl.show()
