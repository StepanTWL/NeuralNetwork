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

cont = cv2.findContours(img_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # ищет контуры (режим (в не иерархичном порядке), метод нахождения контуров (позволяет найти начальные и конечные точки без промежуточных))
cont = imutils.grab_contours(cont) # выдает контуры в удобном формате
cont = sorted(cont, key=cv2.contourArea, reverse=True)[:20] # отсортирует по квардатам

pos = None
for c in cont:
    approx = cv2.approxPolyDP(c, 10, True) # большее число указывает что фигура ближе к квадрату, что начальная и конечная точка фигуры находится в одном месте (закрытая форма)
    if len(approx) == 4: # фигура состоит из 4 элементов (углов)
        pos = approx
        break

mask = np.zeros(gray.shape, np.uint8) # маска


#pl.imshow(cv2.cvtColor(img_edges, cv2.COLOR_BGR2RGB))
#pl.show()
