import cv2
import numpy as np

# photo = np.zeros((450, 450, 3), dtype='uint8') #матрица (рисунок) 300*300 и 3 слоя (3 цвета)
#
# photo[:] = 255, 0, 0
# cv2.rectangle(photo, (50, 70), (200, 200), (100, 100, 100), thickness=5) # квадрат
# cv2.circle(photo, (photo.shape[1]//2, photo.shape[0]//2), 100, (100, 100, 100), thickness=cv2.FILLED)
#
# cv2.imshow('Photo', photo)
# cv2.waitKey(0)

# cap = cv2.VideoCapture('videoplayback.mp4')
#
# while True:
#     success, img = cap.read()
#
#     img = cv2.GaussianBlur(img, (9,9), 0)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     img = cv2.Canny(img, 100, 100)
#
#     kernel = np.ones((5,5), np.uint8)
#     img = cv2.dilate(img, kernel, iterations=1)
#     img = cv2.erode(img, kernel, iterations=1)
#
#     cv2.imshow('Video', img)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

img = cv2.imread('img.jpg')
new_img = np.zeros(img.shape, dtype='uint8')


# img = cv2.flip(img, -1) # зеркальное отображение изображения
def rotate(img_param, angle): # функцуя поворота
    height, width = img_param.shape[:2]
    point = (width // 2, height // 2)
    mat = cv2.getRotationMatrix2D(point, angle, 1)
    rotated = cv2.warpAffine(img_param, mat, (width, height))
    return rotated

#img = rotate(img, 45)

def transform(img_param, x, y): # функия трансформации
    height, width = img_param.shape[:2]
    mat = np.float32([[1, 0, x], [0, 1, y]])
    transformed = cv2.warpAffine(img_param, mat, (width, height))
    return transformed

#img = transform(img, 100, 100)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (7, 7), 0)
img = cv2.Canny(img, 100, 140) # серые цвета меньше 100 заменятся на 0, серые цвета больше 140 заменятся на 255, остальные останутся как есть

con, hir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(new_img, con, -1, (230, 111, 148), 1)

cv2.imshow('Image', new_img)
cv2.waitKey(0)
