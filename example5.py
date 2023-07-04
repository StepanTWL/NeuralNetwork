from io import BytesIO

import keras.applications
import numpy as np
from PIL import Image

model = keras.applications.VGG16()

img = Image.open('pic224.jpg')
imp = np.array(img) # превращение в массив
x = keras.applications.vgg16.preprocess_input(img) # получаем массив в нужном цветовом формате и смещениями
x = np.expand_dims(x, axis=0) # преобразуем в формат необходимый для модели
y = model.predict(x)

print(y)
print(type(y))
with open('types_vgg16.txt', 'r') as file:
    text = file.read()
print(text[text.find(str(y)):(text[text.find(str(y)):].find('\n')+text.find(str(y)))])

