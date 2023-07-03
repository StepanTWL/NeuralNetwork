##################### Распознование рукописных цифр #####################
import os

import numpy as np
from keras.layers import Dropout, BatchNormalization, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist         # библиотека рукописных цифр

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #выходной журнал tensorflow фиксирует только ошибки

(x_train, y_train), (x_test, y_test) = mnist.load_data() # загруска всех выборок (60000(очучающих)+10000(тестовых)) x_train[i]='рисунок 5' y_train[i]=5

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10) # подготавливает данные и преобразовывает цифру в булевый вектор (для 10 синопсов выходного слоя) 5 -> [0,0,0,0,0,1,0,0,0,0]
y_test_cat = keras.utils.to_categorical(y_test, 10)

x_train = np.expand_dims(x_train, axis=3) # добавляется 4 ось (параметр) для нужного формата входных данных
x_test = np.expand_dims(x_test, axis=3)

print(x_train.shape)

model = keras.Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)), # первый сверточный слой
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3, 3), padding='same', activation='relu'), # второй сверточный слой
    MaxPooling2D((2, 2), strides=2),
    Flatten(), # вытязиваем вектор
    Dense(128, activation='relu'), # слой 128 нейронов
    Dense(10, activation='softmax') # выходной слой 10 нейроноел
])

print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train_cat, epochs=5, batch_size=32, validation_split=0.2)
model.evaluate(x_test, y_test_cat)