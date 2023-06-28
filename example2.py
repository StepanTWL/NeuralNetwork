##################### Распознование рукописных цифр #####################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #выходной журнал tensorflow фиксирует только ошибки

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist         # библиотека рукописных цифр

(x_train, y_train), (x_test, y_test) = mnist.load_data() # загруска всех выборок (60000(очучающих)+10000(тестовых)) x_train[i]='рисунок 5' y_train[i]=5

x_train = x_train / 255      # стандартизация входных данных (что бы данные были в диапазоне от 0 до 1 для логичтических функция активации
                             #                                если так не делать то при большой входном значении на нейроне и это может привести к попаданию в пологую область функции
#                                                             а это область где производная функции близка к нулю и изменение весовых коэф. близко к нулю)
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10) # подготавливает данные и преобразовывает цифру в булевый вектор (для 10 синопсов выходного слоя) 5 -> [0,0,0,0,0,1,0,0,0,0]
y_test_cat = keras.utils.to_categorical(y_test, 10)

#Слой1 (входной) Преобразование рисунков (матрицы) выборок (28*28*pix) в вектор длинной 784 значения
#Cлой2 128 нейронов и функция активации relu
#Cлой3 (выходной) 10 нейронов (по оличиству цифр) и функция активации softmax (теорияя вероятности)
model = keras.Sequential([Flatten(input_shape=(28, 28, 1)), Dense(128, activation='relu'), Dense(10, activation='softmax')])

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # оптимизация по adam, ошибки - categorical_crossentropy (лучше использовать для задачах классификации, а на выходе использовать softmax), metrics для вывода метрики

model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2) # после каждых 32 изображений будут обновляться весовые коеф., 5 проходов, 80%/20% обучение/валидация (проверка что не идет переобучение выборки)

model.evaluate(x_test, y_test_cat) # проверка на тестовой выборке

n = 100
x = np.expand_dims(x_test[n], axis=0) # что бы из 2х мерной матрицы сделать 3х мерную
res = model.predict(x) # можно подавать только 3х мерную матрицу (несколько изображений)
print(res)
print(f'Цифра: {np.argmax(res)}') # индекс максимального значения
plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1) # axis=1 во второй области списка (слой3)

mask = pred != y_test
x_false = x_test[mask]
y_false = pred[mask]
print(x_false.shape)

for i in range(10):
    print("Значкение сети:"+str(y_test[i]))
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()