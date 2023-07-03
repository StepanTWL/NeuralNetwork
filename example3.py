##################### Распознование рукописных цифр #####################
import os

from keras.layers import BatchNormalization
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

limit = 5000
x_train_data = x_train[:limit]
y_train_data = y_train_cat[:limit]
x_valid = x_train[limit:limit*2]
y_valid = y_train_cat[limit:limit*2]

#model = keras.Sequential([Flatten(input_shape=(28, 28, 1)), Dense(300, activation='relu'), Dropout(0.5), Dense(10, activation='softmax')]) # 80% нейронов (300*0,8) будут отбрасываться
model = keras.Sequential([Flatten(input_shape=(28, 28, 1)), Dense(300, activation='relu'), BatchNormalization(), Dense(10, activation='softmax')]) # 80% нейронов (300*0,8) будут отбрасываться
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # оптимизация по adam, ошибки - categorical_crossentropy (лучше использовать для задачах классификации, а на выходе использовать softmax), metrics для вывода метрики
his = model.fit(x_train_data, y_train_data, batch_size=32, epochs=50, validation_data=(x_valid, y_valid)) # после каждых 32 изображений будут обновляться весовые коеф., 5 проходов, 80%/20% обучение/валидация (проверка что не идет переобучение выборки)
model.evaluate(x_test, y_test_cat) # проверка на тестовой выборке

plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.show()