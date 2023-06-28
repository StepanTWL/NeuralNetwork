##################### Преобразование С в F #####################
#F=C(x)*1.8(w1)+32(bias)*(w2=1)      х_____w1__>O___y>
#                                    bias__w2__/
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #выходной журнал tensorflow фиксирует только ошибки

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense

c = np.array([-100, -74, -55, -32, -19, -11, -6, 0, 3, 7, 12, 31, 35, 49, 72, 99, 152])
f = np.array([-148, -101.2, -67, -25.6, -2.2, 12.2, 21.2, 32, 37.4, 44.6, 53.6, 87.8, 95, 120.2, 161.6, 210.2, 305.6])

model = keras.Sequential() # создается модель многослойной последовательной нейронной сети
model.add(Dense(units=1, input_shape=(1,), activation='linear')) # один нейрон один вход (не включая bias-смещение, он создается автоматически)
                                                                 # активационная функция - linear

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.2)) # критерий качества (функция потерь, loss) - средний квадрат ошибок
                                                                               # способ обтимизации градиентного спуска - Adam
                                                                               # шаг сходимости - 0.1

history = model.fit(c, f, epochs=500, verbose=False) # тренировка сети (500 проходов всех значений, без вывода лог в консоль (false))

plt.plot(history.history['loss'])
plt.grid(True)
#plt.show()

print(model.predict([2.555]))
print(model.get_weights())
