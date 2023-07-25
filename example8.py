##################### Прогнозирование символов #####################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re

from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer

with open('train_data_true.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff', '') # убираем первый невидимый символ
    text = re.sub(r'[^А-я ]', '',  text) # заменяет все кроме кириллицы на пустые символы

num_characters = 34 # 33 буквы + пробел
tokenirez = Tokenizer(num_words=num_characters, char_level=True) # токенизация текста на уровне символов
tokenirez.fit_on_texts([text]) # формируем токены на основе частотности в нашем тексте
print(tokenirez.word_index) # полученная колекция токенов

inp_chars = 6
data = tokenirez.texts_to_matrix(text) # преобразовывыем текст в матрицу токенов (массив OHE)
n = data.shape[0] - inp_chars # так как мы предсказываем по шести символам - седьмой

X = np.array([data[i:i + inp_chars, :] for i in range(n)]) # 6 символом используемых для предсказания
Y = data[inp_chars:] # предсказываемый символ

print(data.shape)

model = Sequential()
model.add(Input((inp_chars, num_characters))) # при тренировке в рекуррентные модели keras подается вся последовательность
model.add(SimpleRNN(units=128, activation='tanh')) # рекуррентный слой на 128 нейронов (функция кативации - гипербалический тангенс)
model.add(Dense(num_characters, activation='softmax')) # полносвязный слой (функция кативации - softmax)
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],  optimizer='adam') # потери - категориальная кросс ентропия, метрикика - точность,  оптимизатор для градиентного спуска - адам

history = model.fit(X, Y, batch_size=32, epochs=200) # тренеровка модели

def buildPhrase(inp_str, str_len=50):
    for i in range(str_len):
        x = []
        for j in range(i, i + inp_chars):
            x.append(tokenirez.texts_to_matrix(inp_str[j])) # преобразовываеим текст в матрицу токенов (массива OHE)

        x = np.array(x)
        inp = x.reshape(1, inp_chars, num_characters) # формирует начальные символы по которым будет строиться прогноз
        pred = model.predict(inp) # на вход нейронной сети (предсказание 7 символа)
        d = tokenirez.index_word[pred.argmax(axis=1)[0]] # находим индекс наибольшего значения и подаем на колекцию (словарь)

        inp_str += d
    return inp_str

res = buildPhrase('привет')
print(res)