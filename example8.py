import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re

from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer

with open('train_data_true.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff', '') # убираем первый невидимый символ
    text = re.sub(r'[^А-я]', '',  text) # заменяет все кроме кириллицы на пустые символы

num_characters = 34 # 33 буквы + пробел
tokenirez = Tokenizer(num_words=num_characters, char_level=True) # токенизация текста на уровне символов
tokenirez.fit_on_texts([text]) # формируем токены на основе частотности в нашем тексте
print(tokenirez.word_index) # полученная колекция токенов

inp_chars = 6
data = tokenirez.texts_to_matrix(text) # преобразовывыем текст в матрицу токенов (массив OHE)
n = data.shape[0] - inp_chars # так как мы предсказываем по шести символам - седьмой

X = np.array([data[i:i + inp_chars, :] for i in range(n)])
Y = data[inp_chars:] # предсказываемый символ

