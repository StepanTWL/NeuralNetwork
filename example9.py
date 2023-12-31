##################### Прогнозирование слов (рекурентные сети RNN) #####################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from tensorflow.keras.layers import Dense, SimpleRNN, Input, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import to_categorical

with open('text', 'r', encoding='utf-8') as f:
    texts = f.read()
    texts = texts.replace('\ufeff', '')

maxWordsCount = 1000
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                      lower=True, split=' ', char_level=False) # разбиваем текст на слова (первые 1000 часто встречающихся слов)
tokenizer.fit_on_texts([texts]) # преобразовываем текс

dist = list(tokenizer.word_counts.items())
print(dist[:10])

data = tokenizer.texts_to_sequences([texts]) # преобразование в OHE вектор
#res = to_categorical(data[0], num_classes=maxWordsCount) # формируем 3х мерный тензор
#print(res.shape)
res = np.array(data[0])

inp_words = 3
n = res.shape[0] - inp_words

X = np.array([res[i:i + inp_words] for i in range(n)]) # формируем 3х мерный тензоры
# Y = res[inp_words:]
Y = to_categorical(res[inp_words:], num_classes=maxWordsCount)

# Рекурентная нейронная сеть
model = Sequential()
model.add(Embedding(maxWordsCount, 256, input_length = inp_words)) # при тренировке в рекуррентные модели keras подается вся последовательность
model.add(SimpleRNN(128, activation='tanh', return_sequences=True)) # рекуррентный слой на 128 нейронов c данными (batch_size, timesteps, units) для следующего рекурентного слоя
model.add(SimpleRNN(64, activation='tanh')) # рекуррентный слой на 64 нейронов (функция кативации - гипербалический тангенс) с данными (batch_size, units)
model.add(Dense(maxWordsCount, activation='softmax')) # полносвязный слой (функция кативации - softmax)
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
history = model.fit(X, Y, batch_size=32, epochs=50)

def buildPhrase(texts, str_len=20):
    res = texts
    data = tokenizer.texts_to_sequences([texts])[0]
    for i in range(str_len):
        #x = to_categorical(data[i:i + inp_words], num_classes=maxWordsCount)
        #inp = x.reshape(1, inp_words, maxWordsCount)
        x = data[i:i+inp_words]
        inp = np.expand_dims(x, axis=0)

        pred = model.predict(inp) # на вход нейронной сети (предсказание 3 слова)
        indx = pred.argmax(axis=1)[0] # индекатор макисмальноезначения
        data.append(indx)

        res += " " + tokenizer.index_word[indx]

    return res


res = buildPhrase("позитив добавляет годы")
print(res)