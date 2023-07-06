import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #выходной журнал tensorflow фиксирует только ошибки
import numpy as np
from PIL import Image
from keras import Sequential
from keras.layers import Conv2D, InputLayer, UpSampling2D
from matplotlib import pyplot as plt
from tensorflow import keras
from skimage.color import rgb2lab, lab2rgb

img_path = 'cats400.jpg'
img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))


def preprocess_image(image): # преобразование rgb в lab и размер изображения для модели
    image = img.resize((224, 224), Image.BILINEAR)
    image = np.array(image,  dtype=np.float32)
    size = image.shape
    lab = rgb2lab(1.0 / 255 * image)
    X, Y = lab[:, :, 0], lab[:, :, 1:]

    Y /= 128
    X = X.reshape(1, size[0], size[1], 1)
    Y = Y.reshape(1, size[0], size[1], 2)
    return X, Y, size

X, Y, size = preprocess_image(img) # преобразование rgb в lab и размер изображения для модели

model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1))) # на вход модели подается изображение в формате lab
model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) # сверточный слой с 64 каналами
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)) # шаг сканирования 2, карта признака уменьшается в 2 раза это вместо maxpooling если нужно сохранить пространственную структуру
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2))) # увеличивает карту признака в два раза (три раза в два раза уменьшили и потом три раза в два раза увеличили - размер карты признаков на выходе = на входе)
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same')) # два цветовых каналов ('tanh' - что бы получить от -1 до 1)

model.compile(optimizer='adam', loss='mse')
model.fit(x=X, y=Y, batch_size=1, epochs=50)

img_path = 'cats400.jpg'
img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

X, Y, size = preprocess_image(img)

output = model.predict(X)
output *= 128
min_vals, max_vals = -128, 127
ab = np.clip(output[0], min_vals, max_vals)

cur = np.zeros((size[0], size[1], 3))
cur[:,:,0] = np.clip(X[0][:,:,0], 0, 100)
cur[:,:,1:] = ab
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(lab2rgb(cur))