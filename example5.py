from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Загрузка предварительно обученной модели VGG16
model = VGG16(weights='imagenet')

# Загрузка изображения и предварительная обработка
img_path = 'pic224.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Предсказание класса изображения
preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]

# Вывод результатов
for pred in decoded_preds:
    print(f"Class: {pred[1]}, Probability: {pred[2]}")


