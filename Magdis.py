from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from PIL import Image

image_width, image_height = 530, 320 #разрешение для изображений к единому формату

directory_data_train= '/content/drive/MyDrive/ВУЗ/аспирантура/dataset/train' #путь к обучающей выборке train_data_dir
directory_data_validation= '/content/drive/MyDrive/ВУЗ/аспирантура/dataset/val'  #путь к проверочной выборке validation_data_dir

#необходимые параметры

train_sample = 160
validation_sample = 80
epochs = 8
lot_size = 1  #batch_size
if K.image_data_format() != 'channels_first':
     input_shape = (image_width, image_height, 3)
else:
     input_shape = (3, image_width, image_height)

pattern = Sequential() #создание модели

#первый слой нейросети
pattern.add(Conv2D(32, (3, 3), input_shape=input_shape))
pattern.add(Activation('relu'))
pattern.add(MaxPooling2D(pool_size=(2, 2)))

#второй слой нейросети
pattern.add(Conv2D(32, (3, 3)))
pattern.add(Activation('relu'))
pattern.add(MaxPooling2D(pool_size=(2, 2)))

#третий слой нейросети
pattern.add(Conv2D(64, (3, 3)))
pattern.add(Activation('relu'))
pattern.add(MaxPooling2D(pool_size=(2, 2)))

#активация, свертка, объединение, исключение
pattern.add(Flatten())
pattern.add(Dense(64))
pattern.add(Activation('relu'))
pattern.add(Dropout(0.5))
pattern.add(Dense(4))# число классов
pattern.add(Activation('softmax'))

#компиляция модели
pattern.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#параметры аугментации
train_datagen = ImageDataGenerator(rescale=1. / 255)
    #rescale=1. / 255, #коэффициент масштабирования
    #shear_range=0.2, #интенсивность сдвига
    #zoom_range=0.2, #диапазон случайного увеличения
    #horizontal_flip=True) #произвольный поворот по горизонтали
test_datagen = ImageDataGenerator(rescale=1. / 255)

#предобработка обучающей выборки
train_processing = train_datagen.flow_from_directory(
    directory_data_train,
    classes=("neprovar", "no_diff", "pora", "trechina"),
    target_size=(image_width, image_height), #размер изображений
    batch_size=lot_size, #размер пакетов данных
    class_mode='categorical') #режим класса

#предобработка тестовой выборки
validation_processing = test_datagen.flow_from_directory(
    directory_data_validation,
    classes=("neprovar", "no_diff", "pora", "trechina"),
    target_size=(image_width, image_height),
    batch_size=lot_size,
    class_mode='categorical')

pattern.fit(
    train_processing, #обучающая выборка
    steps_per_epoch=train_sample // lot_size, #количество итераций пакета до того, как период обучения считается завершенным
    epochs=epochs, #количество эпох
    validation_data=validation_processing, #проверочная выборка
    validation_steps=validation_sample  // lot_size) #количество итерации, но на проверочном пакете данных

#pattern.save_weights('/content/drive/MyDrive/ВУЗ/аспирантура/neironka/first_model_weights.h5') #сохранение весов модели
#pattern.save('/content/drive/MyDrive/ВУЗ/аспирантура/neironka/') #сохранение модели
#pattern.load_weights('/content/neironka/first_model_weights.h5') #загрузка весов модели
img = Image.open('/content/drive/MyDrive/ВУЗ/аспирантура/dataset/val/neprovar/1_24_11zon.png')
obr_img = np.asarray(img)
obr_img = (np.expand_dims (obr_img, 0))
prediction = pattern.predict(obr_img) #использование модели для предсказания
print('neprovar|no_diff|pora|trechina')
print(prediction)

img = Image.open('/content/drive/MyDrive/ВУЗ/аспирантура/dataset/val/no_diff/1_23_11zon.png')
obr_img = np.asarray(img)
obr_img = (np.expand_dims (obr_img, 0))
prediction = pattern.predict(obr_img) #использование модели для предсказания
print('neprovar|no_diff|pora|trechina')
print(prediction)

img = Image.open('/content/drive/MyDrive/ВУЗ/аспирантура/dataset/val/pora/1_20_11zon.png')
obr_img = np.asarray(img)
obr_img = (np.expand_dims (obr_img, 0))
prediction = pattern.predict(obr_img) #использование модели для предсказания
print('neprovar|no_diff|pora|trechina')
print(prediction)

img = Image.open('/content/drive/MyDrive/ВУЗ/аспирантура/dataset/val/trechina/1_20_11zon.png')
obr_img = np.asarray(img)
obr_img = (np.expand_dims (obr_img, 0))
prediction = pattern.predict(obr_img) #использование модели для предсказания
print('neprovar|no_diff|pora|trechina')
print(prediction)
