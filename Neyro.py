import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Размер изображения
IMG_HEIGHT = 987
IMG_WIDTH = 1589
IMG_CHANNELS = 3  # Цветные изображения (RGB)

train_dir = '' 
val_dir = '' 

train_datagen = ImageDataGenerator(
    rescale=1./255,        
    rotation_range=40,     
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,      
    zoom_range=0.2,       
    horizontal_flip=True,  
    fill_mode='nearest'    
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='categorical'  # Множественная классификация
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='categorical'
)

# Построение модели
model = models.Sequential([
    # Свёрточные слои для извлечения признаков
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # Слои для обучения более сложных признаков
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Dropout для предотвращения переобучения

    # Выходной слой для классификации
    layers.Dense(4, activation='softmax')  # 4 класса
])

# Компиляция модели
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Обучение модели
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Сохранение модели
model.save('defect_classifier.h5')

# Оценка модели
test_loss, test_acc = model.evaluate(validation_generator)
print(f"Test accuracy: {test_acc}")

# Построение графиков обучения
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legends(loc='lower right')
plt.show()
