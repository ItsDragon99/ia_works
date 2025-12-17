import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU, BatchNormalization
)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4000)]  
        )
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

IMG_SIZE   = 128
BATCH_SIZE = 32          
EPOCHS     = 35          
INIT_LR    = 1e-3
SEED       = 42
DATA_DIR = '../dataset/animals/v2'
MODEL_PATH = "../models/animals/animal_cnn_keras.h5"
BEST_MODEL_PATH = "../models/animals/animal_cnn_keras_best.h5"

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

valid_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

valid_generator = valid_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,    
    seed=SEED
)

nClasses = train_generator.num_classes
class_indices = train_generator.class_indices
print("Clases:", class_indices)
os.makedirs("../models/animals", exist_ok=True)
with open("../models/animals/class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=4)


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 activation='linear',
                 input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same', activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.35))
model.add(Conv2D(128, (3, 3), padding='same', activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(nClasses, activation='softmax'))
model.summary()
optimizer = keras.optimizers.Adam(learning_rate=INIT_LR)
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]
try:
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=valid_generator,
        callbacks=callbacks,
        verbose=1
    )
except KeyboardInterrupt:
    print("Entrenamiento interrumpido por el usuario.")
finally:
    model.save(MODEL_PATH)
    print(f"Modelo final guardado en {MODEL_PATH}")
    print(f"Mejor modelo (según val_accuracy) guardado en {BEST_MODEL_PATH}")

valid_generator.reset()
y_true = valid_generator.classes
y_prob = model.predict(valid_generator, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

print("Classification Report (Validación):")
print(classification_report(
    y_true,
    y_pred,
    target_names=list(class_indices.keys())
))
acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)
plt.figure(figsize=(8, 4))
plt.plot(epochs_range, acc, 'o-', label='Training accuracy')
plt.plot(epochs_range, val_acc, 'o-', label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)

plt.figure(figsize=(8, 4))
plt.plot(epochs_range, loss, 'o-', label='Training loss')
plt.plot(epochs_range, val_loss, 'o-', label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)

plt.show()
