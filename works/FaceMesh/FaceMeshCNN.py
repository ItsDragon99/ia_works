import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU
import keras
import mediapipe as mp

# --- Configuración de MediaPipe FaceMesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(234, 255, 233))

# --- Cargar imágenes y procesarlas con FaceMesh ---
dirname = os.path.join(os.getcwd(), '../dataset/emociones')
imgpath = dirname + os.sep

images = []
directories = []
dircount = []
prevRoot = ''
cant = 0
print("leyendo imagenes de ", imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant += 1
            filepath = os.path.join(root, filename)
            image = cv2.imread(filepath)
            if image is None:
                continue
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )
            # Redimensionar para la red (ajusta según tu dataset)
            image = cv2.resize(image, (28, 28))
            images.append(image)
            b = "Leyendo... " + str(cant)
            print(b, end="\r")

    if prevRoot != root:
        print(root, cant)
        prevRoot = root
        directories.append(root)
        dircount.append(cant)
        cant = 0

dircount.append(cant)
dircount = dircount[1:]
print('Directorios leidos: ', len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:', sum(dircount))

# --- Crear etiquetas ---
labels = []
indice = 0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice += 1
print("Cantidad etiquetas creadas: ", len(labels))

emociones = []
indice = 0
for directorio in directories:
    name = directorio.split(os.sep)
    print(indice, name[-1])
    emociones.append(name[-1])
    indice += 1

y = np.array(labels)
X = np.array(images, dtype=np.uint8)

classes = np.unique(y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# --- Dividir y preprocesar ---
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2)
print('\nTraining data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

train_X = train_X.astype('float32') / 255.
test_X = test_X.astype('float32') / 255.

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
print('\nFinal shapes:')
print(train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)

# --- Red neuronal convolucional ---
INIT_LR = 1e-3
epochs = 6
batch_size = 64

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(28, 28, 3)))
emotion_model.add(LeakyReLU(alpha=0.1))
emotion_model.add(MaxPooling2D((2, 2), padding='same'))
emotion_model.add(Dropout(0.5))

emotion_model.add(Flatten())
emotion_model.add(Dense(32, activation='linear'))
emotion_model.add(LeakyReLU(alpha=0.1))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(train_Y_one_hot.shape[1], activation='softmax'))

emotion_model.summary()

emotion_model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adagrad(learning_rate=INIT_LR),
    metrics=['accuracy']
)

emotion_train = emotion_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_label))
test_eval = emotion_model.evaluate(test_X, test_Y_one_hot, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

accuracy = emotion_train.history['accuracy']
val_accuracy = emotion_train.history['val_accuracy']
loss = emotion_train.history['loss']
val_loss = emotion_train.history['val_loss']
epochs_range = range(len(accuracy))
plt.plot(epochs_range, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs_range, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

predicted_classes2 = emotion_model.predict(test_X)
predicted_classes = np.argmax(predicted_classes2, axis=1)

correct = np.where(predicted_classes == test_Y)[0]
print("Found %d correct labels" % len(correct))
for i, correct_idx in enumerate(correct[0:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_X[correct_idx].reshape(28, 28, 3))
    plt.title("{}, {}".format(emociones[predicted_classes[correct_idx]], emociones[test_Y[correct_idx]]))
    plt.tight_layout()

incorrect = np.where(predicted_classes != test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect_idx in enumerate(incorrect[0:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_X[incorrect_idx].reshape(28, 28, 3))
    plt.title("{}, {}".format(emociones[predicted_classes[incorrect_idx]], emociones[test_Y[incorrect_idx]]))
    plt.tight_layout()

target_names = ["Class {}".format(i) for i in range(nClasses)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))
