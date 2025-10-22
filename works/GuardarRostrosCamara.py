import os
import cv2 as cv
import numpy as np
import math

def seleccionar_carpeta_dataset(dataset_dir='./dataset/rostros'):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    carpetas = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    print("carpetas de rostros:", carpetas)
    for i, carpeta in enumerate(carpetas, start=1):
        print(f"{i} .- {carpeta}")
    print("0 .- Crear nueva carpeta")

    while True:
        input_usuario = input("seleccione el numero de la carpeta ").strip()
        if not input_usuario.isdigit():
            print("no hay carpeta con ese numero, hazlo denuevo")
            continue
        opcion = int(input_usuario)
        if opcion == 0:
            nombre_carpeta = input("nueva carpeta?  ").strip()
            if not nombre_carpeta:
                print("nombre mal escroto")
                continue
            if nombre_carpeta in carpetas:
                print("la carpeta ya existe bro")
                continue
            ruta = os.path.join(dataset_dir, nombre_carpeta)
            os.makedirs(ruta)
            print("se acaba de crear la carpeta:", nombre_carpeta)
            return ruta
        elif 1 <= opcion <= len(carpetas):
            return os.path.join(dataset_dir, carpetas[opcion - 1])
        else:
            print("no sabes leer we? hazlo denuevo")

    
def conteo_imagenes(carpeta):
    if not os.path.exists(carpeta):
        return 0
    archivos = [f for f in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, f))]
    return len(archivos)

carpeta_seleccionada = seleccionar_carpeta_dataset('./dataset/rostros')
rostro = cv.CascadeClassifier('./xml/haarcascade_frontalface_alt.xml')
cap = cv.VideoCapture(0)
i = 0  
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    rostros = rostro.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in rostros:
        #frame = cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        frame2 = frame[ y:y+h, x:x+w]
        # #frame3 = frame[x+30:x+w-30, y+30:y+h-30]
        frame2 = cv.resize(frame2, (100, 100), interpolation=cv.INTER_AREA)
        # cv.imshow('rostror', frame2)

        if conteo_imagenes(carpeta_seleccionada) % 100 == 0:
            print("imagenes en la carpeta:", conteo_imagenes(carpeta_seleccionada))
        if (i % 10 == 0 and conteo_imagenes(carpeta_seleccionada) < 1000):
            cv.imwrite(carpeta_seleccionada + '/' + str(i) + '.jpg', frame2)
            cv.imshow('Rostro', frame2)
        else:
            break
        cv.imshow('Camara', frame)


    i = i + 1
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()