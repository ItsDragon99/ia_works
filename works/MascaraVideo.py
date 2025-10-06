import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)

ubb = (0, 100, 100)
uba = (10, 255, 255)
while True:
    ret, frame = cap.read()
    # cv.imshow('Fotograma', frame)
    # gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gris', gris)
    # rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # cv.imshow('RGB', rgb)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # cv.imshow('HSV', hsv)
    mascara = cv.inRange(hsv, ubb, uba)
    resultado = cv.bitwise_and(frame, frame, mask=mascara)
    cv.imshow('Mascara', mascara)
    cv.imshow('Resultado', resultado)
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()