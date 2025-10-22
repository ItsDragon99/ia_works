import cv2 as cv 
import numpy as np 
import os
dataSet = '../dataset/rostros'
faces = os.listdir(dataSet)
print(faces)
faces_banned = ['ger', 'eddy']

labels = []
facesData = []
label = 0 
for face in faces:
    facePath = dataSet + '/' + face
    if face in faces_banned:
        for faceName in os.listdir(facePath):
            labels.append(label)
            facesData.append(cv.imread(facePath + '/' + faceName, 0))
        label = label + 1
print(np.count_nonzero(np.array(labels)==0)) 

faceRecognizer = cv.face.EigenFaceRecognizer_create()
print("Entrenando...")
faceRecognizer.train(facesData, np.array(labels))
print("Entrenamiento completado.")
faceRecognizer.write('../xml/Eigenface.xml')