import cv2 as cv
import numpy as np
img = cv.imread("./test_imgs/figuras.png", 1)


#* Leer la imagen en escala de grises
img_gris = cv.imread("./test_imgs/figuras.png", 0)
alturaIImagen, anchoImagen = img_gris.shape
print('Alto: ', alturaIImagen)
print('Ancho: ', anchoImagen)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#* Umbral para el color rojo
ubb_rojo = (0,60,60)
uba_rojo = (10,255,255)
ubb1_rojo = (170,60,60)
ubb2_rojo = (180,255,255)

#* Umbral para el color azul
ubb_azul = (100,150,0)
uba_azul = (140,255,255)
ubb1_azul = (100,150,0)
ubb2_azul = (140, 255, 255)

#* Umbral para el color verde
ubb_verde = (64,96,0)
uba_verde = (100,255,255)
ubb1_verde = (42,85,0)
ubb2_verde = (100, 255, 255)

#* Umbral para el color amarillo
ubb_amarillo = (32,64,0)
uba_amarillo = (32,255,255)
ubb1_amarillo = (20,42,0)
ubb2_amarillo = (32, 255, 255)

mask1_rojo = cv.inRange(hsv,ubb_rojo,uba_rojo)
mask2_rojo = cv.inRange(hsv, ubb1_rojo, ubb2_rojo)

mask1_azul = cv.inRange(hsv, ubb_azul, uba_azul)
mask2_azul = cv.inRange(hsv, ubb1_azul, ubb2_azul)

mask1_verde = cv.inRange(hsv, ubb_verde, uba_verde)
mask2_verde = cv.inRange(hsv, ubb1_verde, ubb2_verde)

mask1_amarillo = cv.inRange(hsv, ubb_amarillo, uba_amarillo)
mask2_amarillo = cv.inRange(hsv, ubb1_amarillo, ubb2_amarillo)

#!Con la mascara se puede hacer operaciones bit a bit
#! Esto quiere decir que se puede identificar el objeto  que se desea
#! y aislarlo del resto de la imagen
mask_rojo = mask1_rojo + mask2_rojo
resultado_rojo = cv.bitwise_and(img, img, mask=mask_rojo)

coord_mask_rojo = cv.findNonZero(mask_rojo)
print(coord_mask_rojo)

mask_azul = mask1_azul + mask2_azul
resultado_azul = cv.bitwise_and(img, img, mask=mask_azul)


mask_verde = mask1_verde + mask2_verde
resultado_verde = cv.bitwise_and(img, img, mask=mask_verde)

mask_amarillo = mask1_amarillo + mask2_amarillo
resultado_amarillo = cv.bitwise_and(img, img, mask=mask_amarillo)

# cv.imshow('Figuras Rojas',resultado_rojo)
# cv.imshow('Mascara Figuras Rojas',mask_rojo)
# cv.imshow('Figuras Azules',resultado_azul)
# cv.imshow('Mascara Figuras Azules',mask_azul)
# cv.imshow('Figuras Verdes',resultado_verde)
# cv.imshow('Mascara Figuras Verdes',mask_verde)
# cv.imshow('Figuras Amarillas',resultado_amarillo)
# cv.imshow('Mascara Figuras amarillas', mask_amarillo)

# cv.imshow('Imagen en gris', img_gris)

# cv.imshow('img',img)
# cv.imshow('hsv',hsv)
cv.waitKey(0)
cv.destroyAllWindows()