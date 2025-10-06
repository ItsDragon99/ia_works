import cv2 as cv
img = cv.imread("./test_imgs/ITM2.webp", 1)
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
ubb = (0,60,60)
uba = (10,255,255)
ubb1 = (170,60,60)
ubb2 = (180,255,255)
mask1 = cv.inRange(hsv,ubb,uba)
mask2 = cv.inRange(hsv, ubb1, ubb2)
# Con la mascara se puede hacer operaciones bit a bit
# Esto quiere decir que se puede identificar el objeto  que se desea
# y aislarlo del resto de la imagen
mask = mask1 + mask2
resultado = cv.bitwise_and(img,img ,mask=mask)
cv.imshow('resultado',resultado)
cv.imshow('mask',mask)
cv.imshow('img',img)
cv.imshow('hsv',hsv)
cv.waitKey(0)
cv.destroyAllWindows()