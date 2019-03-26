import cv2
import numpy as np
import math

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


img = cv2.imread('img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
while 1:

    faces = cascade.detectMultiScale(img, 1.01, 7)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# if(cascade.detectMultiScale(gray, 1.01, 7).any()):
#     print("Found")
cv2.destroyAllWindows()
