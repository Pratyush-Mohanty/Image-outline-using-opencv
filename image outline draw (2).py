# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:23:28 2020

@author: Pratyush
"""


import cv2
import sys

m=float(input("input variance for guassian "))
n=float(input("input sigma for cannny"))
# ideal value for m is 2 and n is 1.5 but can be varied 
#imagePath = sys.argv[1]
#use the above image path if it shows error otherwise ignore
image = cv2.imread(r"C:\Users\Pratyush\Desktop\New folder\images.jfif")
#path for the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)

print("[INFO] Found {0} Faces.".format(len(faces)))
#searches for the image from the folder or file system

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_color = image[y:y + h, x:x + w]
    print("[INFO] Object found. Saving locally.")
    cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)

status = cv2.imwrite(r"C:\Users\Pratyush\Desktop\New folder\IMG20200427115253.jpg", image)
#image saved in another location (say loc2) for further operation  
print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

cv2.imshow( r"C:\Users\Pratyush\Desktop\New folder\IMG20200427115253.jpg", image)
cv2.waitKey(5000)
#you can see image in taskbar for 5 seconds 
# if want to ignore just put a #before cv2.waitkey(5000) 







img = cv2.imread(r"C:\Users\Pratyush\Desktop\New folder\IMG20200427115253.jpg")
#read from loc2
crop_img = img[y:y+h ,x:x+w]
cv2.imshow("cropped", crop_img)
cv2.waitKey(5000)
#can see the cropped image in taskbar for 5 sec, if want to ignore just put a #before cv2.waitkey(5000) 
#cropping image and saved in one location (say loc 3 or you can replace loc 1 but loc 3 is better)
cv2.imwrite(r"C:\Users\Pratyush\Desktop\New folder\Question_2.jpg",crop_img)



cv2.destroyAllWindows()




im=cv2.imread(r"C:\Users\Pratyush\Desktop\New folder\Question_2.jpg",0)
#read from loc3
import numpy as np
from scipy import ndimage
im = ndimage.rotate(im, 0, mode='constant')
im = ndimage.gaussian_filter(im, m)
from skimage import feature
edges2 = feature.canny(im, sigma=n)
import matplotlib.pyplot as plt
plt.imshow(edges2)