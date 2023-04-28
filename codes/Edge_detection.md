# Edge  Detection
```
import numpy as np       # library used for working with arrays
import matplotlib.pylab as plt  # library used for ploting, graph
import cv2         # library for Open Source Computer vision library
from google.colab.patches import cv2_imshow

img = cv2.imread("building.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=[15,12])                       #specifying the width and height of the figure in inches.
plt.subplot(4,4,1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")

m, n = img.shape

mask = np.ones([3, 3], dtype=int) / 9 #averaging
img_new1 = np.zeros([m, n])

for i in range(1, m-1):
  for j in range(1, n-1):
    temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]
    img_new1[i, j]= temp
img_new1 = img_new1.astype(np.uint8)
plt.subplot(4,4,2)
plt.imshow(img_new1,cmap=plt.cm.gray)
plt.title("Averaging Mask")


mask = np.array([[-1,0],[0,1]]) #roberts1
img_new2 = np.zeros([m, n])


for i in range(1, m-1):
  for j in range(1, n-1):
    temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]
    img_new2[i, j]= temp
img_new2 = img_new2.astype(np.uint8)
plt.subplot(4,4,3)
plt.imshow(img_new2,cmap=plt.cm.gray)
plt.title("Roberts1")


mask = np.array([[0,-1],[1,0]]) #roberts2
img_new3 = np.zeros([m, n])


for i in range(1, m-1):
  for j in range(1, n-1):
    temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]
    img_new3[i, j]= temp
img_new3 = img_new3.astype(np.uint8)
plt.subplot(4,4,4)
plt.imshow(img_new3,cmap=plt.cm.gray)
plt.title("Roberts2")


mask = np.array([[-1,-1,-1],[0,0,0],[1,1,1]]) #prewitt_horizontal
img_new4 = np.zeros([m, n])


for i in range(1, m-1):
  for j in range(1, n-1):
    temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]
    img_new4[i, j]= temp
img_new4 = img_new4.astype(np.uint8)
plt.subplot(4,4,5)
plt.imshow(img_new4,cmap=plt.cm.gray)
plt.title("prewitt_horizontal")


mask = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]) #prewitt_vertical
img_new5 = np.zeros([m, n])


for i in range(1, m-1):
  for j in range(1, n-1):
    temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]
    img_new5[i, j]= temp
img_new5 = img_new5.astype(np.uint8)
plt.subplot(4,4,5)
plt.imshow(img_new5,cmap=plt.cm.gray)
plt.title("prewitt_vertical")


mask = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) #sobel_horizontal
img_new6 = np.zeros([m, n])


for i in range(1, m-1):
  for j in range(1, n-1):
    temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]
    img_new6[i, j]= temp
img_new6 = img_new6.astype(np.uint8)
plt.subplot(4,4,6)
plt.imshow(img_new6,cmap=plt.cm.gray)
plt.title("sobel_horizontal")


mask = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) #sobel_vert
img_new7 = np.zeros([m, n])


for i in range(1, m-1):
  for j in range(1, n-1):
    temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]
    img_new7[i, j]= temp
img_new7 = img_new7.astype(np.uint8)
plt.subplot(4,4,7)
plt.imshow(img_new7,cmap=plt.cm.gray)
plt.title("sobel_vertical")


mask = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]]) #horizontal_2nd order
img_new8 = np.zeros([m, n])


for i in range(1, m-1):
  for j in range(1, n-1):
    temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]
    img_new8[i, j]= temp
img_new8 = img_new8.astype(np.uint8)
plt.subplot(4,4,8)
plt.imshow(img_new8,cmap=plt.cm.gray)
plt.title("horizontal 2nd order")


mask = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]]) #45
img_new9 = np.zeros([m, n])


for i in range(1, m-1):
  for j in range(1, n-1):
    temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]
    img_new9[i, j]= temp
img_new9 = img_new9.astype(np.uint8)
plt.subplot(4,4,9)
plt.imshow(img_new9,cmap=plt.cm.gray)
plt.title("horizontal 2nd order")


mask = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]]) #vertical_2nd order
img_new10 = np.zeros([m, n])


for i in range(1, m-1):
  for j in range(1, n-1):
    temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]
    img_new10[i, j]= temp
img_new10 = img_new10.astype(np.uint8)
plt.subplot(4,4,10)
plt.imshow(img_new10,cmap=plt.cm.gray)
plt.title("vertical 2nd order")


mask = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]]) #-45
img_new11 = np.zeros([m, n])


for i in range(1, m-1):
  for j in range(1, n-1):
    temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]
    img_new11[i, j]= temp
img_new11 = img_new11.astype(np.uint8)
plt.subplot(4,4,11)
plt.imshow(img_new11,cmap=plt.cm.gray)
plt.title("-45")


mask = np.array([[0,1,0],[1,-4, 1],[0,1,0]]) # 2nd order mask
img_new12 = np.zeros([m, n])


for i in range(1, m-1):
  for j in range(1, n-1):
    temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]
    img_new12[i, j]= temp
img_new12 = img_new12.astype(np.uint8)
plt.subplot(4,4,12)
plt.imshow(img_new12,cmap=plt.cm.gray)
plt.title("2nd order mask")
```
