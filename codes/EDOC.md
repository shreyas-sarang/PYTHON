# Erosion, Dilation, Opening, Closing code
```
import numpy as np
import matplotlib.pylab as plt  # library used for ploting, graph
import cv2         # library for Open Source Computer vision library
from google.colab.patches import cv2_imshow

img = cv2.imread("pcb.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=[9,8])                       #specifying the width and height of the figure in inches.
plt.subplot(2,3,1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")

# Find size of the input image
m,n= img.shape 

# Define the structuring element
P=5

SE= np.ones((P,P), dtype=np.uint8)
c = (P-1)//2

# Initialize new image for Eroded as imgEr
imgEr = np.zeros((m,n), dtype=np.uint8)
for i in range(c, m-c):
  for j in range(c, n-c):
    temp= img[i-c:i+c+1,  j-c:j+c+1]
    product= temp*SE
    imgEr[i,j]= np.min(product)

plt.subplot(2,3,2)
plt.imshow(imgEr, cmap="gray")
plt.title("Eroded Image")

# Define the structuring element
P=3

SE= np.ones((P,P), dtype=np.uint8)
c = (P-1)//2

# Initialize new image for Dilated as imgDl
imgDl = np.zeros((m,n), dtype=np.uint8)
for i in range(c, m-c):
  for j in range(c, n-c):
    temp= img[i-c:i+c+1,  j-c:j+c+1]
    product= temp*SE
    imgDl[i,j]= np.max(product)

plt.subplot(2,3,3)
plt.imshow(imgDl, cmap="gray")
plt.title("Dilated Image")

# Initialize new image for Opening as img_opening
img_opening = np.zeros((m,n), dtype =np.uint8)
for i in range(c, m-c):
  for j in range(c, n-c):
    temp= imgEr[i-c:i+c+1,  j-c:j+c+1]
    product= temp*SE
    img_opening[i,j]= np.max(product)

plt.subplot(2,3,4)
plt.imshow(img_opening, cmap="gray")
plt.title("Opening Image")

# Initialize new image for Opening as img_closing
img_closing = np.zeros((m,n), dtype =np.uint8)
for i in range(c, m-c):
  for j in range(c, n-c):
    temp= imgDl[i-c:i+c+1,  j-c:j+c+1]
    product= temp*SE
    img_closing[i,j]= np.min(product)

plt.subplot(2,3,5)
plt.imshow(img_closing, cmap="gray")
plt.title("Closing Image")
```
