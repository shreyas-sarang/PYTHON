# Code
```
import cv2
import matplotlib.pylab as plt
from google.colab.patches import cv2_imshow

img = cv2.imread("/content/pcb.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=[9,8])                       #specifying the width and height of the figure in inches.
plt.subplot(1,1,1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
```
