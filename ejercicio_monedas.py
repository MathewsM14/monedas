import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('monedas.jpg')
img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)  # para ajustar el tamaño
img = cv2.GaussianBlur(img, (5, 5), 0)  # para suavizar la imagen

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #Convertir de BGR a RGB
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #Convertir de RGB a GRIS

#Se aplica la binarización y suavizado
blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
_, imgbin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imgbin2 = 255 - imgbin

imgbin2_erosion = cv2.erode(imgbin2, np.ones((5, 5), np.uint8), iterations=5)

_, markers = cv2.connectedComponents(imgbin2_erosion)
np.unique(markers)

plt.figure(figsize=(10, 5))
plt.imshow(markers, cmap='nipy_spectral', vmin=0, vmax=24)
plt.axis('off')