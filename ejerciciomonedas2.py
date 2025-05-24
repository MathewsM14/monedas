import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('monedas.jpg')
img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convertir a gris
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# *** AQUÍ VA LA CORRECCIÓN MORFOLÓGICA ***
# Crear kernel para eliminar brillos
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))

# Obtener fondo con brillos
background = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)

# Sustraer el fondo para eliminar brillos
img_gray_sin_brillo = cv2.add(img_gray, background)

# Normalizar
img_gray_sin_brillo = cv2.normalize(img_gray_sin_brillo, None, 128, 255, cv2.NORM_MINMAX)

# *** CONTINÚA CON TU CÓDIGO NORMAL ***
# Ahora usa img_gray_sin_brillo en lugar de img_gray

# Suavizado
blur = cv2.GaussianBlur(img_gray_sin_brillo, (3, 3), 0)

# Binarización
_, imgbin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imgbin2 = 255 - imgbin
imgbin2_erosion = cv2.erode(imgbin2, np.ones((5, 5), np.uint8), iterations=5)

# Componentes conectados
_, markers = cv2.connectedComponents(imgbin2_erosion)

# Mostrar resultado
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_gray_sin_brillo, cmap='gray')
plt.title('Sin Brillo')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(markers, cmap='nipy_spectral', vmin=0, vmax=24)
plt.title('Componentes Detectados')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Número de componentes únicos detectados: {len(np.unique(markers))}")