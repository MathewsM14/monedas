import cv2
import matplotlib.pyplot as plt
import numpy as np

def procesar_imagen(ruta_imagen, escala=0.4, salida='resultado.png'):
    # Cargar y redimensionar imagen
    img = cv2.imread(ruta_imagen)
    img = cv2.resize(img, (0, 0), fx=escala, fy=escala)
    
    # Suavizado y conversión a escala de grises
    img = cv2.GaussianBlur(img, (7, 7), 0)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Binarización y morfología
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, imgbin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imgbin_inv = 255 - imgbin
    imgbin_morf = cv2.erode(imgbin_inv, np.ones((5, 5), np.uint8), iterations=5)
    
    # Componentes conectadas
    num_labels, markers = cv2.connectedComponents(imgbin_morf)
    
    # Visualización
    plt.figure(figsize=(10, 5))
    plt.imshow(markers, cmap='nipy_spectral')
    plt.axis('off')
    plt.title(f"{ruta_imagen} - Monedas detectadas: {num_labels - 1}")
    plt.show()
    
    cv2.imwrite(salida, labeled_img)
    print(f"[{ruta_imagen}] Monedas detectadas: {num_labels - 1}")
    print(f"Imagen guardada como: {salida}")

    print(f"[{ruta_imagen}] Número de monedas detectadas: {num_labels - 1}")
    return markers

# Procesar ambas imágenes
procesar_imagen('monedas.jpg', escala=0.3, salida='resultado_moneda_1.png')
procesar_imagen('mi_moneda.jpg', escala=0.4, salida='resultado_moneda_2.png')
