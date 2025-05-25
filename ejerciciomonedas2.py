import cv2
import matplotlib.pyplot as plt
import numpy as np

def procesar_imagen(ruta_imagen, escala=0.4, salida='resultado.png'):
    # Cargar y redimensionar imagen
    img = cv2.imread(ruta_imagen)
    img = cv2.resize(img, (0, 0), fx=escala, fy=escala)

    # Suavizado y conversión a escala de grises
    img_blur = cv2.GaussianBlur(img, (7, 7), 0)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    # Binarización y morfología
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_inv = 255 - img_bin
    img_morf = cv2.erode(img_inv, np.ones((5, 5), np.uint8), iterations=5)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_morf)

    # Crear imagen coloreada
    labeled_img = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    contador_monedas = 0

    area_min=300
    area_max=4000
    
    for i in range(1, num_labels):  # Saltar fondo
        area = stats[i, cv2.CC_STAT_AREA]
        if area_min <= area <= area_max:
            contador_monedas += 1
            mask = labels == i
            color = np.random.randint(0, 255, size=3)
            labeled_img[mask] = color

            x, y = centroids[i]
            cv2.putText(labeled_img, str(contador_monedas), (int(x) - 10, int(y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Mostrar resultado
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.title(f"{ruta_imagen} - Monedas detectadas: {contador_monedas}")
    plt.axis('off')
    plt.show()

    cv2.imwrite(salida, labeled_img)
    print(f"[{ruta_imagen}] Monedas válidas detectadas: {contador_monedas}")
    print(f"Imagen guardada como: {salida}")

    return contador_monedas

# Procesar ambas imágenes con filtro de tamaño
procesar_imagen('monedas.jpg', escala=0.3, salida='resultado_moneda_1.png')
procesar_imagen('mi_moneda.jpg', escala=0.4, salida='resultado_moneda_2.png')
