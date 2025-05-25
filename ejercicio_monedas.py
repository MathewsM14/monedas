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
    img_morf = cv2.erode(imgbin_inv, np.ones((5, 5), np.uint8), iterations=10)
    
    # Componentes conectadas 
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_morf)
    
    # Crear imagen coloreada para visualizar etiquetas
    labeled_img = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    contador_monedas = 0

    for label in range(1, num_labels):  # Ignorar fondo (etiqueta 0)
        color = np.random.randint(0, 255, size=3)
        labeled_img[labels == label] = color

        # Obtener centroide y escribir número
        cX, cY = int(centroids[label][0]), int(centroids[label][1])
        cv2.putText(labeled_img, str(label), (cX - 10, cY + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        contador_monedas += 1

    # Mostrar la imagen etiquetada
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"{ruta_imagen} - Monedas detectadas: {contador_monedas}")
    plt.show()
    
    # Guardar imagen etiquetada
    cv2.imwrite(salida, labeled_img)
    print(f"[{ruta_imagen}] Número de monedas detectadas: {contador_monedas}")
    
    return labels


procesar_imagen('monedas.jpg', escala=0.2, salida='resultado_moneda_1.png')
procesar_imagen('mi_moneda.jpg', escala=0.4, salida='resultado_moneda_2.png')

