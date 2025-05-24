import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import os

class CoinCounter:
    def __init__(self):
        self.results = {}
    
    def load_image(self, image_path):
        """Carga una imagen desde el archivo especificado"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Convertir de BGR a RGB para matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    
    def preprocess_image(self, img, resize_factor=0.5):
        """
        Preprocesa la imagen aplicando:
        - Redimensionamiento
        - Suavizado
        - Conversión a escala de grises
        - Binarización
        - Operaciones morfológicas
        """
        # Redimensionar imagen para reducir resolución
        height, width = img.shape[:2]
        new_height = int(height * resize_factor)
        new_width = int(width * resize_factor)
        img_resized = cv2.resize(img, (new_width, new_height))
        
        # Aplicar suavizado gaussiano
        img_blur = cv2.GaussianBlur(img_resized, (7, 7), 0)
        
        # Convertir a escala de grises
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
        
        # Binarización usando umbral de Otsu
        _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invertir la imagen binaria (monedas en blanco, fondo en negro)
        img_binary_inv = 255 - img_binary
        
        # Operaciones morfológicas para limpiar ruido y separar objetos
        # Apertura para eliminar ruido pequeño
        kernel_opening = np.ones((3, 3), np.uint8)
        img_opened = cv2.morphologyEx(img_binary_inv, cv2.MORPH_OPEN, kernel_opening, iterations=2)
        
        # Erosión para separar monedas que puedan estar conectadas
        kernel_erosion = np.ones((5, 5), np.uint8)
        img_eroded = cv2.erode(img_opened, kernel_erosion, iterations=5)
        
        return {
            'original': img_resized,
            'gray': img_gray,
            'binary': img_binary,
            'binary_inv': img_binary_inv,
            'processed': img_eroded
        }
    
    def detect_connected_components(self, binary_img, min_area=100, max_area=5000):
        """
        Detecta componentes conectadas y filtra por área para identificar monedas
        """
        # Encontrar componentes conectadas
        num_labels, labels = cv2.connectedComponents(binary_img)
        
        # Calcular estadísticas de cada componente
        stats = cv2.connectedComponentsWithStats(binary_img, connectivity=8)[2]
        
        valid_coins = []
        valid_labels = []
        
        # Filtrar componentes por área (excluir fondo que es label 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                valid_coins.append(i)
                valid_labels.append(labels == i)
        
        return labels, valid_coins, len(valid_coins)
    
    def visualize_results(self, original_img, labels, valid_coins, image_name):
        """
        Visualiza los resultados del procesamiento
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Análisis de Monedas - {image_name}', fontsize=16)
        
        # Imagen original
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Imagen Original Redimensionada')
        axes[0, 0].axis('off')
        
        # Componentes conectadas coloreadas
        colored_labels = np.zeros_like(labels)
        for i, coin_label in enumerate(valid_coins):
            colored_labels[labels == coin_label] = coin_label
        
        axes[0, 1].imshow(colored_labels, cmap='nipy_spectral')
        axes[0, 1].set_title(f'Componentes Conectadas ({len(valid_coins)} monedas)')
        axes[0, 1].axis('off')
        
        # Imagen con monedas marcadas
        result_img = original_img.copy()
        overlay = np.zeros_like(original_img)
        
        for coin_label in valid_coins:
            mask = (labels == coin_label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Dibujar contorno
                cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
                
                # Encontrar centroide para etiquetar
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(overlay, str(coin_label), (cx-10, cy+5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        result_with_overlay = cv2.addWeighted(result_img, 0.7, overlay, 0.3, 0)
        axes[1, 0].imshow(result_with_overlay)
        axes[1, 0].set_title('Monedas Detectadas y Etiquetadas')
        axes[1, 0].axis('off')
        
        # Estadísticas
        axes[1, 1].text(0.1, 0.8, f'Total de monedas detectadas: {len(valid_coins)}', 
                       fontsize=14, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'Componentes totales encontradas: {np.max(labels)}', 
                       fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f'Resolución procesada: {original_img.shape[1]}x{original_img.shape[0]}', 
                       fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Estadísticas')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def process_image(self, image_path, resize_factor=0.5, min_area=100, max_area=5000):
        """
        Procesa una imagen completa siguiendo todo el pipeline
        """
        print(f"\n=== Procesando: {image_path} ===")
        
        # 1. Cargar imagen
        img = self.load_image(image_path)
        print(f"✓ Imagen cargada: {img.shape}")
        
        # 2. Preprocesamiento
        processed_imgs = self.preprocess_image(img, resize_factor)
        print(f"✓ Preprocesamiento completado, nueva resolución: {processed_imgs['processed'].shape}")
        
        # 3. Detección de componentes conectadas
        labels, valid_coins, coin_count = self.detect_connected_components(
            processed_imgs['processed'], min_area, max_area
        )
        print(f"✓ Componentes conectadas detectadas: {np.max(labels)}")
        print(f"✓ Monedas válidas encontradas: {coin_count}")
        
        # 4. Visualización
        image_name = os.path.basename(image_path)
        fig = self.visualize_results(processed_imgs['original'], labels, valid_coins, image_name)
        
        # Guardar resultado
        output_path = f"resultado_{image_name.split('.')[0]}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Resultado guardado en: {output_path}")
        
        # Almacenar resultados
        self.results[image_path] = {
            'coin_count': coin_count,
            'total_components': np.max(labels),
            'processed_shape': processed_imgs['processed'].shape
        }
        
        plt.show()
        return coin_count
    
    def process_multiple_images(self, image_paths, **kwargs):
        """
        Procesa múltiples imágenes
        """
        total_coins = 0
        
        for image_path in image_paths:
            try:
                coins = self.process_image(image_path, **kwargs)
                total_coins += coins
            except Exception as e:
                print(f"❌ Error procesando {image_path}: {str(e)}")
        
        print(f"\n=== RESUMEN FINAL ===")
        print(f"Total de imágenes procesadas: {len(self.results)}")
        print(f"Total de monedas detectadas: {total_coins}")
        
        for image_path, result in self.results.items():
            print(f"  - {os.path.basename(image_path)}: {result['coin_count']} monedas")
        
        return self.results
    
    
counter = CoinCounter()
coin_count = counter.process_image('monedas.jpg')