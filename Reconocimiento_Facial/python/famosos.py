import os
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

# 1. Configuración principal
path_datos = 'Dataset'

print("Descargando el dataset LFW de internet... (Esto puede tomar un par de minutos)")

# 2. Descargar LFW
lfw = fetch_lfw_people(min_faces_per_person=70, color=True)

print("¡Descarga completada! Procesando imágenes...")

# 3. Procesar y guardar cada imagen
for i, image_array in enumerate(lfw.images):
    nombre_famoso = lfw.target_names[lfw.target[i]].replace(" ", "_")
    
    path_completo = os.path.join(path_datos, nombre_famoso)
    if not os.path.exists(path_completo):
        os.makedirs(path_completo)
        print(f"Creando carpeta para: {nombre_famoso}")
    
    count = len(os.listdir(path_completo))
    
    if count >= 200:
        continue 
        
    # --- CORRECCIÓN DE COLOR AQUÍ ---
    # Multiplicar por 255 para restaurar la escala de color normal
    if image_array.max() <= 1.0:
        image_array = image_array * 255.0
        
    img_bgr = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    # --------------------------------
    
    img_redimensionada = cv2.resize(img_bgr, (160, 160), interpolation=cv2.INTER_CUBIC)
    
    ruta_archivo = os.path.join(path_completo, f'rostro_{count}.jpg')
    cv2.imwrite(ruta_archivo, img_redimensionada)

print("¡Proceso finalizado! Revisa tu carpeta 'Dataset'.")