import os

import cv2
import numpy as np


path_datos = "Dataset"
path_salida = "Dataset_aumentado"
extensiones_validas = (".jpg", ".jpeg", ".png")


def rotar_imagen(imagen, angulo):
    alto, ancho = imagen.shape[:2]
    centro = (ancho // 2, alto // 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    return cv2.warpAffine(
        imagen,
        matriz,
        (ancho, alto),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def ajustar_brillo(imagen, factor):
    imagen_float = imagen.astype(np.float32) * factor
    return np.clip(imagen_float, 0, 255).astype(np.uint8)


def obtener_imagenes_originales(carpeta):
    archivos = []
    for archivo in os.listdir(carpeta):
        nombre, extension = os.path.splitext(archivo)
        if extension.lower() not in extensiones_validas:
            continue
        if "_flip" in nombre or "_rot" in nombre or "_bright" in nombre:
            continue
        archivos.append(archivo)
    return sorted(archivos)


total_generadas = 0
total_copiadas = 0

os.makedirs(path_salida, exist_ok=True)

for clase in sorted(os.listdir(path_datos)):
    path_clase = os.path.join(path_datos, clase)
    if not os.path.isdir(path_clase):
        continue

    path_clase_salida = os.path.join(path_salida, clase)
    os.makedirs(path_clase_salida, exist_ok=True)

    imagenes = obtener_imagenes_originales(path_clase)
    if not imagenes:
        continue

    generadas_clase = 0
    copiadas_clase = 0
    for archivo in imagenes:
        ruta_imagen = os.path.join(path_clase, archivo)
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            continue

        nombre_base, _ = os.path.splitext(archivo)

        ruta_original_salida = os.path.join(path_clase_salida, archivo)
        if not os.path.exists(ruta_original_salida):
            cv2.imwrite(ruta_original_salida, imagen)
            copiadas_clase += 1
            total_copiadas += 1

        variantes = {
            f"{nombre_base}_flip.jpg": cv2.flip(imagen, 1),
            f"{nombre_base}_rot_pos.jpg": rotar_imagen(imagen, 10),
            f"{nombre_base}_rot_neg.jpg": rotar_imagen(imagen, -10),
            f"{nombre_base}_bright_up.jpg": ajustar_brillo(imagen, 1.2),
            f"{nombre_base}_bright_down.jpg": ajustar_brillo(imagen, 0.8),
        }

        for nuevo_nombre, nueva_imagen in variantes.items():
            ruta_salida_archivo = os.path.join(path_clase_salida, nuevo_nombre)
            if os.path.exists(ruta_salida_archivo):
                continue
            cv2.imwrite(ruta_salida_archivo, nueva_imagen)
            generadas_clase += 1
            total_generadas += 1

    print(
        f"{clase}: {copiadas_clase} originales copiadas, "
        f"{generadas_clase} imagenes aumentadas generadas"
    )

print(f"Proceso finalizado. Total de originales copiadas: {total_copiadas}")
print(f"Proceso finalizado. Total de imagenes aumentadas generadas: {total_generadas}")
