import os
import time
import re

import cv2
from mtcnn import MTCNN


nombre_persona = input("Ingresa el nombre de la persona: ").strip()
if not nombre_persona:
    nombre_persona = "Persona_sin_nombre"

path_datos = "Dataset"
path_completo = os.path.join(path_datos, nombre_persona)

if not os.path.exists(path_completo):
    print(f"Carpeta creada: {path_completo}")
    os.makedirs(path_completo)


def obtener_siguiente_indice(carpeta):
    indices = []
    for archivo in os.listdir(carpeta):
        coincidencia = re.match(r"rostro_(\d+)\.jpg$", archivo)
        if coincidencia:
            indices.append(int(coincidencia.group(1)))
    if not indices:
        return 0
    return max(indices) + 1


detector = MTCNN()
cap = cv2.VideoCapture(0)
count = obtener_siguiente_indice(path_completo)
capturas_iniciales = count
objetivo_fotos = 200
intervalo_captura = 0.4
ultimo_guardado = 0.0

instrucciones = [
    (0, 25, "Mira de frente"),
    (25, 50, "Sonrie ligeramente"),
    (50, 75, "Voltea un poco a la derecha"),
    (75, 100, "Voltea un poco a la izquierda"),
    (100, 125, "Levanta un poco la cabeza"),
    (125, 150, "Baja un poco la cabeza"),
    (150, 175, "Ponte lentes si tienes"),
    (175, 200, "Cambia tu expresion"),
]

print(f"Capturando rostros para {nombre_persona}... Presiona 'q' para detener.")
if count > 0:
    print(f"La carpeta ya tenia {count} imagenes. Se continuara desde ese numero.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    alto_frame, ancho_frame, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = detector.detect_faces(rgb_frame)
    tiempo_actual = time.time()

    progreso_actual = count - capturas_iniciales
    instruccion_actual = "Mantente frente a la camara"
    for inicio, fin, texto in instrucciones:
        if inicio <= progreso_actual < fin:
            instruccion_actual = texto
            break

    for resultado in resultados:
        x, y, w, h = resultado["box"]

        margen_arriba = int(h * 0.40)
        margen_abajo = int(h * 0.10)
        margen_lados = int(w * 0.20)

        y1 = max(0, y - margen_arriba)
        y2 = min(alto_frame, y + h + margen_abajo)
        x1 = max(0, x - margen_lados)
        x2 = min(ancho_frame, x + w + margen_lados)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        rostro_recortado = frame[y1:y2, x1:x2]
        if rostro_recortado.size == 0:
            continue

        rostro_redimensionado = cv2.resize(
            rostro_recortado, (160, 160), interpolation=cv2.INTER_CUBIC
        )

        if tiempo_actual - ultimo_guardado >= intervalo_captura:
            ruta_archivo = os.path.join(path_completo, f"rostro_{count}.jpg")
            cv2.imwrite(ruta_archivo, rostro_redimensionado)
            count += 1
            ultimo_guardado = tiempo_actual
        break

    cv2.rectangle(frame, (10, alto_frame - 70), (ancho_frame - 10, alto_frame - 15), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Instruccion: {instruccion_actual}",
        (20, alto_frame - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Fotos nuevas: {progreso_actual}/{objetivo_fotos}",
        (20, alto_frame - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Captura de Rostros MTCNN", frame)

    if cv2.waitKey(1) & 0xFF == ord("q") or progreso_actual >= objetivo_fotos:
        break

print(f"Proceso finalizado. Se guardaron {count - capturas_iniciales} imagenes nuevas en {path_completo}")
cap.release()
cv2.destroyAllWindows()
