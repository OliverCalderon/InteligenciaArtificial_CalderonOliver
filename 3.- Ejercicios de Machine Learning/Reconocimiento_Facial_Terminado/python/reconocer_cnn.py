import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "modelo_cnn" / "mejor_modelo.keras"
DEFAULT_LABELS = PROJECT_ROOT / "modelo_cnn" / "labels.json"
IMAGE_SIZE = (160, 160)


def importar_tensorflow():
    try:
        import tensorflow as tf
    except ImportError:
        print("Falta la dependencia 'tensorflow'.")
        print("Instalala con: py -3.13 -m pip install -r requirements.txt")
        sys.exit(1)
    return tf


def importar_mtcnn():
    try:
        from mtcnn import MTCNN
    except ImportError:
        print("Falta la dependencia 'mtcnn'.")
        print("Instalala con: py -3.13 -m pip install -r requirements.txt")
        sys.exit(1)
    return MTCNN


def crear_parser():
    parser = argparse.ArgumentParser(
        description="Reconoce rostros usando el modelo CNN entrenado."
    )
    parser.add_argument(
        "--modelo",
        default=str(DEFAULT_MODEL),
        help="Ruta del modelo .keras entrenado.",
    )
    parser.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS),
        help="Ruta del archivo labels.json.",
    )
    parser.add_argument(
        "--imagen",
        help="Ruta de una imagen para reconocer. Si se omite, se usa la webcam.",
    )
    parser.add_argument(
        "--imagen-recortada",
        action="store_true",
        help="Usa la imagen completa como rostro, sin detector. Util para imagenes del dataset.",
    )
    parser.add_argument(
        "--camara",
        type=int,
        default=0,
        help="Indice de la camara. Por defecto: 0",
    )
    parser.add_argument(
        "--umbral",
        type=float,
        default=0.62,
        help="Confianza minima para mostrar una clase. Por defecto: 0.62",
    )
    parser.add_argument(
        "--margen",
        type=float,
        default=0.22,
        help="Diferencia minima entre top 1 y top 2. Por defecto: 0.22",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Cantidad de predicciones a mostrar por imagen. Por defecto: 3",
    )
    parser.add_argument(
        "--suavizado",
        type=int,
        default=8,
        help="Frames usados para promediar predicciones en webcam. Por defecto: 8",
    )
    parser.add_argument(
        "--reset-iou",
        type=float,
        default=0.25,
        help="Reinicia suavizado si el rostro cambia de posicion. Por defecto: 0.25",
    )
    parser.add_argument(
        "--detector",
        choices=["auto", "mtcnn", "haar"],
        default="auto",
        help="Detector de rostro para reconocimiento. Por defecto: auto",
    )
    parser.add_argument(
        "--mostrar-top",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Muestra las mejores predicciones en la ventana de webcam. Por defecto: activado",
    )
    return parser


def cargar_labels(labels_path):
    labels_path = Path(labels_path)
    if not labels_path.is_file():
        print(f"No existe el archivo de etiquetas: {labels_path}")
        sys.exit(1)

    with labels_path.open("r", encoding="utf-8") as archivo:
        data = json.load(archivo)

    class_names = data.get("class_names")
    if not class_names:
        print("labels.json no contiene class_names.")
        sys.exit(1)
    return class_names


def seleccionar_mejor_rostro(detecciones):
    candidatas = []
    for deteccion in detecciones:
        x, y, w, h = deteccion.get("box", [0, 0, 0, 0])
        if w <= 0 or h <= 0:
            continue
        area = w * h
        confianza = deteccion.get("confidence", 0.0)
        candidatas.append((confianza, area, deteccion))

    if not candidatas:
        return None

    candidatas.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidatas[0][2]


def crear_detector_haar():
    ruta = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(ruta)
    if detector.empty():
        print(f"No se pudo cargar el detector Haar: {ruta}")
        sys.exit(1)
    return detector


def detectar_rostros_mtcnn(detector, imagen_rgb):
    try:
        return detector.detect_faces(imagen_rgb)
    except (ValueError, RuntimeError, cv2.error) as error:
        mensaje = str(error)
        if "empty output" in mensaje or "shape=(0" in mensaje:
            return []
        return []


def detectar_rostros_haar(detector, imagen_rgb):
    gris = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)
    rostros = detector.detectMultiScale(
        gris,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    detecciones = []
    for x, y, w, h in rostros:
        detecciones.append(
            {
                "box": [int(x), int(y), int(w), int(h)],
                "confidence": 1.0,
            }
        )
    return detecciones


def detectar_rostros(detector, detector_tipo, imagen_rgb):
    if detector_tipo == "auto":
        detector_mtcnn, detector_haar = detector
        detecciones = detectar_rostros_mtcnn(detector_mtcnn, imagen_rgb)
        if detecciones:
            return detecciones
        return detectar_rostros_haar(detector_haar, imagen_rgb)
    if detector_tipo == "haar":
        return detectar_rostros_haar(detector, imagen_rgb)
    return detectar_rostros_mtcnn(detector, imagen_rgb)


def recortar_rostro_rgb(imagen_rgb, caja):
    x, y, w, h = caja
    alto, ancho = imagen_rgb.shape[:2]

    margen_arriba = int(h * 0.40)
    margen_abajo = int(h * 0.10)
    margen_lados = int(w * 0.20)

    x1 = max(0, x - margen_lados)
    y1 = max(0, y - margen_arriba)
    x2 = min(ancho, x + w + margen_lados)
    y2 = min(alto, y + h + margen_abajo)

    if x1 >= x2 or y1 >= y2:
        return None, None

    rostro = imagen_rgb[y1:y2, x1:x2]
    if rostro.size == 0:
        return None, None

    rostro = cv2.resize(rostro, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
    return rostro, (x1, y1, x2, y2)


def calcular_iou(caja_a, caja_b):
    if caja_a is None or caja_b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = caja_a
    bx1, by1, bx2, by2 = caja_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_ancho = max(0, ix2 - ix1)
    inter_alto = max(0, iy2 - iy1)
    inter_area = inter_ancho * inter_alto
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def ordenar_probabilidades(class_names, probabilidades, top):
    top = min(top, len(class_names))
    indices = np.argsort(probabilidades)[::-1][:top]
    return [(class_names[i], float(probabilidades[i])) for i in indices]


def predecir_probabilidades(model, rostro_rgb):
    entrada = np.expand_dims(rostro_rgb.astype(np.float32), axis=0)
    return model.predict(entrada, verbose=0)[0]


def predecir(model, class_names, rostro_rgb, top):
    probabilidades = predecir_probabilidades(model, rostro_rgb)
    return ordenar_probabilidades(class_names, probabilidades, top)


def imprimir_predicciones(predicciones, umbral, margen):
    etiqueta, mejor_confianza = decidir_etiqueta(predicciones, umbral, margen)
    print(f"Resultado: {etiqueta} ({mejor_confianza:.2%})")
    print("Top predicciones:")
    for nombre, confianza in predicciones:
        print(f"- {nombre}: {confianza:.2%}")


def decidir_etiqueta(predicciones, umbral, margen):
    mejor_nombre, mejor_confianza = predicciones[0]
    segunda_confianza = predicciones[1][1] if len(predicciones) > 1 else 0.0
    diferencia = mejor_confianza - segunda_confianza
    if mejor_confianza < umbral or diferencia < margen:
        return "Desconocido", mejor_confianza
    return mejor_nombre, mejor_confianza


def reconocer_imagen(
    model,
    class_names,
    detector,
    detector_tipo,
    imagen_path,
    umbral,
    margen,
    top,
    imagen_recortada,
):
    imagen_bgr = cv2.imread(str(imagen_path))
    if imagen_bgr is None:
        print(f"No se pudo abrir la imagen: {imagen_path}")
        sys.exit(1)

    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    if imagen_recortada:
        rostro_rgb = cv2.resize(imagen_rgb, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
        predicciones = predecir(model, class_names, rostro_rgb, top)
        imprimir_predicciones(predicciones, umbral, margen)
        return

    detecciones = detectar_rostros(detector, detector_tipo, imagen_rgb)
    mejor = seleccionar_mejor_rostro(detecciones)
    if mejor is None:
        print("No se detecto ningun rostro en la imagen.")
        sys.exit(1)

    rostro_rgb, _ = recortar_rostro_rgb(imagen_rgb, mejor["box"])
    if rostro_rgb is None:
        print("No se pudo recortar el rostro detectado.")
        sys.exit(1)

    predicciones = predecir(model, class_names, rostro_rgb, top)
    imprimir_predicciones(predicciones, umbral, margen)


def reconocer_webcam(
    model,
    class_names,
    detector,
    detector_tipo,
    indice_camara,
    umbral,
    margen,
    top,
    suavizado,
    mostrar_top,
    reset_iou,
):
    cap = cv2.VideoCapture(indice_camara)
    if not cap.isOpened():
        print(f"No se pudo abrir la camara: {indice_camara}")
        sys.exit(1)

    window_name = "Reconocimiento Facial CNN"
    historial_probabilidades = deque(maxlen=max(1, suavizado))
    ultima_caja = None
    print("Reconociendo con webcam. Haz clic en la ventana y presiona q o Esc para salir.")

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detecciones = detectar_rostros(detector, detector_tipo, frame_rgb)
            mejor = seleccionar_mejor_rostro(detecciones)

            if mejor is None:
                historial_probabilidades.clear()
                ultima_caja = None
            else:
                rostro_rgb, caja = recortar_rostro_rgb(frame_rgb, mejor["box"])
                if rostro_rgb is not None:
                    if ultima_caja is not None and calcular_iou(ultima_caja, caja) < reset_iou:
                        historial_probabilidades.clear()
                    ultima_caja = caja

                    probabilidades = predecir_probabilidades(model, rostro_rgb)
                    historial_probabilidades.append(probabilidades)
                    probabilidades_promedio = np.mean(
                        np.stack(historial_probabilidades),
                        axis=0,
                    )
                    predicciones = ordenar_probabilidades(
                        class_names,
                        probabilidades_promedio,
                        top,
                    )
                    etiqueta, mejor_confianza = decidir_etiqueta(
                        predicciones,
                        umbral,
                        margen,
                    )
                    x1, y1, x2, y2 = caja
                    color = (0, 255, 0) if etiqueta != "Desconocido" else (0, 165, 255)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame_bgr,
                        f"{etiqueta} {mejor_confianza:.0%}",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                    if mostrar_top:
                        for indice, (nombre, confianza) in enumerate(predicciones, start=1):
                            cv2.putText(
                                frame_bgr,
                                f"{indice}. {nombre}: {confianza:.0%}",
                                (15, 25 + indice * 25),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )

            cv2.imshow(window_name, frame_bgr)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    except KeyboardInterrupt:
        print()
        print("Reconocimiento interrumpido.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = crear_parser().parse_args()
    modelo_path = Path(args.modelo)
    if not modelo_path.is_file():
        print(f"No existe el modelo: {modelo_path}")
        print("Primero entrena con: py -3.13 python\\entrenar_cnn.py")
        sys.exit(1)

    tf = importar_tensorflow()
    model = tf.keras.models.load_model(modelo_path)
    class_names = cargar_labels(args.labels)
    print("Clases cargadas: " + ", ".join(class_names))
    print(f"Modelo: {modelo_path}")
    if args.detector == "auto":
        MTCNN = importar_mtcnn()
        detector = (MTCNN(), crear_detector_haar())
    elif args.detector == "mtcnn":
        MTCNN = importar_mtcnn()
        detector = MTCNN()
    else:
        detector = crear_detector_haar()

    if args.imagen:
        reconocer_imagen(
            model,
            class_names,
            detector,
            args.detector,
            Path(args.imagen),
            args.umbral,
            args.margen,
            args.top,
            args.imagen_recortada,
        )
    else:
        reconocer_webcam(
            model,
            class_names,
            detector,
            args.detector,
            args.camara,
            args.umbral,
            args.margen,
            args.top,
            args.suavizado,
            args.mostrar_top,
            args.reset_iou,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Proceso interrumpido por el usuario.")
