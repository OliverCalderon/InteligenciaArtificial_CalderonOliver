import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "modelo_equipo" / "mejor_modelo.keras"
DEFAULT_LABELS = PROJECT_ROOT / "modelo_equipo" / "labels.json"
IMAGE_SIZE = (160, 160)


def importar_tensorflow():
    try:
        import tensorflow as tf
    except ImportError:
        print("Falta la dependencia 'tensorflow'.")
        print("Instalala con: py -3.13 -m pip install -r requirements.txt")
        sys.exit(1)
    return tf


def resolver_ruta(valor):
    ruta = Path(valor)
    if ruta.is_absolute():
        return ruta
    return PROJECT_ROOT / ruta


def crear_parser():
    parser = argparse.ArgumentParser(
        description="Evalua el modelo CNN y muestra matriz de confusion por clase."
    )
    parser.add_argument("--dataset", default="Dataset_equipo")
    parser.add_argument("--modelo", default=str(DEFAULT_MODEL))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def cargar_labels(labels_path):
    with Path(labels_path).open("r", encoding="utf-8") as archivo:
        return json.load(archivo)["class_names"]


def crear_validacion(tf, dataset_dir, batch_size, validation_split, seed):
    return tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )


def main():
    args = crear_parser().parse_args()
    dataset_dir = resolver_ruta(args.dataset)
    modelo_path = resolver_ruta(args.modelo)
    labels_path = resolver_ruta(args.labels)

    if not dataset_dir.is_dir():
        print(f"No existe el dataset: {dataset_dir}")
        sys.exit(1)
    if not modelo_path.is_file():
        print(f"No existe el modelo: {modelo_path}")
        sys.exit(1)
    if not labels_path.is_file():
        print(f"No existe labels.json: {labels_path}")
        sys.exit(1)

    tf = importar_tensorflow()
    class_names = cargar_labels(labels_path)
    model = tf.keras.models.load_model(modelo_path)
    val_ds = crear_validacion(
        tf,
        dataset_dir,
        args.batch_size,
        args.validation_split,
        args.seed,
    )

    matriz = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    total = 0
    aciertos = 0

    for imagenes, etiquetas in val_ds:
        predicciones = model.predict(imagenes, verbose=0)
        y_true = np.argmax(etiquetas.numpy(), axis=1)
        y_pred = np.argmax(predicciones, axis=1)
        for real, predicha in zip(y_true, y_pred):
            matriz[real, predicha] += 1
            total += 1
            if real == predicha:
                aciertos += 1

    print(f"Accuracy validacion: {aciertos / max(1, total):.2%}")
    print()
    print("Matriz de confusion:")
    encabezado = "Real \\ Pred".ljust(28) + "".join(nombre[:10].rjust(12) for nombre in class_names)
    print(encabezado)
    for i, nombre in enumerate(class_names):
        fila = nombre[:26].ljust(28) + "".join(str(valor).rjust(12) for valor in matriz[i])
        print(fila)

    print()
    print("Accuracy por clase:")
    for i, nombre in enumerate(class_names):
        total_clase = matriz[i].sum()
        aciertos_clase = matriz[i, i]
        accuracy = aciertos_clase / max(1, total_clase)
        print(f"- {nombre}: {accuracy:.2%} ({aciertos_clase}/{total_clase})")


if __name__ == "__main__":
    main()
