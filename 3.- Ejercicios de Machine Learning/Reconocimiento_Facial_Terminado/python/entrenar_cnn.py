import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = (160, 160)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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
        description="Entrena una CNN para reconocer las clases dentro del dataset."
    )
    parser.add_argument(
        "--dataset",
        default="Dataset",
        help="Carpeta del dataset. Por defecto: Dataset",
    )
    parser.add_argument(
        "--salida",
        default="modelo_cnn",
        help="Carpeta donde se guardara el modelo. Por defecto: modelo_cnn",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Epocas para entrenar la cabeza del modelo. Por defecto: 20",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=10,
        help="Epocas extra para ajustar las ultimas capas del modelo base. Por defecto: 10",
    )
    parser.add_argument(
        "--arquitectura",
        choices=["mobilenet", "simple"],
        default="mobilenet",
        help="Arquitectura CNN a usar. Por defecto: mobilenet",
    )
    parser.add_argument(
        "--pesos",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Pesos iniciales para MobileNetV2. Por defecto: imagenet",
    )
    parser.add_argument(
        "--unfreeze-layers",
        type=int,
        default=30,
        help="Ultimas capas del modelo base a descongelar en fine-tuning. Por defecto: 30",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamano de lote. Por defecto: 32",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Proporcion del dataset usada para validacion. Por defecto: 0.2",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Tasa de aprendizaje inicial. Por defecto: 0.001",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Semilla para separar entrenamiento/validacion. Por defecto: 123",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Epocas sin mejora antes de detener. Por defecto: 6",
    )
    parser.add_argument(
        "--sin-aumentacion",
        action="store_true",
        help="Desactiva la aumentacion en linea dentro del modelo.",
    )
    return parser


def contar_dataset(dataset_dir):
    conteos = {}
    for carpeta in sorted(dataset_dir.iterdir()):
        if not carpeta.is_dir():
            continue
        total = sum(
            1
            for archivo in carpeta.iterdir()
            if archivo.is_file() and archivo.suffix.lower() in IMAGE_EXTENSIONS
        )
        if total:
            conteos[carpeta.name] = total
    return conteos


def validar_dataset(dataset_dir, validation_split):
    if not dataset_dir.is_dir():
        print(f"La carpeta del dataset no existe: {dataset_dir}")
        sys.exit(1)

    if not 0.05 <= validation_split <= 0.5:
        print("--validation-split debe estar entre 0.05 y 0.5")
        sys.exit(1)

    conteos = contar_dataset(dataset_dir)
    if len(conteos) < 2:
        print("El dataset debe tener al menos 2 carpetas/clases con imagenes.")
        sys.exit(1)

    clases_pequenas = [clase for clase, total in conteos.items() if total < 5]
    if clases_pequenas:
        print("Estas clases tienen muy pocas imagenes:")
        for clase in clases_pequenas:
            print(f"- {clase}: {conteos[clase]}")
        print("Agrega mas imagenes antes de entrenar.")
        sys.exit(1)

    return conteos


def crear_datasets(tf, dataset_dir, batch_size, validation_split, seed):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )

    class_names = list(train_ds.class_names)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def crear_aumentacion(tf):
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.06),
            tf.keras.layers.RandomContrast(0.12),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
        ],
        name="aumentacion",
    )


def crear_modelo_simple(tf, num_clases, learning_rate, usar_aumentacion):
    regularizador = tf.keras.regularizers.l2(0.001)
    capas = [
        tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
    ]

    if usar_aumentacion:
        capas.append(crear_aumentacion(tf))

    capas.extend(
        [
            tf.keras.layers.Conv2D(
                16,
                3,
                padding="same",
                activation="relu",
                kernel_regularizer=regularizador,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Conv2D(
                32,
                3,
                padding="same",
                activation="relu",
                kernel_regularizer=regularizador,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Conv2D(
                64,
                3,
                padding="same",
                activation="relu",
                kernel_regularizer=regularizador,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(
                128,
                3,
                padding="same",
                activation="relu",
                kernel_regularizer=regularizador,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.50),
            tf.keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=regularizador,
            ),
            tf.keras.layers.Dropout(0.40),
            tf.keras.layers.Dense(num_clases, activation="softmax"),
        ]
    )

    model = tf.keras.Sequential(capas)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, None


def crear_modelo_mobilenet(tf, num_clases, learning_rate, usar_aumentacion, pesos):
    weights = "imagenet" if pesos == "imagenet" else None
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = weights is None

    inputs = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))
    x = inputs
    if usar_aumentacion:
        x = crear_aumentacion(tf)(x)
    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="preprocesamiento_mobilenet",
    )(x)
    x = base_model(x, training=weights is None)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_clases, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="cnn_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def crear_modelo(tf, num_clases, args):
    if args.arquitectura == "simple":
        return crear_modelo_simple(
            tf,
            num_clases=num_clases,
            learning_rate=args.learning_rate,
            usar_aumentacion=not args.sin_aumentacion,
        )

    return crear_modelo_mobilenet(
        tf,
        num_clases=num_clases,
        learning_rate=args.learning_rate,
        usar_aumentacion=not args.sin_aumentacion,
        pesos=args.pesos,
    )


def preparar_fine_tuning(tf, model, base_model, learning_rate, unfreeze_layers):
    if base_model is None:
        return False

    base_model.trainable = True
    capas_descongeladas = max(1, unfreeze_layers)
    limite = max(0, len(base_model.layers) - capas_descongeladas)

    for layer in base_model.layers[:limite]:
        layer.trainable = False
    for layer in base_model.layers[limite:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return True


def crear_callbacks(tf, salida_dir, patience):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            salida_dir / "mejor_modelo.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-6,
        ),
    ]


def guardar_metadata(salida_dir, class_names, conteos, args, val_loss, val_accuracy):
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_size": list(IMAGE_SIZE),
        "class_names": class_names,
        "class_counts": conteos,
        "validation_loss": float(val_loss),
        "validation_accuracy": float(val_accuracy),
        "args": {
            "dataset": str(resolver_ruta(args.dataset)),
            "salida": str(resolver_ruta(args.salida)),
            "epochs": args.epochs,
            "fine_tune_epochs": args.fine_tune_epochs,
            "arquitectura": args.arquitectura,
            "pesos": args.pesos,
            "unfreeze_layers": args.unfreeze_layers,
            "batch_size": args.batch_size,
            "validation_split": args.validation_split,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "patience": args.patience,
            "sin_aumentacion": args.sin_aumentacion,
        },
    }

    with (salida_dir / "labels.json").open("w", encoding="utf-8") as archivo:
        json.dump(
            {
                "class_names": class_names,
                "image_size": list(IMAGE_SIZE),
            },
            archivo,
            indent=2,
            ensure_ascii=False,
        )

    with (salida_dir / "metadata.json").open("w", encoding="utf-8") as archivo:
        json.dump(metadata, archivo, indent=2, ensure_ascii=False)


def guardar_historial(salida_dir, histories):
    historial = {}
    for nombre_fase, history in histories:
        for clave, valores in history.history.items():
            historial.setdefault(clave, [])
            historial[clave].extend(float(valor) for valor in valores)
        historial.setdefault("fase", [])
        historial["fase"].extend([nombre_fase] * len(history.epoch))

    with (salida_dir / "training_history.json").open("w", encoding="utf-8") as archivo:
        json.dump(historial, archivo, indent=2)


def main():
    args = crear_parser().parse_args()
    dataset_dir = resolver_ruta(args.dataset)
    salida_dir = resolver_ruta(args.salida)
    salida_dir.mkdir(parents=True, exist_ok=True)

    conteos = validar_dataset(dataset_dir, args.validation_split)
    print("Dataset detectado:")
    for clase, total in conteos.items():
        print(f"- {clase}: {total} imagenes")

    tf = importar_tensorflow()
    train_ds, val_ds, class_names = crear_datasets(
        tf,
        dataset_dir,
        args.batch_size,
        args.validation_split,
        args.seed,
    )
    model, base_model = crear_modelo(
        tf,
        num_clases=len(class_names),
        args=args,
    )

    print()
    model.summary()
    print()
    print(f"Entrenando CNN con {len(class_names)} clases...")
    print(f"Arquitectura: {args.arquitectura}")
    if args.arquitectura == "mobilenet":
        print(f"Pesos iniciales: {args.pesos}")
        if args.pesos == "none":
            print("Aviso: sin pesos ImageNet el entrenamiento sera desde cero y necesitara mas epocas.")

    histories = []
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=crear_callbacks(tf, salida_dir, args.patience),
    )
    histories.append(("head", history))

    if (
        args.arquitectura == "mobilenet"
        and args.pesos == "imagenet"
        and args.fine_tune_epochs > 0
    ):
        print()
        print("Iniciando fine-tuning de las ultimas capas de MobileNetV2...")
        fine_tuning_activo = preparar_fine_tuning(
            tf,
            model,
            base_model,
            learning_rate=args.learning_rate * 0.1,
            unfreeze_layers=args.unfreeze_layers,
        )
        if fine_tuning_activo:
            fine_history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.fine_tune_epochs,
                callbacks=crear_callbacks(tf, salida_dir, args.patience),
            )
            histories.append(("fine_tune", fine_history))

    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
    model.save(salida_dir / "modelo.keras")
    guardar_historial(salida_dir, histories)
    guardar_metadata(salida_dir, class_names, conteos, args, val_loss, val_accuracy)

    print()
    print("Entrenamiento finalizado.")
    print(f"Modelo guardado en: {salida_dir / 'modelo.keras'}")
    print(f"Mejor checkpoint: {salida_dir / 'mejor_modelo.keras'}")
    print(f"Etiquetas guardadas en: {salida_dir / 'labels.json'}")
    print(f"Accuracy de validacion: {val_accuracy:.4f}")
    print(f"Loss de validacion: {val_loss:.4f}")


if __name__ == "__main__":
    main()
