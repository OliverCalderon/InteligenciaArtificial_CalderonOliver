import argparse
import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASES_EQUIPO = [
    "Cristofer Sarabia Sosa",
    "Diego_Monreal",
    "Oliver_Calderon",
]


def resolver_ruta(valor):
    ruta = Path(valor)
    if ruta.is_absolute():
        return ruta
    return PROJECT_ROOT / ruta


def crear_parser():
    parser = argparse.ArgumentParser(
        description="Crea un dataset reducido con solo las clases indicadas."
    )
    parser.add_argument(
        "--origen",
        default="Dataset",
        help="Dataset de origen. Por defecto: Dataset",
    )
    parser.add_argument(
        "--salida",
        default="Dataset_equipo",
        help="Dataset reducido de salida. Por defecto: Dataset_equipo",
    )
    parser.add_argument(
        "--clases",
        nargs="+",
        default=CLASES_EQUIPO,
        help="Nombres de carpetas/clases a incluir.",
    )
    parser.add_argument(
        "--copiar",
        action="store_true",
        help="Copia archivos en lugar de crear hardlinks.",
    )
    parser.add_argument(
        "--con-desconocido",
        action="store_true",
        help="Agrega una clase Desconocido usando carpetas que no son del equipo.",
    )
    parser.add_argument(
        "--clase-desconocido",
        default="Desconocido",
        help="Nombre de la clase para rostros fuera del equipo. Por defecto: Desconocido",
    )
    parser.add_argument(
        "--max-desconocidos",
        type=int,
        default=300,
        help="Maximo de imagenes para la clase Desconocido. Por defecto: 300",
    )
    return parser


def enlazar_o_copiar(origen, destino, copiar):
    if destino.exists():
        return False

    destino.parent.mkdir(parents=True, exist_ok=True)
    if copiar:
        shutil.copy2(origen, destino)
        return True

    try:
        os.link(origen, destino)
    except OSError:
        shutil.copy2(origen, destino)
    return True


def obtener_imagenes(carpeta):
    return [
        archivo
        for archivo in sorted(carpeta.iterdir())
        if archivo.is_file() and archivo.suffix.lower() in IMAGE_EXTENSIONS
    ]


def crear_clase_desconocido(origen, salida, clases_equipo, nombre_clase, maximo, copiar):
    carpeta_salida = salida / nombre_clase
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    carpetas_candidatas = [
        carpeta
        for carpeta in sorted(origen.iterdir())
        if carpeta.is_dir() and carpeta.name not in set(clases_equipo)
    ]

    imagenes_por_carpeta = [
        (carpeta.name, obtener_imagenes(carpeta))
        for carpeta in carpetas_candidatas
    ]
    imagenes_por_carpeta = [
        (nombre, imagenes)
        for nombre, imagenes in imagenes_por_carpeta
        if imagenes
    ]

    creadas = 0
    indice_global = 0
    while creadas < maximo:
        agrego_en_ronda = False
        for nombre_clase_origen, imagenes in imagenes_por_carpeta:
            if indice_global >= len(imagenes):
                continue
            origen_archivo = imagenes[indice_global]
            destino = carpeta_salida / f"{nombre_clase_origen}_{origen_archivo.name}"
            if enlazar_o_copiar(origen_archivo, destino, copiar):
                creadas += 1
                agrego_en_ronda = True
            if creadas >= maximo:
                break
        if not agrego_en_ronda:
            break
        indice_global += 1

    existentes = len(obtener_imagenes(carpeta_salida))
    print(f"{nombre_clase}: {existentes} imagenes ({creadas} nuevas)")
    return existentes


def main():
    args = crear_parser().parse_args()
    origen = resolver_ruta(args.origen)
    salida = resolver_ruta(args.salida)

    if not origen.is_dir():
        print(f"No existe el dataset de origen: {origen}")
        sys.exit(1)

    salida.mkdir(parents=True, exist_ok=True)
    total = 0
    faltantes = []

    for clase in args.clases:
        carpeta_origen = origen / clase
        if not carpeta_origen.is_dir():
            faltantes.append(clase)
            continue

        carpeta_salida = salida / clase
        carpeta_salida.mkdir(parents=True, exist_ok=True)
        creadas = 0
        for archivo in sorted(carpeta_origen.iterdir()):
            if not archivo.is_file() or archivo.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            destino = carpeta_salida / archivo.name
            if enlazar_o_copiar(archivo, destino, args.copiar):
                creadas += 1

        existentes = len(
            [
                archivo
                for archivo in carpeta_salida.iterdir()
                if archivo.is_file() and archivo.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )
        total += existentes
        print(f"{clase}: {existentes} imagenes ({creadas} nuevas)")

    if faltantes:
        print()
        print("Clases no encontradas:")
        for clase in faltantes:
            print(f"- {clase}")

    if args.con_desconocido:
        total += crear_clase_desconocido(
            origen,
            salida,
            args.clases,
            args.clase_desconocido,
            args.max_desconocidos,
            args.copiar,
        )

    print()
    print(f"Dataset reducido listo: {salida}")
    print(f"Total de imagenes: {total}")


if __name__ == "__main__":
    main()
