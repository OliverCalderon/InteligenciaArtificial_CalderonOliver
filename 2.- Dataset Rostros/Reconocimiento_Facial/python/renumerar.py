import argparse
import os
import re
import sys
import uuid


PATH_DATOS = "Dataset"
PATRON_ARCHIVO = re.compile(r"^(rostro_)(\d+)(\.[A-Za-z0-9]+)$")


def crear_parser():
    parser = argparse.ArgumentParser(
        description="Renumera archivos rostro_N.ext de una carpeta para eliminar huecos."
    )
    parser.add_argument(
        "--persona",
        help="Nombre de la carpeta dentro de Dataset. Ejemplo: P Diddy",
    )
    parser.add_argument(
        "--ruta",
        help="Ruta completa o relativa a la carpeta a renumerar.",
    )
    parser.add_argument(
        "--inicio",
        type=int,
        default=1,
        help="Numero inicial para la renumeracion. Por defecto: 1",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo muestra como quedarian los nombres, sin renombrar archivos.",
    )
    return parser


def pedir_texto(mensaje, valor_por_defecto=None):
    texto = input(mensaje).strip()
    if texto:
        return texto
    return valor_por_defecto


def pedir_entero(mensaje, valor_por_defecto):
    texto = input(mensaje).strip()
    if not texto:
        return valor_por_defecto
    try:
        return int(texto)
    except ValueError:
        print("Valor invalido. Se usara el valor por defecto.")
        return valor_por_defecto


def resolver_carpeta(args):
    if args.ruta:
        return args.ruta

    persona = args.persona or pedir_texto("Nombre de la carpeta/persona a renumerar: ")
    if not persona:
        print("Debes indicar una carpeta.")
        sys.exit(1)
    return os.path.join(PATH_DATOS, persona)


def obtener_archivos_rostro(carpeta):
    encontrados = []
    ignorados = []

    for nombre in os.listdir(carpeta):
        ruta = os.path.join(carpeta, nombre)
        if not os.path.isfile(ruta):
            continue

        coincidencia = PATRON_ARCHIVO.match(nombre)
        if not coincidencia:
            ignorados.append(nombre)
            continue

        prefijo, indice_texto, extension = coincidencia.groups()
        encontrados.append(
            {
                "nombre_actual": nombre,
                "ruta_actual": ruta,
                "prefijo": prefijo,
                "indice_actual": int(indice_texto),
                "extension": extension,
            }
        )

    encontrados.sort(key=lambda item: (item["indice_actual"], item["nombre_actual"].casefold()))
    return encontrados, ignorados


def construir_plan(archivos, inicio):
    plan = []
    indice_nuevo = inicio

    for archivo in archivos:
        nuevo_nombre = f"{archivo['prefijo']}{indice_nuevo}{archivo['extension']}"
        plan.append(
            {
                **archivo,
                "nuevo_nombre": nuevo_nombre,
            }
        )
        indice_nuevo += 1

    return plan


def mostrar_resumen(plan, ignorados, inicio):
    print(f"Archivos detectados para renumerar: {len(plan)}")
    print(f"Numeracion inicial: {inicio}")
    if ignorados:
        print(f"Archivos ignorados por no coincidir con el patron: {len(ignorados)}")

    cambios = [item for item in plan if item["nombre_actual"] != item["nuevo_nombre"]]
    print(f"Archivos que cambiaran de nombre: {len(cambios)}")

    if not plan:
        return

    vista_previa = cambios[:10] if cambios else plan[:10]
    print("Vista previa:")
    for item in vista_previa:
        print(f"- {item['nombre_actual']} -> {item['nuevo_nombre']}")


def aplicar_renumeracion(carpeta, plan):
    token = uuid.uuid4().hex
    renombres_temporales = []

    for posicion, item in enumerate(plan, start=1):
        ruta_temporal = os.path.join(carpeta, f"__tmp_renumerar_{token}_{posicion}{item['extension']}")
        os.replace(item["ruta_actual"], ruta_temporal)
        renombres_temporales.append(
            {
                **item,
                "ruta_temporal": ruta_temporal,
                "ruta_final": os.path.join(carpeta, item["nuevo_nombre"]),
            }
        )

    for item in renombres_temporales:
        os.replace(item["ruta_temporal"], item["ruta_final"])


def main():
    args = crear_parser().parse_args()
    carpeta = resolver_carpeta(args)
    inicio = args.inicio

    if not os.path.isdir(carpeta):
        print(f"La carpeta no existe: {carpeta}")
        sys.exit(1)

    archivos, ignorados = obtener_archivos_rostro(carpeta)
    if not archivos:
        print("No se encontraron archivos con el patron rostro_<numero>.<extension>.")
        sys.exit(1)

    plan = construir_plan(archivos, inicio)
    mostrar_resumen(plan, ignorados, inicio)

    if args.dry_run:
        print("Dry run activado. No se realizaron cambios.")
        return

    confirmar = pedir_texto("Deseas aplicar la renumeracion? [s/N]: ", "n")
    if confirmar.lower() not in {"s", "si", "sí", "y", "yes"}:
        print("Operacion cancelada.")
        return

    aplicar_renumeracion(carpeta, plan)
    print("Renumeracion completada.")
    if plan:
        print(f"Primer archivo esperado: {plan[0]['nuevo_nombre']}")
        print(f"Ultimo archivo esperado: {plan[-1]['nuevo_nombre']}")


if __name__ == "__main__":
    main()
