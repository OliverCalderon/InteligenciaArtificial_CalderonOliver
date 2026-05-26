import argparse
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np


PATH_DATOS = "Dataset"
TAMANO_SALIDA = (160, 160)
MAX_BYTES_DESCARGA = 15 * 1024 * 1024
MAX_LADO_DETECCION = 1024
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/136.0.0.0 Safari/537.36"
)
API_USER_AGENT = "CodexDatasetTool/1.0"
ALIAS_DESCARTADOS = {"love"}
MAX_ALIAS_AUTOMATICOS = 1
PATRONES_CONSULTA_CERRADOS = [
    "{nombre} portrait face",
    "{nombre} close up face",
    "{nombre} headshot",
    "{nombre} face",
    "{nombre} portrait",
    "{nombre} press photo portrait",
    "{nombre} interview face",
]
PATRONES_CONSULTA_ABIERTOS = [
    "{nombre} photo",
    "{nombre} news photo",
    "{nombre} public appearance",
    "{nombre} interview",
    "{nombre} event photo",
    "{nombre} archive photo",
    "{nombre} young photo",
    "{nombre}",
]
REGIONES_FALLBACK = ["wt-wt", "mx-es", "us-en", "uk-en"]


def importar_ddgs():
    try:
        from ddgs import DDGS
    except ImportError:
        print("Falta la dependencia 'ddgs'.")
        print("Instalala con: py -3.13 -m pip install ddgs")
        sys.exit(1)
    return DDGS


def importar_mtcnn():
    try:
        from mtcnn import MTCNN
    except ImportError:
        print("Falta la dependencia 'mtcnn'.")
        print("Instalala con: py -3.13 -m pip install mtcnn tensorflow")
        sys.exit(1)
    return MTCNN


def crear_parser():
    parser = argparse.ArgumentParser(
        description="Busca imagenes en internet, detecta el rostro principal y lo guarda en Dataset."
    )
    parser.add_argument("--persona", help="Nombre de la persona o personaje.")
    parser.add_argument(
        "--consulta",
        help="Busqueda personalizada principal. Si no se indica, se generan consultas automaticamente.",
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        help="Alias adicional de la persona. Puede repetirse varias veces.",
    )
    parser.add_argument(
        "--cantidad",
        type=int,
        help="Numero de imagenes nuevas que se intentaran guardar.",
    )
    parser.add_argument(
        "--max-resultados",
        type=int,
        help="Cantidad maxima de resultados a pedir al buscador por cada consulta.",
    )
    parser.add_argument(
        "--region",
        default="wt-wt",
        help="Region del buscador. Ejemplo: wt-wt, us-en, mx-es.",
    )
    parser.add_argument(
        "--safesearch",
        default="moderate",
        choices=["on", "moderate", "off"],
        help="Nivel de filtrado del buscador.",
    )
    parser.add_argument(
        "--umbral-rostro",
        type=float,
        default=0.90,
        help="Confianza minima para priorizar una deteccion de rostro.",
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
        valor = int(texto)
    except ValueError:
        print("Valor invalido. Se usara el valor por defecto.")
        return valor_por_defecto
    return max(1, valor)


def obtener_siguiente_indice(carpeta):
    indices = []
    for archivo in os.listdir(carpeta):
        coincidencia = re.match(r"rostro_(\d+)\.jpg$", archivo)
        if coincidencia:
            indices.append(int(coincidencia.group(1)))
    if not indices:
        return 0
    return max(indices) + 1


def normalizar_texto(texto):
    return " ".join(texto.strip().split())


def normalizar_para_comparar(texto):
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(caracter for caracter in texto if not unicodedata.combining(caracter))
    texto = texto.casefold()
    return re.sub(r"[^a-z0-9]+", "", texto)


def agregar_si_no_existe(destino, vistos, valor, funcion_clave=None):
    valor_normalizado = normalizar_texto(valor)
    if not valor_normalizado:
        return
    clave = (
        funcion_clave(valor_normalizado)
        if funcion_clave is not None
        else valor_normalizado.casefold()
    )
    if clave in vistos:
        return
    vistos.add(clave)
    destino.append(valor_normalizado)


def consultar_json(url_base, params):
    url = f"{url_base}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": API_USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as respuesta:
        return json.load(respuesta)


def buscar_entidad_wikidata(nombre):
    try:
        data = consultar_json(
            WIKIDATA_API_URL,
            {
                "action": "wbsearchentities",
                "search": nombre,
                "language": "en",
                "type": "item",
                "limit": 5,
                "format": "json",
            },
        )
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None

    objetivo = normalizar_para_comparar(nombre)
    for resultado in data.get("search", []):
        candidatos = [resultado.get("label", "")]
        candidatos.extend(resultado.get("aliases", []))

        match = resultado.get("match", {})
        if match.get("text"):
            candidatos.append(match["text"])

        for candidato in candidatos:
            if normalizar_para_comparar(candidato) == objetivo:
                return resultado.get("id")

    return None


def puntuar_alias(alias):
    alias_limpio = normalizar_texto(alias)
    tokens = alias_limpio.replace('"', " ").split()
    puntaje = 0

    if 2 <= len(tokens) <= 4:
        puntaje += 4
    elif len(tokens) == 1:
        puntaje += 2

    if 6 <= len(alias_limpio) <= 24:
        puntaje += 2

    if "." in alias_limpio or "-" in alias_limpio:
        puntaje += 1

    if '"' in alias_limpio or "(" in alias_limpio or ")" in alias_limpio:
        puntaje -= 1

    return puntaje


def alias_es_util(alias):
    alias_limpio = normalizar_texto(alias)
    if len(alias_limpio) < 4:
        return False
    if alias_limpio.casefold() in ALIAS_DESCARTADOS:
        return False
    return True


def obtener_aliases_wikidata(entity_id):
    try:
        data = consultar_json(
            WIKIDATA_API_URL,
            {
                "action": "wbgetentities",
                "ids": entity_id,
                "languages": "en|es",
                "languagefallback": "1",
                "props": "labels|aliases",
                "format": "json",
            },
        )
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return []

    entidad = data.get("entities", {}).get(entity_id, {})
    labels = []
    aliases = []
    vistos = set()

    for label in entidad.get("labels", {}).values():
        valor = label.get("value", "")
        if alias_es_util(valor):
            agregar_si_no_existe(labels, vistos, valor, normalizar_para_comparar)

    for grupo_aliases in entidad.get("aliases", {}).values():
        for alias in grupo_aliases:
            valor = alias.get("value", "")
            if alias_es_util(valor):
                agregar_si_no_existe(aliases, vistos, valor, normalizar_para_comparar)

    aliases.sort(key=lambda valor: (-puntuar_alias(valor), valor.casefold()))
    combinados = labels + aliases
    return combinados[:MAX_ALIAS_AUTOMATICOS]


def descubrir_aliases_publicos(persona, aliases_usuario=None):
    candidatos = []
    vistos = set()
    semillas = [persona]
    semillas.extend(aliases_usuario or [])

    for semilla in semillas:
        entity_id = buscar_entidad_wikidata(semilla)
        if not entity_id:
            continue
        for alias in obtener_aliases_wikidata(entity_id):
            agregar_si_no_existe(candidatos, vistos, alias)

    return candidatos


def construir_consultas(persona, consulta_personalizada=None, aliases=None):
    consultas = []
    consultas_vistas = set()
    semillas = []
    semillas_vistas = set()

    agregar_si_no_existe(semillas, semillas_vistas, persona)
    for alias in aliases or []:
        agregar_si_no_existe(semillas, semillas_vistas, alias)

    if consulta_personalizada:
        agregar_si_no_existe(consultas, consultas_vistas, consulta_personalizada)

    for semilla in semillas:
        for patron in PATRONES_CONSULTA_CERRADOS:
            agregar_si_no_existe(consultas, consultas_vistas, patron.format(nombre=semilla))

    for semilla in semillas:
        for patron in PATRONES_CONSULTA_ABIERTOS:
            agregar_si_no_existe(consultas, consultas_vistas, patron.format(nombre=semilla))

    return consultas


def construir_regiones_busqueda(region_principal):
    regiones = []
    vistos = set()
    for region in [region_principal, *REGIONES_FALLBACK]:
        if not region or region in vistos:
            continue
        vistos.add(region)
        regiones.append(region)
    return regiones


def buscar_resultados(DDGS, consulta, region, safesearch, max_resultados):
    with DDGS() as buscador:
        resultados = buscador.images(
            query=consulta,
            region=region,
            safesearch=safesearch,
            max_results=max_resultados,
        )
    if not resultados:
        return []
    return resultados


def buscar_resultados_con_regiones(DDGS, consulta, regiones, safesearch, max_resultados):
    ultimo_error = None

    for region in regiones:
        try:
            resultados = buscar_resultados(
                DDGS,
                consulta,
                region,
                safesearch,
                max_resultados,
            )
        except Exception as error:
            ultimo_error = error
            continue

        if resultados:
            return resultados, region

    if ultimo_error is not None:
        raise ultimo_error

    return [], None


def descargar_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as respuesta:
        content_length = respuesta.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > MAX_BYTES_DESCARGA:
                    raise ValueError("Archivo demasiado grande")
            except ValueError as error:
                raise ValueError("Archivo demasiado grande") from error

        data = respuesta.read(MAX_BYTES_DESCARGA + 1)
        if len(data) > MAX_BYTES_DESCARGA:
            raise ValueError("Archivo demasiado grande")
        return data


def decodificar_imagen(data):
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def seleccionar_mejor_rostro(detecciones, umbral_rostro):
    candidatas = []
    for deteccion in detecciones:
        x, y, w, h = deteccion.get("box", [0, 0, 0, 0])
        if w <= 0 or h <= 0:
            continue
        area = w * h
        confianza = deteccion.get("confidence", 0.0)
        candidatas.append((confianza >= umbral_rostro, area, deteccion))

    if not candidatas:
        return None

    candidatas.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidatas[0][2]


def recortar_rostro(imagen, caja):
    x, y, w, h = caja
    alto, ancho = imagen.shape[:2]

    margen_arriba = int(h * 0.40)
    margen_abajo = int(h * 0.10)
    margen_lados = int(w * 0.20)

    x1 = max(0, x - margen_lados)
    y1 = max(0, y - margen_arriba)
    x2 = min(ancho, x + w + margen_lados)
    y2 = min(alto, y + h + margen_abajo)

    if x1 >= x2 or y1 >= y2:
        return None

    rostro = imagen[y1:y2, x1:x2]
    if rostro.size == 0:
        return None

    return cv2.resize(rostro, TAMANO_SALIDA, interpolation=cv2.INTER_CUBIC)


def reducir_imagen_para_deteccion(imagen):
    alto, ancho = imagen.shape[:2]
    lado_maximo = max(alto, ancho)
    if lado_maximo <= MAX_LADO_DETECCION:
        return imagen

    escala = MAX_LADO_DETECCION / float(lado_maximo)
    nuevo_ancho = max(1, int(ancho * escala))
    nuevo_alto = max(1, int(alto * escala))
    return cv2.resize(imagen, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)


def extraer_rostro(detector, imagen, umbral_rostro):
    if imagen is None:
        return None

    imagen = reducir_imagen_para_deteccion(imagen)
    alto, ancho = imagen.shape[:2]
    if alto < 80 or ancho < 80:
        return None

    rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    try:
        detecciones = detector.detect_faces(rgb)
    except Exception:
        return None
    mejor_rostro = seleccionar_mejor_rostro(detecciones, umbral_rostro)
    if mejor_rostro is None:
        return None

    return recortar_rostro(imagen, mejor_rostro["box"])


def guardar_resultados(
    detector,
    resultados,
    carpeta_destino,
    indice_inicial,
    cantidad_objetivo,
    umbral_rostro,
    urls_vistas=None,
):
    guardadas = 0
    sin_rostro = 0
    fallidas = 0
    urls_vistas = urls_vistas if urls_vistas is not None else set()
    indice_actual = indice_inicial

    for posicion, resultado in enumerate(resultados, start=1):
        if guardadas >= cantidad_objetivo:
            break

        urls_posibles = [resultado.get("image"), resultado.get("thumbnail")]
        urls_posibles = [url for url in urls_posibles if url and url not in urls_vistas]
        if not urls_posibles:
            continue

        imagen_procesada = None
        ultima_url = None

        for url in urls_posibles:
            ultima_url = url
            urls_vistas.add(url)
            try:
                data = descargar_bytes(url)
            except (urllib.error.URLError, TimeoutError, ValueError, OSError):
                continue

            imagen = decodificar_imagen(data)
            imagen_procesada = extraer_rostro(detector, imagen, umbral_rostro)
            if imagen_procesada is not None:
                break

        if imagen_procesada is None:
            sin_rostro += 1
            print(f"[{posicion}] Sin rostro util: {ultima_url or 'sin URL'}")
            continue

        ruta_salida = os.path.join(carpeta_destino, f"rostro_{indice_actual}.jpg")
        guardado = cv2.imwrite(ruta_salida, imagen_procesada)
        if not guardado:
            fallidas += 1
            print(f"[{posicion}] No se pudo guardar la imagen procesada.")
            continue

        print(f"[{posicion}] Guardada: rostro_{indice_actual}.jpg")
        indice_actual += 1
        guardadas += 1

    return guardadas, sin_rostro, fallidas


def main():
    args = crear_parser().parse_args()

    persona = args.persona or pedir_texto("Nombre de la persona o personaje: ")
    if not persona:
        print("Debes indicar un nombre.")
        sys.exit(1)

    os.makedirs(PATH_DATOS, exist_ok=True)
    carpeta_destino = os.path.join(PATH_DATOS, persona)
    os.makedirs(carpeta_destino, exist_ok=True)

    indice_inicial = obtener_siguiente_indice(carpeta_destino)
    aliases_automaticos = descubrir_aliases_publicos(persona, args.alias)
    aliases_finales = []
    aliases_vistos = set()
    agregar_si_no_existe(aliases_finales, aliases_vistos, persona)
    for alias in args.alias:
        agregar_si_no_existe(aliases_finales, aliases_vistos, alias)
    for alias in aliases_automaticos:
        agregar_si_no_existe(aliases_finales, aliases_vistos, alias)

    consultas = construir_consultas(persona, args.consulta, aliases_finales[1:])

    cantidad_sugerida = 50
    if indice_inicial < 200:
        cantidad_sugerida = max(1, 200 - indice_inicial)

    cantidad_objetivo = args.cantidad
    if cantidad_objetivo is None:
        cantidad_objetivo = pedir_entero(
            f"Cuantas imagenes nuevas quieres guardar? [{cantidad_sugerida}]: ",
            cantidad_sugerida,
        )

    if cantidad_objetivo <= 0:
        print("La cantidad debe ser mayor a 0.")
        sys.exit(1)

    max_resultados = args.max_resultados or max(cantidad_objetivo * 4, 100)
    regiones_busqueda = construir_regiones_busqueda(args.region)

    DDGS = importar_ddgs()
    MTCNN = importar_mtcnn()

    print(f"Buscando imagenes para: {persona}")
    if aliases_automaticos:
        print("Alias detectados automaticamente: " + ", ".join(aliases_automaticos))
    print(f"Consultas preparadas: {len(consultas)}")
    print(f"Resultados solicitados por consulta: {max_resultados}")
    print("Regiones de intento: " + ", ".join(regiones_busqueda))

    detector = MTCNN()
    urls_vistas = set()
    guardadas_totales = 0
    sin_rostro_total = 0
    fallidas_totales = 0
    consultas_con_resultados = 0

    for numero_consulta, consulta in enumerate(consultas, start=1):
        restantes = cantidad_objetivo - guardadas_totales
        if restantes <= 0:
            break

        print()
        print(f"Consulta {numero_consulta}/{len(consultas)}: {consulta}")
        try:
            resultados, region_usada = buscar_resultados_con_regiones(
                DDGS,
                consulta,
                regiones_busqueda,
                args.safesearch,
                max_resultados,
            )
        except Exception as error:
            print(f"No se pudo consultar el buscador con esa consulta: {error}")
            continue

        if not resultados:
            print("Sin resultados.")
            continue

        consultas_con_resultados += 1
        if region_usada:
            print(f"Region usada: {region_usada}")
        print(f"Resultados recibidos: {len(resultados)}")

        guardadas, sin_rostro, fallidas = guardar_resultados(
            detector,
            resultados,
            carpeta_destino,
            indice_inicial + guardadas_totales,
            restantes,
            args.umbral_rostro,
            urls_vistas,
        )
        guardadas_totales += guardadas
        sin_rostro_total += sin_rostro
        fallidas_totales += fallidas

    print()
    print("Proceso finalizado.")
    print(f"Carpeta destino: {carpeta_destino}")
    print(f"Consultas con resultados: {consultas_con_resultados}/{len(consultas)}")
    print(f"Imagenes nuevas guardadas: {guardadas_totales}")
    print(f"Resultados descartados por no detectar rostro: {sin_rostro_total}")
    print(f"Errores al guardar: {fallidas_totales}")
    print(f"Total de imagenes en la carpeta ahora: {obtener_siguiente_indice(carpeta_destino)}")

    if guardadas_totales < cantidad_objetivo:
        print("Sugerencia: agrega alias con --alias o una consulta mas especifica con --consulta.")
        print(
            'Ejemplo: py -3.13 python\\internet.py --persona "P Diddy" '
            '--alias "Sean Combs" --cantidad 80'
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Proceso interrumpido por el usuario.")
        print("Las imagenes ya guardadas se conservaron en la carpeta destino.")
