import os
import random
import re
import unicodedata
from collections import defaultdict

import scipy.io.wavfile as wavfile
import torch
from transformers import AutoModel, AutoProcessor, VitsModel, AutoTokenizer, pipeline


# Modelos Hugging Face usados por el proyecto.
VISION_MODEL_NAME = os.getenv("COMENTARISTA_VISION_MODEL", "google/siglip-so400m-patch14-384")
VISION_ENSEMBLE_MODEL_NAME = os.getenv("COMENTARISTA_ENSEMBLE_VISION_MODEL", "openai/clip-vit-large-patch14")
NLP_MODEL_NAME = os.getenv("COMENTARISTA_NLP_MODEL", "google/flan-t5-small")
NLP_TASK_NAME = os.getenv("COMENTARISTA_NLP_TASK", "text2text-generation")
TTS_MODEL_NAME = os.getenv("COMENTARISTA_TTS_MODEL", "facebook/mms-tts-spa")

# Determinar dispositivo para ejecucion (GPU si esta disponible).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = 0 if torch.cuda.is_available() else -1

# Variables globales para inicializacion perezosa.
detector = None
clip_processor = None
ensemble_detector = None
ensemble_processor = None
narrador = None
narrador_error = None
tts_tokenizer = None
tts_model = None
ultima_prediccion = {}


PLANTILLAS_COMENTARIOS = {
    "minecraft": {
        "combate": [
            "¡Pelea directa! Le estan dando durisimo.",
            "¡Golpe va, golpe viene! Esto se puso serio.",
            "¡Combate a muerte! No puede fallar ahora.",
            "Espada en mano y a sobrevivir.",
            "Se defiende como puede, pero la presion esta encima.",
            "¡Vaya pelea intensa en Minecraft!"
        ],
        "exploración": [
            "Paseando por el mapa mientras busca la siguiente jugada.",
            "Explora terreno nuevo con bastante calma.",
            "Va leyendo el bioma y buscando recursos.",
            "Momento tranquilo para reconocer el mapa.",
            "Rotacion de exploracion, sin pelear todavia."
        ],
        "talar": [
            "Hacha en mano, esta talando madera.",
            "Va por troncos para juntar recursos de construccion.",
            "Arbol abajo, madera asegurada."
        ],
        "peligro": [
            "¡Cuidado! La vida esta bajisima.",
            "Esta a un golpe de caer, necesita reaccionar ya.",
            "¡Peligro total! Curarse es prioridad.",
            "Casi se va al lobby, que tension.",
            "La vida esta en minimos y cualquier error cuesta caro."
        ],
        "menú": [
            "Revisa el inventario para preparar la siguiente accion.",
            "Pausa tactica para craftear y ordenar recursos.",
            "Menu abierto, momento de pensar la ruta.",
            "Esta administrando objetos antes de volver a moverse."
        ],
        "victoria": [
            "¡Lo logro! Jugada cerrada con autoridad.",
            "Mision cumplida, se lleva el momento.",
            "¡Espectacular! Resuelve la situacion.",
            "Que jugadon para cerrar la secuencia."
        ],
        "minería": [
            "Pico en mano y directo por recursos.",
            "Minando bajo tierra, buscando valor.",
            "Esta picando con paciencia, a ver si aparece diamante.",
            "Ojo con la lava en esa zona subterranea."
        ],
        "minería_carbón": [
            "Esta sacando carbon para antorchas y hornos.",
            "Carbon detectado, recurso basico pero necesario.",
            "Buen momento para llenar inventario de combustible."
        ],
        "minería_hierro": [
            "Hierro en pantalla, esto mejora el equipo.",
            "Pico trabajando sobre mena de hierro.",
            "Ese hierro puede convertirse en armadura o herramientas."
        ],
        "minería_cobre": [
            "Cobre a la vista, recurso para construccion y decoracion.",
            "Esta picando cobre en la cueva.",
            "Mena de cobre detectada, bloque por bloque."
        ],
        "minería_oro": [
            "Oro encontrado, cuidado con la zona.",
            "Esta picando oro, recurso valioso para intercambios.",
            "Mena dorada en pantalla, buen hallazgo."
        ],
        "minería_redstone": [
            "Redstone detectada, se vienen mecanismos.",
            "Esta sacando redstone para circuitos.",
            "Polvo rojo asegurado para automatizar cosas."
        ],
        "minería_lapislázuli": [
            "Lapislazuli en pantalla, clave para encantamientos.",
            "Esta minando lapis para preparar la mesa de encantos.",
            "Recurso azul detectado, buena parada en la cueva."
        ],
        "minería_diamante": [
            "¡Diamante! Este si cambia la partida.",
            "Mena de diamante detectada, momento importante.",
            "Pico directo al diamante, hallazgo de alto valor."
        ],
        "minería_esmeralda": [
            "Esmeralda encontrada, ideal para comerciar.",
            "Mena de esmeralda en pantalla, hallazgo raro.",
            "Ese verde vale para tratos con aldeanos."
        ],
        "minería_escombros": [
            "Escombros ancestrales, esto apunta a netherita.",
            "Hallazgo de alto nivel en el Nether.",
            "Si lo procesa bien, puede mejorar a netherita."
        ],
        "construcción": [
            "Construye bloque a bloque con buena idea.",
            "Levanta estructura y gana control del espacio.",
            "Modo arquitecto activado en plena partida.",
            "Esta colocando bloques para preparar la zona."
        ],
        "crafteo": [
            "Mesa de crafteo abierta, prepara la siguiente herramienta.",
            "Esta combinando recursos para avanzar.",
            "Momento de crafteo antes de seguir la ruta."
        ],
        "fundición": [
            "Horno activo, toca convertir minerales en lingotes.",
            "Esta fundiendo recursos para mejorar equipo.",
            "Procesando materiales, buen paso de preparacion."
        ],
        "cultivo": [
            "Trabajando la granja, comida asegurada.",
            "Cultivos en pantalla, gestionando recursos de supervivencia.",
            "Esta sembrando o cosechando para sostener la partida."
        ],
        "pesca": [
            "Caña en mano, momento de pesca.",
            "Busca botin o comida desde el agua.",
            "Pesca tranquila, a ver que sale del anzuelo."
        ],
        "ganadería": [
            "Animales cerca, posible granja o comida.",
            "Gestionando ganado para recursos constantes.",
            "Momento de crianza o recoleccion con animales."
        ],
        "cofre": [
            "Cofre abierto, esta administrando recursos.",
            "Revisa almacenamiento antes de seguir avanzando.",
            "Organizacion de inventario desde el cofre."
        ],
        "encantamiento": [
            "Mesa de encantamientos abierta, busca mejorar el equipo.",
            "Momento de encantamientos, puede salir algo fuerte.",
            "Esta preparando poder extra para herramientas o armadura."
        ],
        "pociones": [
            "Soporte de pociones activo, preparando efectos.",
            "Esta mezclando ingredientes para una pocion.",
            "Alquimia en marcha, buen recurso para sobrevivir."
        ],
        "comer": [
            "Se esta curando con comida, buena decision.",
            "Toca recuperar hambre antes de seguir.",
            "Comer ahora puede salvar la siguiente pelea."
        ],
        "dormir": [
            "Cama lista, va a saltarse la noche.",
            "Dormir para reiniciar punto de aparicion.",
            "Pausa nocturna antes de continuar la aventura."
        ],
        "nadar": [
            "Esta nadando, ojo con el oxigeno.",
            "Cruza por agua buscando una nueva ruta.",
            "Movimiento acuatico, necesita controlar la respiracion."
        ],
        "bote": [
            "Va en bote, rotacion rapida por agua.",
            "Navegacion tranquila para cubrir distancia.",
            "Bote en marcha, buena forma de explorar."
        ],
        "portal": [
            "Portal activo, cambio de dimension en puerta.",
            "Se prepara para cruzar el portal.",
            "Ese portal puede llevarlo directo a otra fase."
        ],
        "nether": [
            "Esta en el Nether, todo es mas peligroso.",
            "Bioma infernal en pantalla, cuidado con lava y mobs.",
            "Zona del Nether, aqui cada paso pesa."
        ],
        "aldea": [
            "Aldea detectada, posible descanso y recursos.",
            "Casas y aldeanos cerca, buen punto estrategico.",
            "Llego a una aldea, momento de explorar."
        ],
        "comercio": [
            "Comercio con aldeano, puede conseguir recursos clave.",
            "Intercambio abierto, revisa ofertas.",
            "Los aldeanos pueden darle una mejora importante."
        ]
    }
}


# Prompts descriptivos para CLIP. Usar varias frases por estado reduce falsos
# positivos porque el score se agrega por categoria y no por una sola palabra.
PROMPTS_CLIP = {
    "minecraft": {
        "combate": [
            "a Minecraft gameplay screenshot where the player is fighting a hostile mob",
            "Minecraft combat with sword, bow, shield, enemies, or damage particles",
            "a Minecraft player attacking or being attacked by zombies skeletons spiders or monsters",
            "an intense Minecraft fight scene with danger and enemies nearby"
        ],
        "exploración": [
            "a calm Minecraft exploration screenshot in the overworld",
            "Minecraft player walking through terrain, forest, plains, desert, or mountains",
            "a Minecraft landscape view while exploring and searching for resources",
            "Minecraft gameplay with no menu and no combat, just moving around the map"
        ],
        "talar": [
            "Minecraft player chopping a tree with an axe",
            "Minecraft gameplay focused on cutting wood logs from a tree",
            "Minecraft oak birch spruce jungle acacia or dark oak tree being chopped",
            "Minecraft player collecting logs and wood blocks"
        ],
        "peligro": [
            "Minecraft gameplay with critically low health hearts in the HUD",
            "Minecraft player in danger taking damage burning drowning falling or near lava",
            "Minecraft survival moment with red damage overlay and almost no hearts",
            "Minecraft character about to die or escaping from enemies"
        ],
        "menú": [
            "Minecraft inventory screen open with item slots",
            "Minecraft crafting table chest furnace pause menu or GUI menu open",
            "a Minecraft menu interface covering the gameplay screen",
            "Minecraft player managing items in inventory"
        ],
        "victoria": [
            "Minecraft advancement achieved popup notification on screen",
            "Minecraft end credits or game completed screen",
            "Minecraft boss defeated or raid victory text on screen",
            "Minecraft achievement toast message in the corner of the screen"
        ],
        "minería": [
            "Minecraft underground cave mining ores with a pickaxe",
            "Minecraft player digging stone coal iron diamond or blocks underground",
            "Minecraft mining tunnel cave or deepslate scene",
            "Minecraft gameplay focused on collecting blocks below ground"
        ],
        "minería_carbón": [
            "Minecraft coal ore block with black speckles being mined",
            "Minecraft player mining coal ore in a cave",
            "Minecraft coal ore vein in stone or deepslate"
        ],
        "minería_hierro": [
            "Minecraft iron ore block with brown tan speckles being mined",
            "Minecraft player mining iron ore in stone",
            "Minecraft raw iron ore vein underground"
        ],
        "minería_cobre": [
            "Minecraft copper ore block with orange green speckles being mined",
            "Minecraft player mining copper ore underground",
            "Minecraft copper ore vein in stone or deepslate"
        ],
        "minería_oro": [
            "Minecraft gold ore block with yellow speckles being mined",
            "Minecraft player mining gold ore underground",
            "Minecraft gold ore vein in cave or deepslate"
        ],
        "minería_redstone": [
            "Minecraft redstone ore block glowing red being mined",
            "Minecraft player mining redstone ore underground",
            "Minecraft red ore vein with red particles in a cave"
        ],
        "minería_lapislázuli": [
            "Minecraft lapis lazuli ore block with blue speckles being mined",
            "Minecraft player mining lapis lazuli ore underground",
            "Minecraft blue ore vein in a cave"
        ],
        "minería_diamante": [
            "Minecraft diamond ore block with cyan blue speckles being mined",
            "Minecraft player mining diamond ore underground",
            "Minecraft deepslate diamond ore vein in a cave"
        ],
        "minería_esmeralda": [
            "Minecraft emerald ore block with green speckles being mined",
            "Minecraft player mining emerald ore in mountains or caves",
            "Minecraft rare green emerald ore vein"
        ],
        "minería_escombros": [
            "Minecraft ancient debris block being mined in the Nether",
            "Minecraft player mining ancient debris for netherite",
            "Minecraft brown ancient debris block in netherrack"
        ],
        "construcción": [
            "Minecraft player building a house bridge wall or structure with blocks",
            "Minecraft construction scene placing blocks",
            "Minecraft base building or architecture gameplay",
            "Minecraft player creating a structure block by block"
        ],
        "crafteo": [
            "Minecraft crafting table interface open",
            "Minecraft player crafting tools weapons or items",
            "Minecraft 3 by 3 crafting grid with items"
        ],
        "fundición": [
            "Minecraft furnace blast furnace or smoker interface open",
            "Minecraft smelting ores into ingots in a furnace",
            "Minecraft furnace cooking food or processing materials"
        ],
        "cultivo": [
            "Minecraft farming crops wheat carrots potatoes beetroot or seeds",
            "Minecraft player planting or harvesting crops",
            "Minecraft farmland with crops and water"
        ],
        "pesca": [
            "Minecraft player fishing with a fishing rod",
            "Minecraft fishing bobber in water",
            "Minecraft player standing near water waiting for fish"
        ],
        "ganadería": [
            "Minecraft player farming animals cows sheep pigs chickens or horses",
            "Minecraft animal pen with livestock",
            "Minecraft breeding feeding or collecting resources from animals"
        ],
        "cofre": [
            "Minecraft chest inventory interface open",
            "Minecraft player looking inside a chest with item slots",
            "Minecraft double chest storage screen open"
        ],
        "encantamiento": [
            "Minecraft enchantment table interface open",
            "Minecraft player enchanting a tool weapon armor or book",
            "Minecraft enchantment table with lapis lazuli and experience levels"
        ],
        "pociones": [
            "Minecraft brewing stand interface open",
            "Minecraft player brewing potions with bottles and ingredients",
            "Minecraft potion brewing screen with blaze powder"
        ],
        "comer": [
            "Minecraft player eating food in first person",
            "Minecraft food item held near the screen while eating",
            "Minecraft eating animation with bread meat apple or golden carrot"
        ],
        "dormir": [
            "Minecraft bed sleep screen or player in a bed",
            "Minecraft player trying to sleep at night",
            "Minecraft bedroom with bed used to skip the night"
        ],
        "nadar": [
            "Minecraft player swimming underwater",
            "Minecraft underwater gameplay with oxygen bubbles",
            "Minecraft player moving through water or ocean"
        ],
        "bote": [
            "Minecraft player riding a boat on water",
            "Minecraft boat travel across river ocean or lake",
            "Minecraft first person view from inside a boat"
        ],
        "portal": [
            "Minecraft nether portal purple blocks active",
            "Minecraft player standing near an active portal",
            "Minecraft obsidian portal with purple swirling effect"
        ],
        "nether": [
            "Minecraft Nether dimension with netherrack lava and fire",
            "Minecraft crimson forest warped forest basalt delta or nether wastes",
            "Minecraft gameplay in the Nether with lava and hostile mobs"
        ],
        "aldea": [
            "Minecraft village with houses villagers farms and paths",
            "Minecraft player walking inside a village",
            "Minecraft village buildings and villagers visible"
        ],
        "comercio": [
            "Minecraft villager trading interface open",
            "Minecraft player trading emeralds with a villager",
            "Minecraft merchant trade screen with offers"
        ],
    },
}


PRIORS_ESTADO = {
    "minecraft": {
        "combate": 1.10,
        "exploración": 1.00,
        "talar": 1.05,
        "peligro": 0.85,
        "menú": 1.00,
        "victoria": 0.12,
        "minería": 0.95,
        "minería_carbón": 0.90,
        "minería_hierro": 0.90,
        "minería_cobre": 0.90,
        "minería_oro": 0.90,
        "minería_redstone": 0.95,
        "minería_lapislázuli": 0.95,
        "minería_diamante": 1.00,
        "minería_esmeralda": 0.85,
        "minería_escombros": 0.90,
        "construcción": 1.00,
        "crafteo": 1.05,
        "fundición": 1.05,
        "cultivo": 1.00,
        "pesca": 1.00,
        "ganadería": 1.00,
        "cofre": 1.05,
        "encantamiento": 1.05,
        "pociones": 1.05,
        "comer": 0.95,
        "dormir": 1.00,
        "nadar": 1.00,
        "bote": 1.00,
        "portal": 1.05,
        "nether": 1.00,
        "aldea": 1.00,
        "comercio": 1.05,
    },
}

UMBRAL_ESTADOS_RAROS = {
    "minecraft": {
        "peligro": 0.38,
        "victoria": 0.50,
        "minería_diamante": 0.22,
        "minería_esmeralda": 0.22,
        "minería_escombros": 0.22,
        "comer": 0.24,
        "portal": 0.24,
    },
}


ETIQUETAS_JUEGOS = {
    "minecraft": [
        "combate", "exploración", "talar", "peligro", "menú", "victoria",
        "minería", "minería_carbón", "minería_hierro", "minería_cobre",
        "minería_oro", "minería_redstone", "minería_lapislázuli",
        "minería_diamante", "minería_esmeralda", "minería_escombros",
        "construcción", "crafteo", "fundición", "cultivo", "pesca", "ganadería",
        "cofre", "encantamiento", "pociones", "comer", "dormir", "nadar",
        "bote", "portal", "nether", "aldea", "comercio"
    ]
}


DESCRIPCIONES_ESTADO = {
    "minecraft": {
        "combate": "el jugador esta peleando contra enemigos",
        "exploración": "el jugador explora el mapa y busca recursos",
        "talar": "el jugador corta arboles y junta madera",
        "peligro": "el jugador tiene poca vida o esta por morir",
        "menú": "el jugador revisa inventario, crafteo o un menu",
        "victoria": "el jugador acaba de resolver una jugada importante",
        "minería": "el jugador mina bloques u minerales bajo tierra",
        "minería_carbón": "el jugador mina carbon para combustible o antorchas",
        "minería_hierro": "el jugador mina hierro para herramientas y armadura",
        "minería_cobre": "el jugador mina cobre",
        "minería_oro": "el jugador mina oro",
        "minería_redstone": "el jugador mina redstone para mecanismos",
        "minería_lapislázuli": "el jugador mina lapislazuli para encantamientos",
        "minería_diamante": "el jugador encuentra y mina diamante",
        "minería_esmeralda": "el jugador mina esmeralda",
        "minería_escombros": "el jugador mina escombros ancestrales para netherita",
        "construcción": "el jugador construye una estructura con bloques",
        "crafteo": "el jugador usa la mesa de crafteo para crear objetos",
        "fundición": "el jugador usa horno para cocinar o fundir materiales",
        "cultivo": "el jugador siembra o cosecha cultivos",
        "pesca": "el jugador pesca con caña",
        "ganadería": "el jugador interactua con animales o una granja",
        "cofre": "el jugador abre o administra un cofre",
        "encantamiento": "el jugador usa una mesa de encantamientos",
        "pociones": "el jugador prepara pociones",
        "comer": "el jugador come para recuperar hambre o vida",
        "dormir": "el jugador usa una cama",
        "nadar": "el jugador nada o esta bajo el agua",
        "bote": "el jugador navega en bote",
        "portal": "el jugador usa o mira un portal activo",
        "nether": "el jugador esta en el Nether",
        "aldea": "el jugador encuentra o explora una aldea",
        "comercio": "el jugador comercia con aldeanos",
    },
}

PALABRAS_CLAVE_COMENTARIO = {
    "combate": {
        "pelea", "combate", "golpe", "daño", "ataca", "atacan",
        "enemigo", "enemigos", "mobs", "espada", "arco", "escudo", "presion"
    },
    "exploración": {
        "explora", "explorando", "mapa", "ruta", "terreno", "recursos",
        "bioma", "cueva", "mueve"
    },
    "talar": {
        "tala", "talando", "arbol", "arboles", "madera", "troncos",
        "hacha", "logs", "wood"
    },
    "peligro": {
        "peligro", "vida", "poca", "baja", "morir", "caer", "escapa",
        "escapar", "sobrevive", "limite", "cuidado"
    },
    "menú": {
        "menu", "inventario", "objetos", "craftear", "pausa", "revisa"
    },
    "victoria": {
        "victoria", "gana", "logro", "cierra", "cumplido", "celebracion", "final"
    },
    "minería": {
        "mina", "minando", "pico", "piedra", "diamante", "mineral",
        "cueva", "subterranea", "bloques"
    },
    "minería_carbón": {"carbon", "coal", "antorchas", "combustible"},
    "minería_hierro": {"hierro", "iron", "lingote", "armadura"},
    "minería_cobre": {"cobre", "copper"},
    "minería_oro": {"oro", "gold", "dorado"},
    "minería_redstone": {"redstone", "rojo", "circuitos", "mecanismos"},
    "minería_lapislázuli": {"lapis", "lapislazuli", "azul", "encantamientos"},
    "minería_diamante": {"diamante", "diamond", "cian", "azul"},
    "minería_esmeralda": {"esmeralda", "emerald", "verde", "aldeanos"},
    "minería_escombros": {"escombros", "ancient", "debris", "netherita", "nether"},
    "construcción": {
        "construye", "construccion", "bloques", "estructura", "base",
        "casa", "coloca", "levanta"
    },
    "crafteo": {"crafteo", "craftear", "mesa", "herramienta", "receta", "crafting"},
    "fundición": {"horno", "fundiendo", "fundicion", "lingotes", "cocinar", "smelting"},
    "cultivo": {"cultivo", "cultivos", "granja", "sembrando", "cosecha", "trigo", "zanahoria"},
    "pesca": {"pesca", "pescando", "cana", "anzuelo", "pez", "agua"},
    "ganadería": {"animales", "ganado", "vacas", "ovejas", "cerdos", "gallinas", "crianza"},
    "cofre": {"cofre", "almacenamiento", "guardar", "recursos", "inventario"},
    "encantamiento": {"encantamiento", "encantar", "mesa", "lapis", "experiencia", "equipo"},
    "pociones": {"pocion", "pociones", "brewing", "soporte", "ingredientes", "alquimia"},
    "comer": {"comer", "comida", "hambre", "curando", "pan", "carne", "manzana"},
    "dormir": {"dormir", "cama", "noche", "aparicion", "descanso"},
    "nadar": {"nadar", "nadando", "agua", "oxigeno", "respiracion", "acuatico"},
    "bote": {"bote", "navegacion", "navega", "agua", "rio", "oceano"},
    "portal": {"portal", "obsidiana", "dimension", "morado", "cruzar"},
    "nether": {"nether", "lava", "netherrack", "infierno", "dimension"},
    "aldea": {"aldea", "aldeanos", "casas", "villa", "pueblo"},
    "comercio": {"comercio", "aldeano", "aldeanos", "intercambio", "esmeraldas", "ofertas"},
}


def init_detector():
    """
    Inicializa el modelo zero-shot de vision para clasificar frames con prompts descriptivos.
    """
    global detector, clip_processor
    if detector is None:
        print(f"[IA Engine] Cargando modelo de vision ({VISION_MODEL_NAME}) en {DEVICE}...")
        model_kwargs = {"torch_dtype": torch.float16} if DEVICE == "cuda" else {}
        detector = AutoModel.from_pretrained(VISION_MODEL_NAME, **model_kwargs)
        detector.eval()
        if DEVICE == "cuda":
            detector = detector.to("cuda")
        clip_processor = AutoProcessor.from_pretrained(VISION_MODEL_NAME)
    return detector


def init_ensemble_detector():
    """
    Carga un segundo modelo de vision para hacer ensemble si esta configurado.
    """
    global ensemble_detector, ensemble_processor
    if not VISION_ENSEMBLE_MODEL_NAME or VISION_ENSEMBLE_MODEL_NAME == VISION_MODEL_NAME:
        return None, None
    if ensemble_detector is None:
        print(f"[IA Engine] Cargando modelo de ensemble ({VISION_ENSEMBLE_MODEL_NAME}) en {DEVICE}...")
        model_kwargs = {"torch_dtype": torch.float16} if DEVICE == "cuda" else {}
        ensemble_detector = AutoModel.from_pretrained(VISION_ENSEMBLE_MODEL_NAME, **model_kwargs)
        ensemble_detector.eval()
        if DEVICE == "cuda":
            ensemble_detector = ensemble_detector.to("cuda")
        ensemble_processor = AutoProcessor.from_pretrained(VISION_ENSEMBLE_MODEL_NAME)
    return ensemble_detector, ensemble_processor


def init_narrador():
    """
    Inicializa un modelo NLP de Hugging Face para generar comentarios creativos.
    Si el modelo no esta descargado o falla, se conserva el fallback por plantillas.
    """
    global narrador, narrador_error
    if narrador is None and narrador_error is None:
        try:
            print(f"[IA Engine] Cargando narrador NLP ({NLP_MODEL_NAME}) en {DEVICE}...")
            narrador = pipeline(
                NLP_TASK_NAME,
                model=NLP_MODEL_NAME,
                device=DEVICE_ID,
            )
            if hasattr(narrador, "tokenizer") and narrador.tokenizer.pad_token_id is None:
                narrador.tokenizer.pad_token = narrador.tokenizer.eos_token
        except Exception as exc:
            narrador_error = str(exc)
            narrador = None
            print(f"[IA Engine] No se pudo cargar el narrador NLP: {exc}")
    return narrador


def init_tts():
    """
    Inicializa el modelo de Texto a Voz (MMS-TTS-SPA) de Meta.
    """
    global tts_tokenizer, tts_model
    if tts_model is None:
        print(f"[IA Engine] Cargando MMS-TTS-SPA ({TTS_MODEL_NAME}) en {DEVICE}...")
        tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_NAME)
        tts_model = VitsModel.from_pretrained(TTS_MODEL_NAME)
        tts_model.eval()
        if DEVICE == "cuda":
            tts_model = tts_model.to("cuda")
    return tts_tokenizer, tts_model


def _mover_inputs_a_dispositivo(inputs, model=None):
    model = model or detector
    model_dtype = next(model.parameters()).dtype
    moved = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        value = value.to(DEVICE)
        if key == "pixel_values" and value.is_floating_point():
            value = value.to(dtype=model_dtype)
        moved[key] = value
    return moved


def _recortes_para_clasificacion(imagen_pil, juego):
    """
    Clasifica con varios recortes. En Minecraft el centro de pantalla suele
    contener el bloque/accion real; el frame completo conserva contexto.
    """
    imagen_pil = imagen_pil.convert("RGB")
    ancho, alto = imagen_pil.size

    if juego != "minecraft" or ancho < 320 or alto < 240:
        return [(imagen_pil, 1.0)]

    centro = imagen_pil.crop((
        int(ancho * 0.20),
        int(alto * 0.12),
        int(ancho * 0.80),
        int(alto * 0.82),
    ))
    punto_mira = imagen_pil.crop((
        int(ancho * 0.34),
        int(alto * 0.25),
        int(ancho * 0.66),
        int(alto * 0.68),
    ))
    sin_hotbar = imagen_pil.crop((
        int(ancho * 0.03),
        int(alto * 0.03),
        int(ancho * 0.97),
        int(alto * 0.86),
    ))
    return [(imagen_pil, 0.20), (sin_hotbar, 0.25), (centro, 0.30), (punto_mira, 0.25)]


def _prompts_personalizados(etiquetas_personalizadas):
    prompts = []
    owners = []
    for etiqueta in etiquetas_personalizadas:
        etiqueta = etiqueta.strip()
        if not etiqueta:
            continue
        variantes = [
            f"a video game screenshot showing {etiqueta}",
            f"gameplay scene: {etiqueta}",
            f"current game state: {etiqueta}",
        ]
        prompts.extend(variantes)
        owners.extend([etiqueta] * len(variantes))
    return prompts, owners


def _prompts_por_juego(juego):
    prompts = []
    owners = []
    escenas = PROMPTS_CLIP.get(juego, PROMPTS_CLIP["minecraft"])
    for estado, variantes in escenas.items():
        prompts.extend(variantes)
        owners.extend([estado] * len(variantes))
    return prompts, owners


def _agrupar_scores_por_estado(probabilidades, owners):
    scores = defaultdict(list)
    for probabilidad, estado in zip(probabilidades, owners):
        scores[estado].append(float(probabilidad))

    agregados = {}
    for estado, valores in scores.items():
        mejores = sorted(valores, reverse=True)[:2]
        agregados[estado] = sum(mejores) / len(mejores)

    total = sum(agregados.values()) or 1.0
    return {estado: valor / total for estado, valor in agregados.items()}


def _normalizar_features(features):
    return features / features.norm(dim=-1, keepdim=True)


def _escala_logit_modelo(model=None):
    model = model or detector
    escala = getattr(model, "logit_scale", None)
    if escala is None:
        return 1.0
    try:
        return float(escala.exp().detach().float().cpu())
    except Exception:
        try:
            return float(escala.detach().float().cpu())
        except Exception:
            return 1.0


def _logits_imagen_texto(imagen_pil, prompts, model=None, processor=None):
    model = model or detector
    processor = processor or clip_processor
    inputs = processor(
        text=prompts,
        images=imagen_pil.convert("RGB"),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = _mover_inputs_a_dispositivo(inputs, model)

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = getattr(outputs, "logits_per_image", None)
        if logits is not None:
            return logits[0]

        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_kwargs = {
            key: value
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask", "position_ids", "token_type_ids"}
        }
        text_features = model.get_text_features(**text_kwargs)
        image_features = _normalizar_features(image_features)
        text_features = _normalizar_features(text_features)
        return (image_features @ text_features.T)[0] * _escala_logit_modelo(model)


def _clasificar_clip(imagen_pil, juego="minecraft", etiquetas_personalizadas=None):
    init_detector()
    modelos_vision = [(detector, clip_processor, 0.68)]
    try:
        detector_secundario, processor_secundario = init_ensemble_detector()
        if detector_secundario is not None and processor_secundario is not None:
            modelos_vision.append((detector_secundario, processor_secundario, 0.32))
    except Exception as exc:
        print(f"[IA Engine] Ensemble de vision desactivado: {exc}")

    imagen_pil = imagen_pil.convert("RGB")

    if etiquetas_personalizadas:
        prompts, owners = _prompts_personalizados(etiquetas_personalizadas)
    else:
        prompts, owners = _prompts_por_juego(juego)

    if not prompts:
        fallback = etiquetas_personalizadas[0] if etiquetas_personalizadas else "exploración"
        return fallback, 0.0, []

    scores_acumulados = defaultdict(float)
    peso_total = 0.0
    for crop, peso in _recortes_para_clasificacion(imagen_pil, juego):
        for model, processor, peso_modelo in modelos_vision:
            logits = _logits_imagen_texto(crop, prompts, model, processor)
            probabilidades = torch.softmax(logits, dim=0).detach().cpu().tolist()
            scores_crop = _agrupar_scores_por_estado(probabilidades, owners)
            peso_final = peso * peso_modelo
            for estado_crop, score_crop in scores_crop.items():
                scores_acumulados[estado_crop] += score_crop * peso_final
            peso_total += peso_final

    normalizador = peso_total or 1.0
    scores = {estado: score / normalizador for estado, score in scores_acumulados.items()}
    if not etiquetas_personalizadas:
        priors = PRIORS_ESTADO.get(juego, {})
        scores = {estado: score * priors.get(estado, 1.0) for estado, score in scores.items()}
        total = sum(scores.values()) or 1.0
        scores = {estado: score / total for estado, score in scores.items()}
    ordenados = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    estado, confianza = ordenados[0]

    if not etiquetas_personalizadas:
        umbrales_raros = UMBRAL_ESTADOS_RAROS.get(juego, {})
        umbral = umbrales_raros.get(estado)
        if umbral is not None and confianza < umbral and len(ordenados) > 1:
            for candidato_estado, candidato_confianza in ordenados[1:]:
                candidato_umbral = umbrales_raros.get(candidato_estado, 0.0)
                if candidato_confianza >= candidato_umbral:
                    estado, confianza = candidato_estado, candidato_confianza
                    break

    return estado, float(confianza), ordenados


def analizar_cuadro(imagen_pil, juego="minecraft", etiquetas_personalizadas=None):
    """
    Analiza un frame con CLIP zero-shot y retorna estado + confianza real.
    """
    global ultima_prediccion
    try:
        estado, confianza, ranking = _clasificar_clip(
            imagen_pil,
            juego=juego,
            etiquetas_personalizadas=etiquetas_personalizadas,
        )
        ultima_prediccion = {
            "modelo": (
                f"{VISION_MODEL_NAME} + {VISION_ENSEMBLE_MODEL_NAME}"
                if VISION_ENSEMBLE_MODEL_NAME and VISION_ENSEMBLE_MODEL_NAME != VISION_MODEL_NAME
                else VISION_MODEL_NAME
            ),
            "estado": estado,
            "confianza": confianza,
            "scores": ranking[:5],
        }
        return estado, round(confianza, 3)
    except Exception as exc:
        print(f"[IA Engine] Error en CLIP: {exc}")
        fallback = etiquetas_personalizadas[0] if etiquetas_personalizadas else "exploración"
        ultima_prediccion = {
            "modelo": VISION_MODEL_NAME,
            "estado": fallback,
            "confianza": 0.0,
            "scores": [],
            "error": str(exc),
        }
        return fallback, 0.0


def obtener_ultima_prediccion():
    return dict(ultima_prediccion)


def estado_narrador_nlp():
    return {
        "modelo": NLP_MODEL_NAME,
        "tarea": NLP_TASK_NAME,
        "activo": narrador is not None,
        "error": narrador_error,
    }


def _comentario_base(estado, juego):
    plantillas = PLANTILLAS_COMENTARIOS.get(juego, PLANTILLAS_COMENTARIOS["minecraft"])
    lista_comentarios = plantillas.get(estado, ["Jugada interesante en pantalla."])
    return random.choice(lista_comentarios)


def _decorar_por_confianza(texto, confianza):
    adornos_altos = ["¡Ojo!", "¡Atencion!", "¡Wow!", "¡Vamos!"]
    adornos_duda = ["Parece que", "Creo que"]

    if confianza > 0.55 and random.random() > 0.65:
        return f"{random.choice(adornos_altos)} {texto}"
    if confianza < 0.22:
        texto = texto[:1].lower() + texto[1:]
        return f"{random.choice(adornos_duda)} {texto}"
    return texto


def _limpiar_generacion(texto):
    texto = texto.replace("\r", " ").replace("\n", " ")
    texto = re.sub(r"\s+", " ", texto).strip(" \"'`")
    texto = re.sub(r"^(comentario|narrador|frase)\s*:\s*", "", texto, flags=re.IGNORECASE)
    texto = re.split(
        r"\b(Juego|Estado|Situacion|Situación|Ejemplo|Comentario)\s*:",
        texto,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" \"'`")

    if not texto:
        return None

    partes = re.split(r"(?<=[.!?])\s+", texto)
    texto = partes[0].strip()
    if len(texto) > 130:
        texto = texto[:130].rsplit(" ", 1)[0].strip()
    if texto and texto[-1] not in ".!?":
        texto = f"{texto}!"

    texto_lower = texto.lower()
    tokens = re.findall(r"[a-záéíóúñü]+", texto_lower)
    if len(texto) < 10 or len(texto) > 140:
        return None
    if not re.search(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]", texto):
        return None
    if "http" in texto_lower or "www." in texto_lower:
        return None
    if ":" in texto[:24]:
        return None
    if any(marcador in texto_lower for marcador in ["estado:", "juego:", "tono:", "contexto:", "comentario de caster"]):
        return None
    if len(tokens) >= 7 and len(set(tokens)) / len(tokens) < 0.45:
        return None
    return texto


def _normalizar_texto(texto):
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = texto.encode("ascii", "ignore").decode("ascii")
    return texto.lower()


def _palabras_clave_estado(estado):
    estado_norm = _normalizar_texto(estado)
    for clave, palabras in PALABRAS_CLAVE_COMENTARIO.items():
        if _normalizar_texto(clave) == estado_norm:
            return {_normalizar_texto(palabra) for palabra in palabras}
    return set()


def _comentario_es_relevante(comentario, estado):
    palabras = _palabras_clave_estado(estado)
    if not palabras:
        return True

    texto = _normalizar_texto(comentario)
    return any(palabra in texto for palabra in palabras)


def _generar_comentario_nlp(estado, confianza, juego, comentario_base):
    generator = init_narrador()
    if generator is None:
        return None

    nombre_juego = "Minecraft"
    descripcion = DESCRIPCIONES_ESTADO.get(juego, {}).get(estado, estado)
    tono = "muy urgente" if estado in {"combate", "peligro", "victoria"} else "claro y breve"
    prompt = (
        "Genera una sola frase corta en español para narrar un clip de videojuego. "
        "No inventes nombres, equipos ni otros deportes. "
        f"Videojuego: {nombre_juego}. "
        f"Estado detectado: {estado}. "
        f"Contexto real: {descripcion}. "
        f"Tono: {tono}. "
        f"Estilo permitido: {comentario_base}. "
        "Responde solo el comentario final en maximo 16 palabras."
    )

    try:
        resultado = generator(
            prompt,
            max_new_tokens=32,
            do_sample=False,
            repetition_penalty=1.05,
            num_return_sequences=1,
            pad_token_id=getattr(generator.tokenizer, "eos_token_id", None),
        )
        texto_completo = resultado[0].get("generated_text", "")
        generado = texto_completo[len(prompt):] if texto_completo.startswith(prompt) else texto_completo
        comentario = _limpiar_generacion(generado)
        if comentario and _comentario_es_relevante(comentario, estado):
            return _decorar_por_confianza(comentario, confianza)
    except Exception as exc:
        print(f"[IA Engine] Error generando comentario NLP: {exc}")
    return None


def generar_comentario(estado, confianza, juego="minecraft", usar_nlp=True):
    """
    Genera un comentario corto a partir del estado detectado. Por defecto intenta
    usar un modelo NLP de Hugging Face y cae a plantillas si la generacion no es limpia.
    """
    base = _comentario_base(estado, juego)
    if usar_nlp:
        generado = _generar_comentario_nlp(estado, confianza, juego, base)
        if generado:
            return generado
    return _decorar_por_confianza(base, confianza)


def sintetizar_voz(texto, output_folder="temp_audio", filename="comentario.wav"):
    """
    Sintetiza texto en voz en español y lo guarda en un archivo WAV.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)

    tokenizer, model = init_tts()
    inputs = tokenizer(texto, return_tensors="pt")

    if DEVICE == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.inference_mode():
        output = model(**inputs).waveform

    audio_numpy = output.cpu().numpy().squeeze()
    wavfile.write(output_path, rate=model.config.sampling_rate, data=audio_numpy)
    return output_path
