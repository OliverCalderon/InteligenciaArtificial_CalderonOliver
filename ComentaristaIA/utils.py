import cv2
import numpy as np
import os
import unicodedata
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
except ImportError:
    from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

def extraer_info_video(video_path):
    """
    Extrae información general del video: FPS original, número total de frames y duración.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion = total_frames / fps if fps > 0 else 0
    cap.release()
    return {"fps": fps, "total_frames": total_frames, "duracion": duracion}

def calcular_similitud_hsv(frame1, frame2):
    """
    Compara dos frames en formato RGB utilizando histogramas en el espacio de color HSV.
    Retorna un valor de similitud entre -1.0 y 1.0 (1.0 indica que son idénticos).
    """
    # Convertir de RGB a BGR y luego a HSV
    frame1_bgr = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
    frame2_bgr = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
    
    hsv1 = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2HSV)
    
    # Calcular histogramas para H (Hue) y S (Saturation)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    
    # Normalizar histogramas para poder compararlos
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    # Comparar usando la correlación de histogramas
    similitud = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similitud

def suavizar_predicciones(historial, ultimo_estado=None, margen_cambio=0.12):
    """
    Suaviza varias predicciones recientes para evitar saltos de etiqueta entre
    frames consecutivos. `historial` debe contener tuplas (estado, confianza).
    """
    if not historial:
        return ultimo_estado, 0.0

    pesos = {}
    mejor_confianza_por_estado = {}
    for estado, confianza in historial:
        confianza = max(float(confianza), 0.01)
        pesos[estado] = pesos.get(estado, 0.0) + confianza
        mejor_confianza_por_estado[estado] = max(
            mejor_confianza_por_estado.get(estado, 0.0),
            confianza
        )

    total = sum(pesos.values()) or 1.0
    ordenados = sorted(pesos.items(), key=lambda item: item[1], reverse=True)
    mejor_estado, mejor_peso = ordenados[0]

    if ultimo_estado and mejor_estado != ultimo_estado:
        peso_ultimo = pesos.get(ultimo_estado, 0.0)
        ventaja = (mejor_peso - peso_ultimo) / total
        if peso_ultimo > 0 and ventaja < margen_cambio:
            confianza = max(peso_ultimo / total, mejor_confianza_por_estado.get(ultimo_estado, 0.0) * 0.9)
            return ultimo_estado, min(0.99, confianza)

    confianza = max(mejor_peso / total, mejor_confianza_por_estado.get(mejor_estado, 0.0))
    return mejor_estado, min(0.99, confianza)

def slug_estado(estado):
    """
    Convierte una etiqueta de estado a una clase CSS estable sin acentos.
    """
    normalizado = unicodedata.normalize("NFKD", str(estado))
    ascii_text = normalizado.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower().replace(" / ", "-").replace(" ", "-")

def ajustar_volumen(clip, factor):
    """
    Ajusta el volumen de un clip de audio de forma compatible con MoviePy v1 y v2.
    """
    if hasattr(clip, "with_volume_scaled"):
        return clip.with_volume_scaled(factor)
    elif hasattr(clip, "volumex"):
        return clip.volumex(factor)
    elif hasattr(clip, "multiply_volume"):
        return clip.multiply_volume(factor)
    return clip

def establecer_inicio_clip(clip, timestamp):
    """
    Establece el segundo de inicio de un clip de forma compatible con MoviePy v1 y v2.
    """
    if hasattr(clip, "with_start"):
        return clip.with_start(timestamp)
    return clip.set_start(timestamp)

def establecer_audio_video(video_clip, audio_clip):
    """
    Asigna la pista de audio a un video de forma compatible con MoviePy v1 y v2.
    """
    if hasattr(video_clip, "with_audio"):
        return video_clip.with_audio(audio_clip)
    return video_clip.set_audio(audio_clip)

def mezclar_narracion_video(video_path, audios_info, output_path, volumen_juego=0.2):
    """
    Mezcla múltiples pistas de audio individuales (.wav) en tiempos específicos
    sobre el video original y guarda el resultado.
    
    audios_info: Lista de tuplas/diccionarios [(timestamp_segundos, ruta_audio_wav), ...]
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"El video no existe: {video_path}")
        
    video = VideoFileClip(video_path)
    audio_clips = []
    
    # Reducir el volumen del video original para no tapar al narrador
    if video.audio is not None:
        video_audio = ajustar_volumen(video.audio, volumen_juego)
        audio_clips.append(video_audio)
        
    for timestamp, audio_path in audios_info:
        if os.path.exists(audio_path):
            # Cargar el clip de audio de la IA e indicar en qué segundo debe iniciar
            ai_audio = establecer_inicio_clip(AudioFileClip(audio_path), timestamp)
            audio_clips.append(ai_audio)
            
    if not audio_clips:
        # Si no hay audios que mezclar, guardamos el video tal cual
        video.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
        return
        
    # Crear una mezcla compuesta de audios
    audio_compuesto = CompositeAudioClip(audio_clips)
    
    # Asignar la mezcla de audio al video original
    video_final = establecer_audio_video(video, audio_compuesto)
    
    # Escribir el archivo final. Ajustamos los parámetros para compatibilidad óptima
    video_final.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac",
        temp_audiofile="temp-audio-render.m4a",
        remove_temp=True,
        logger=None
    )
    
    # Cerrar clips para liberar los archivos
    video.close()
    video_final.close()
    for ac in audio_clips:
        ac.close()
