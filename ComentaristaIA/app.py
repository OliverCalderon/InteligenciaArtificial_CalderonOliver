import streamlit as st
import cv2
import os
import time
import base64
import tempfile
from collections import deque
from PIL import Image
import torch

# Importar nuestros módulos locales
import ia_engine
import utils

# Configuración inicial de la página Streamlit (Debe ser la primera llamada de Streamlit)
st.set_page_config(
    page_title="ComentaristaIA - Narrador de Videojuegos",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS (Aesthetics: Dark Mode, Glassmorphism, Neon Gradients) ---
st.markdown("""
<style>
    /* Importar fuente Outfit */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Configuración global */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Fondo principal y barra lateral */
    .stApp {
        background-color: #0b0c10;
        color: #c5c6c7;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #1f2833;
        border-right: 1px solid #45f3ff33;
    }
    
    /* Título con gradiente de neón */
    .neon-title {
        background: linear-gradient(45deg, #00f2fe, #4facfe, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 5px;
        text-align: center;
        text-shadow: 0 0 20px rgba(79, 172, 254, 0.2);
    }
    
    .neon-subtitle {
        color: #4facfe;
        text-align: center;
        font-weight: 400;
        margin-bottom: 25px;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Tarjetas tipo Glassmorphism */
    .glass-card {
        background: rgba(31, 40, 51, 0.45);
        border: 1px solid rgba(79, 172, 254, 0.15);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    
    /* Badges de estado del Caster */
    .caster-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 30px;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 1px;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
    }
    
    .badge-combate {
        background-color: rgba(255, 75, 75, 0.25);
        color: #ff4b4b;
        border: 1px solid #ff4b4b;
    }

    .badge-exploracion {
        background-color: rgba(9, 171, 59, 0.25);
        color: #09ab3b;
        border: 1px solid #09ab3b;
    }
    
    .badge-peligro {
        background-color: rgba(255, 165, 0, 0.25);
        color: #ffa500;
        border: 1px solid #ffa500;
    }
    
    .badge-menu {
        background-color: rgba(168, 85, 247, 0.25);
        color: #c084fc;
        border: 1px solid #a855f7;
    }
    
    .badge-celebracion,
    .badge-victoria {
        background-color: rgba(6, 182, 212, 0.25);
        color: #22d3ee;
        border: 1px solid #06b6d4;
    }
    
    .badge-mineria {
        background-color: rgba(234, 179, 8, 0.25);
        color: #facc15;
        border: 1px solid #eab308;
    }

    .caster-badge[class*="badge-mineria_"] {
        background-color: rgba(245, 158, 11, 0.25);
        color: #fbbf24;
        border: 1px solid #f59e0b;
    }

    .badge-construccion {
        background-color: rgba(20, 184, 166, 0.25);
        color: #2dd4bf;
        border: 1px solid #14b8a6;
    }

    .badge-talar,
    .badge-cultivo,
    .badge-ganaderia {
        background-color: rgba(22, 163, 74, 0.25);
        color: #86efac;
        border: 1px solid #16a34a;
    }

    .badge-crafteo,
    .badge-fundicion,
    .badge-pesca,
    .badge-cofre,
    .badge-encantamiento,
    .badge-pociones {
        background-color: rgba(14, 165, 233, 0.25);
        color: #7dd3fc;
        border: 1px solid #0ea5e9;
    }

    .badge-comer,
    .badge-dormir {
        background-color: rgba(244, 114, 182, 0.25);
        color: #f9a8d4;
        border: 1px solid #f472b6;
    }

    .badge-nadar,
    .badge-bote {
        background-color: rgba(6, 182, 212, 0.25);
        color: #67e8f9;
        border: 1px solid #06b6d4;
    }

    .badge-portal,
    .badge-nether {
        background-color: rgba(168, 85, 247, 0.25);
        color: #d8b4fe;
        border: 1px solid #a855f7;
    }

    .badge-aldea,
    .badge-comercio {
        background-color: rgba(202, 138, 4, 0.25);
        color: #fde68a;
        border: 1px solid #ca8a04;
    }

</style>
""", unsafe_allow_html=True)

# --- ESTADO DE SESIÓN DE STREAMLIT ---
if "comentarios" not in st.session_state:
    st.session_state.comentarios = []
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "video_name" not in st.session_state:
    st.session_state.video_name = None
if "video_exportado" not in st.session_state:
    st.session_state.video_exportado = None
if "dispositivo_detectado" not in st.session_state:
    st.session_state.dispositivo_detectado = ia_engine.DEVICE

# Función para reproducir audio automáticamente inyectando un tag HTML5 en un placeholder
def reproducir_audio_automatico(ruta_audio, placeholder):
    with open(ruta_audio, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        # Generar tag HTML con autoplay
        html_code = f"""
            <audio autoplay="true" style="display:none;">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
        """
        placeholder.markdown(html_code, unsafe_allow_html=True)

# Limpiar audios temporales anteriores
def limpiar_audios_temporales():
    carpetas = {
        "temp_audio": (".wav",),
        "temp_frames": (".jpg", ".jpeg", ".png"),
    }
    for folder, extensiones in carpetas.items():
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if not file.lower().endswith(extensiones):
                    continue
                try:
                    os.remove(os.path.join(folder, file))
                except Exception:
                    pass


def guardar_frame_comentado(frame_rgb, timestamp_seg):
    os.makedirs("temp_frames", exist_ok=True)
    frame_filename = f"frame_{int(timestamp_seg * 1000)}ms.jpg"
    frame_path = os.path.join("temp_frames", frame_filename)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(frame_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return frame_path


def formatear_tiempo(segundos):
    minutos = int(segundos // 60)
    segundos_restantes = int(segundos % 60)
    return f"{minutos:02d}:{segundos_restantes:02d}"

# --- HEADER PRINCIPAL ---
st.markdown('<div class="neon-title">🎙️ ComentaristaIA</div>', unsafe_allow_html=True)
st.markdown('<div class="neon-subtitle">Narrador de Videojuegos Inteligente con Visión y Voz Multimodal</div>', unsafe_allow_html=True)

# Mostrar estado de hardware
hardware_icon = "🚀" if torch.cuda.is_available() else "🐌"
st.sidebar.markdown(f"**Hardware activo:** {hardware_icon} `{st.session_state.dispositivo_detectado.upper()}`")

# --- BARRA LATERAL (CONFIGURACIÓN) ---
st.sidebar.header("🕹️ Panel de Configuración")

# Proyecto enfocado en Minecraft para mejorar precisión y reducir ruido de etiquetas.
juego_seleccionado = "minecraft"
st.sidebar.markdown("**Videojuego:** Minecraft ⛏️")

# Modo de Reproducción/Procesamiento
modo_tiempo_real = st.sidebar.toggle(
    "Simular Transmisión en Vivo",
    value=True,
    help="Sincroniza la consola con el tiempo del video (1x) y reproduce la voz de la IA al instante. Si se desactiva, el procesamiento será lo más rápido posible y no se reproducirá el audio en tiempo real (ideal para exportar rápidamente)."
)

# Carga de Video
uploaded_file = st.sidebar.file_uploader(
    "Sube el video de gameplay (.mp4):",
    type=["mp4"]
)

# Parámetros del Narrador (Heurísticas)
st.sidebar.subheader("⚙️ Parámetros de la IA")

fps_analisis = st.sidebar.slider(
    "Frames analizados por segundo (FPS de IA):",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.5,
    help="Con qué frecuencia procesamos un frame. 1 FPS significa evaluar 1 frame cada segundo de video."
)

sensibilidad_cambio = st.sidebar.slider(
    "Sensibilidad al cambio visual:",
    min_value=0.70,
    max_value=0.99,
    value=0.95,
    step=0.01,
    help="Umbral de similitud HSV. Menor valor significa que se necesitan más cambios visuales para activar un comentario."
)

cooldown_segundos = st.sidebar.slider(
    "Enfriamiento de voz (Cooldown en segundos):",
    min_value=3,
    max_value=15,
    value=8,
    step=1,
    help="Tiempo mínimo de espera para que el narrador vuelva a hablar después de iniciar un comentario."
)

confianza_minima = st.sidebar.slider(
    "Confianza mínima para comentar:",
    min_value=0.10,
    max_value=0.75,
    value=0.28,
    step=0.01,
    help="Evita narrar cuando CLIP no tiene una categoria suficientemente clara. Bajalo si tu video es muy oscuro o tiene muchos overlays."
)

usar_nlp_hf = st.sidebar.toggle(
    "Narración creativa con FLAN-T5",
    value=True,
    help="Usa un modelo NLP instruction de Hugging Face para redactar el comentario. Si falla la descarga o la generación, la app usa plantillas como respaldo."
)

etiquetas_para_ia = None

# Botón para limpiar caché / temporales
if st.sidebar.button("🧹 Limpiar temporales"):
    limpiar_audios_temporales()
    st.session_state.comentarios = []
    st.session_state.video_exportado = None
    st.sidebar.success("Caché limpiada correctamente.")

# --- DISPOSICIÓN PRINCIPAL DE LA APP ---
if uploaded_file is not None:
    # Si subió un nuevo archivo, guardar en temporal
    if st.session_state.video_name != uploaded_file.name:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.session_state.video_name = uploaded_file.name
        st.session_state.comentarios = []
        st.session_state.video_exportado = None
        
    video_info = utils.extraer_info_video(st.session_state.video_path)
    
    # Crear layout de dos columnas
    col_izq, col_der = st.columns([3, 2])
    
    with col_izq:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎥 Transmisión de Gameplay")
        
        # Mostrar el video original cargado
        st.video(st.session_state.video_path)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Botón para iniciar el procesamiento en tiempo real
        iniciar_btn = st.button("🎙️ Iniciar Transmisión y Narración", use_container_width=True, type="primary")
        
    with col_der:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Consola del Caster")
        
        # Métricas principales
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            estado_actual_placeholder = st.empty()
            estado_actual_placeholder.metric("Estado Detectado", "Apagado", delta=None)
        with m_col2:
            confianza_placeholder = st.empty()
            confianza_placeholder.metric("Confianza de IA", "0.0%", delta=None)
            
        st.markdown("---")
        
        # Monitor de heurísticas
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            similitud_placeholder = st.empty()
            similitud_placeholder.metric("Similitud de Frame", "1.000", delta=None)
        with h_col2:
            cooldown_placeholder = st.empty()
            cooldown_placeholder.metric("Estado de Cooldown", "Listo", delta=None)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Zona de ejecución / loop
    if iniciar_btn:
        st.session_state.comentarios = []
        st.session_state.video_exportado = None
        limpiar_audios_temporales()
        
        # Indicador de inicialización de modelos
        with st.spinner("Inicializando modelos de visión, lenguaje y voz..."):
            ia_engine.init_detector()
            if usar_nlp_hf:
                ia_engine.init_narrador()
            ia_engine.init_tts()

        estado_nlp = ia_engine.estado_narrador_nlp()
        if usar_nlp_hf and estado_nlp["error"]:
            st.warning("No se pudo cargar el modelo NLP; se usaran plantillas de respaldo para no detener el procesamiento.")
            
        # Abrir el video con OpenCV
        cap = cv2.VideoCapture(st.session_state.video_path)
        fps_original = video_info["fps"]
        total_frames = video_info["total_frames"]
        
        # Calcular intervalo de frames según los FPS de la IA deseados
        intervalo_frames = max(1, int(fps_original / fps_analisis))
        
        # Variables de control
        frame_anterior = None
        ultimo_comentario_tiempo = -999.0
        ultimo_estado = None
        historial_predicciones = deque(maxlen=3)
        
        # Contenedor dinámico y unificado para reproducir audio sin superposiciones
        audio_placeholder = st.empty()
        
        # Tiempo inicial real para el pacing de la simulación
        tiempo_inicio_real = time.time()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Marcadores visuales interactivos en col_izq
        with col_izq:
            st.markdown("### 👁️ Vista de la IA")
            vista_ia_placeholder = st.empty()
            
        frame_count = 0
        
        # Bucle de procesamiento
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % intervalo_frames == 0:
                # Calcular el tiempo de este frame en el video (segundos)
                timestamp_seg = frame_count / fps_original
                
                # Pacing: Sincronizar el procesamiento con el tiempo real del video si está activado
                if modo_tiempo_real:
                    tiempo_transcurrido_real = time.time() - tiempo_inicio_real
                    if tiempo_transcurrido_real < timestamp_seg:
                        time.sleep(timestamp_seg - tiempo_transcurrido_real)
                
                # Convertir de BGR a RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Mostrar en el panel de vista de la IA
                vista_ia_placeholder.image(frame_rgb, width='stretch')
                
                # Convertir a imagen PIL para los modelos de Hugging Face
                imagen_pil = Image.fromarray(frame_rgb)
                
                # 1. Ejecutar análisis de escena con CLIP y suavizar la etiqueta
                estado_crudo, confianza_cruda = ia_engine.analizar_cuadro(imagen_pil, juego_seleccionado, etiquetas_para_ia)
                historial_predicciones.append((estado_crudo, confianza_cruda))
                estado, confianza = utils.suavizar_predicciones(historial_predicciones, ultimo_estado)

                # 2. Calcular similitud con el frame anterior usando HSV
                similitud = 0.0
                if frame_anterior is not None:
                    similitud = utils.calcular_similitud_hsv(frame_anterior, frame_rgb)
                else:
                    # Primer frame
                    similitud = 0.0
                    
                # 3. Evaluar Heurísticas
                tiempo_desde_ultimo = timestamp_seg - ultimo_comentario_tiempo
                cooldown_activo = tiempo_desde_ultimo < cooldown_segundos
                cambio_estado = estado != ultimo_estado
                cambio_visual_suficiente = similitud < sensibilidad_cambio
                prediccion_confiable = confianza >= confianza_minima
                evento_importante = estado in {
                    "combate", "peligro", "victoria", "menú",
                    "talar", "minería", "minería_carbón", "minería_hierro",
                    "minería_cobre", "minería_oro", "minería_redstone",
                    "minería_lapislázuli", "minería_diamante", "minería_esmeralda",
                    "minería_escombros", "crafteo", "fundición", "cultivo",
                    "pesca", "ganadería", "cofre", "encantamiento", "pociones",
                    "comer", "dormir", "nadar", "bote", "portal", "nether",
                    "aldea", "comercio"
                }
                
                # Decidir si el narrador debe hablar
                debe_comentar = False
                razon = ""
                
                if not cooldown_activo and prediccion_confiable:
                    # Comentar solo cuando hay una señal semantica clara.
                    if frame_anterior is None:
                        debe_comentar = True
                        razon = "Inicio del video"
                    elif cambio_estado and (cambio_visual_suficiente or evento_importante):
                        debe_comentar = True
                        razon = "Cambio de estado"
                    elif cambio_visual_suficiente and evento_importante:
                        debe_comentar = True
                        razon = "Evento importante"
                
                # Actualizar métricas del panel derecho
                estado_actual_placeholder.metric("Estado Detectado", estado.title())
                confianza_placeholder.metric("Confianza de IA", f"{confianza*100:.1f}%")
                similitud_placeholder.metric("Similitud de Frame", f"{similitud:.3f}")
                
                if cooldown_activo:
                    tiempo_restante = max(0.0, cooldown_segundos - tiempo_desde_ultimo)
                    cooldown_placeholder.metric("Estado de Cooldown", f"Espera {tiempo_restante:.1f}s", delta="-Activo", delta_color="inverse")
                else:
                    cooldown_placeholder.metric("Estado de Cooldown", "Listo", delta="Disponible", delta_color="normal")

                # Si se decide comentar, ejecutamos el generador y TTS
                if debe_comentar:
                    comentario_texto = ia_engine.generar_comentario(estado, confianza, juego_seleccionado, usar_nlp=usar_nlp_hf)
                    
                    # Generar audio
                    audio_filename = f"comentario_{int(timestamp_seg)}s.wav"
                    audio_local_path = ia_engine.sintetizar_voz(comentario_texto, "temp_audio", audio_filename)
                    frame_local_path = guardar_frame_comentado(frame_rgb, timestamp_seg)
                    
                    # Guardar comentario en el historial
                    st.session_state.comentarios.append({
                        "timestamp": timestamp_seg,
                        "estado": estado,
                        "confianza": confianza,
                        "texto": comentario_texto,
                        "audio_path": audio_local_path,
                        "frame_path": frame_local_path,
                        "razon": razon
                    })
                    
                    # Actualizar tiempo de último comentario
                    ultimo_comentario_tiempo = timestamp_seg
                    
                    # Reproducción de audio en tiempo real inyectando HTML (solo en modo tiempo real)
                    if modo_tiempo_real:
                        reproducir_audio_automatico(audio_local_path, audio_placeholder)
                    
                # Guardar frame actual para comparar en el siguiente ciclo
                frame_anterior = frame_rgb.copy()
                ultimo_estado = estado
                
            # Actualizar barra de progreso del video
            progreso_porcentaje = min(1.0, frame_count / total_frames)
            progress_bar.progress(progreso_porcentaje)
            status_text.text(f"Procesando frame {frame_count} de {total_frames} ({progreso_porcentaje*100:.1f}%)")
            
            frame_count += 1
            
        cap.release()
        audio_placeholder.empty() # Limpiar el tag de audio para evitar reproducciones residuales
        progress_bar.progress(1.0)
        status_text.success("🎉 ¡Procesamiento y análisis de transmisión finalizado!")
        
    # --- GALERÍA DE FRAMES NARRADOS ---
    if len(st.session_state.comentarios) > 0:
        st.markdown("---")
        with st.expander(f"🖼️ Frames donde habló el narrador ({len(st.session_state.comentarios)})", expanded=False):
            columnas_galeria = st.columns(3)
            for idx, comentario in enumerate(st.session_state.comentarios):
                with columnas_galeria[idx % 3]:
                    frame_path = comentario.get("frame_path")
                    if frame_path and os.path.exists(frame_path):
                        st.image(frame_path, width='stretch')
                    else:
                        st.info("Frame no disponible")
                    st.caption(f"Minuto {formatear_tiempo(comentario['timestamp'])}")
                    st.markdown(f"**{comentario['texto']}**")

    # --- SECCIÓN DE EXPORTACIÓN DE VIDEO ---
    if len(st.session_state.comentarios) > 0:
        st.markdown("---")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎬 Exportar e Integrar Narración al Video Original")
        st.write("Esta opción combinará todas las voces generadas por la IA con el video original en sus marcas de tiempo correctas. El audio del juego se mantendrá de fondo con volumen reducido.")
        
        col_exp1, col_exp2 = st.columns([1, 2])
        
        with col_exp1:
            vol_juego = st.slider("Volumen de fondo del gameplay:", 0.0, 1.0, 0.2, 0.05)
            generar_btn = st.button("🚀 Renderizar Video Final", use_container_width=True)
            
        with col_exp2:
            if generar_btn:
                # Nombre del video de salida
                output_name = f"narrado_{st.session_state.video_name}"
                output_path = os.path.join("temp_audio", output_name)
                
                # Mapear los comentarios al formato que espera la función de mezcla
                audios_info = [(c["timestamp"], c["audio_path"]) for c in st.session_state.comentarios]
                
                with st.spinner("Ensamblando pistas de audio y renderizando video final con MoviePy..."):
                    try:
                        utils.mezclar_narracion_video(
                            video_path=st.session_state.video_path,
                            audios_info=audios_info,
                            output_path=output_path,
                            volumen_juego=vol_juego
                        )
                        st.session_state.video_exportado = output_path
                        st.success("¡Video renderizado con éxito!")
                    except Exception as e:
                        st.error(f"Error al renderizar el video: {str(e)}")
                        
            if st.session_state.video_exportado is not None and os.path.exists(st.session_state.video_exportado):
                st.write("#### 📥 ¡Tu video narrado está listo!")
                with open(st.session_state.video_exportado, "rb") as file:
                    st.download_button(
                        label="⬇️ Descargar Video Narrado",
                        data=file,
                        file_name=f"narrado_{st.session_state.video_name}",
                        mime="video/mp4",
                        use_container_width=True
                    )
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("👋 Sube un archivo de video en la barra lateral izquierda para iniciar la transmisión del Comentarista IA.")
