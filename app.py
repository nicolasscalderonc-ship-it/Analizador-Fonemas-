import io
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Fonemas - MATLAB a Python",
    page_icon="🎙️",
    layout="wide",
)

# --- CREACIÓN DE PESTAÑAS PRINCIPALES ---
tab1, tab2, tab3 = st.tabs(
    ["🎙️ Analizador en Vivo", "📚 Marco Teórico", "👥 Créditos del Proyecto"]
)

# ==========================================
# PESTAÑA 1: ANALIZADOR EN VIVO (FUNCIONALIDAD)
# ==========================================
with tab1:
  st.title("🎙️ Analizador de Señales de Voz y FFT")
  st.markdown(
      "Interfaz desarrollada para el análisis en el **Dominio del Tiempo** y"
      " la **Transformada de Fourier (FFT)** de vocales y consonantes."
  )
  st.markdown("---")

  # Panel de control lateral
  st.sidebar.header("🎛️ Panel de Control")
  fonema = st.sidebar.selectbox(
      "Selecciona el fonema a analizar:", ("A", "E", "I", "O", "U", "F", "S")
  )
  metodo_entrada = st.sidebar.radio(
      "Método de captura:", ("Grabar con el micrófono", "Subir archivo WAV")
  )

  st.sidebar.markdown("---")
  st.sidebar.info(
      "💡 **Nota:** Se procesará y centrará automáticamente una ventana de"
      " **2 segundos** de audio."
  )

  senal = None
  fs = None

  # Captura de audio
  if metodo_entrada == "Grabar con el micrófono":
    st.subheader(
        f"1. Grabación: : **'{fonema}'** (Ventana objetivo: 2s)"
    )
    audio_bytes = st.audio_input(
        "Haz clic en el micrófono y pronuncia el fonema:"
    )

    if audio_bytes is not None:
      st.audio(audio_bytes, format="audio/wav")
      fs, senal = wavfile.read(io.BytesIO(audio_bytes.read()))

  else:
    st.subheader(f"1. Cargar Archivo para el Fonema: **'{fonema}'**")
    uploaded_file = st.file_uploader(
        "Sube tu archivo de audio en formato .wav", type=["wav"]
    )
    if uploaded_file is not None:
      fs, senal = wavfile.read(uploaded_file)
      st.audio(uploaded_file, format="audio/wav")

  # Procesamiento matemático
  if senal is not None:
    st.markdown("---")
    st.subheader("2. Resultados del Análisis Espectral")

    # Convertir a mono si es estéreo y asegurar tipo float
    if len(senal.shape) > 1:
      senal = senal[:, 0]
    senal = senal.astype(float)

    # --- CONFIGURACIÓN DE LA VENTANA DE 2 SEGUNDOS CENTRADA ---
    duracion_objetivo = 2.0  # 2 segundos exactos
    muestras_objetivo = int(fs * duracion_objetivo)

    if len(senal) > muestras_objetivo:
      # Recortar tomando exactamente el fragmento central
      inicio = (len(senal) - muestras_objetivo) // 2
      senal = senal[inicio : inicio + muestras_objetivo]
    elif len(senal) < muestras_objetivo:
      # Si es menor a 2 segundos, rellenamos con ceros al final para estandarizar
      relleno = np.zeros(muestras_objetivo - len(senal))
      senal = np.concatenate((senal, relleno))

    N = len(senal)

    if N > 0:
      # Cálculo de la Transformada de Fourier (FFT)
      Y = np.fft.fft(senal)
      P2 = np.abs(Y / N)
      P1 = P2[: N // 2 + 1]
      P1[1:-1] = 2 * P1[1:-1]
      f = fs * np.arange(N // 2 + 1) / N
      t = np.arange(N) / fs

      # Distribución en dos columnas para las gráficas
      col1, col2 = st.columns(2)

      with col1:
        st.markdown("### Dominio del Tiempo (2s)")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(t, senal, color="#2b5c8f", linewidth=1.2)
        ax1.set_xlabel("Tiempo (s)", fontsize=10)
        ax1.set_ylabel("Amplitud", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.set_title(f'Señal Temporal - "{fonema}"', fontsize=11)
        ax1.set_xlim(0, duracion_objetivo)
        st.pyplot(fig1)

      with col2:
        st.markdown("### Dominio de la Frecuencia (FFT)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(f, P1, color="#d95f02", linewidth=1.2)
        ax2.set_xlabel("Frecuencia (Hz)", fontsize=10)
        ax2.set_ylabel("|X(f)|", fontsize=10)
        ax2.set_xlim(0, 5000)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.set_title(f'Espectro de Fourier - "{fonema}"', fontsize=11)
        st.pyplot(fig2)

    else:
      st.warning("El audio capturado está vacío. Intenta grabar de nuevo.")

# ==========================================
# PESTAÑA 2: MARCO TEÓRICO
# ==========================================
with tab2:
  st.title("📚 Fundamento Teórico y Metodología")
  st.markdown(
      "Esta sección describe los principios de procesamiento digital de"
      " señales aplicados en la aplicación."
  )
  st.markdown("---")

  st.subheader("1. Ventana Temporal de 2 Segundos")
  st.markdown(
      "Para estandarizar el análisis de los fonemas, la señal de voz capturada"
      " se procesa automáticamente extrayendo la **porción central exacta de 2"
      " segundos**, garantizando que el estado estacionario de la vocal o la"
      " fricción de la consonante se evalúen bajo las mismas condiciones."
  )

  st.subheader("2. Transformada Rápida de Fourier (FFT)")
  st.markdown(
      "La FFT permite convertir la señal del dominio del tiempo $x(t)$ al"
      " dominio de la frecuencia $X(f)$ mediante la siguiente relación"
      " discreta:"
  )
  st.latex(r"X(k) = \sum_{n=0}^{N-1} x(n) \cdot e^{-j 2 \pi k n / N}")
  st.markdown(
      "Esto facilita la identificación visual de los **formantes** característicos"
      " en las vocales (A, E, I, O, U) y la distribución espectral de ruido en"
      " las consonantes fricativas (F, S)."
  )

# ==========================================
# PESTAÑA 3: CRÉDITOS
# ==========================================
with tab3:
  st.title("👥 Información del Equipo de Trabajo")
  st.markdown(
      "Aplicación desarrollada como solución académica para el análisis"
      " avanzado de señales acústicas."
  )
  st.markdown("---")

  st.success(
      "**Integrantes del Proyecto:**\n\n"
      "- **Danna Valentina Aguilar Caro** — Código: `133318`\n"
      "- **Nicolas Steven Calderon Cano** — Código: `126153`"
  )
