#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Preprocesamiento de audio: VAD desde cero + extraccion de MFCC.
# Uso: python preprocess.py <ruta_audio.wav>

import numpy as np
import os
import sys
from types import ModuleType

# Simular dependencias opcionales de librosa no disponibles en ARM 32-bit
try:
    import numba
except ImportError:
    def _jit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f
    _numba = ModuleType('numba')
    _numba.jit = _jit
    _numba.njit = _jit
    _numba.stencil = _jit
    _numba.guvectorize = _jit
    _numba.vectorize = _jit
    _numba.prange = range
    sys.modules['numba'] = _numba
    sys.modules['numba.core'] = ModuleType('numba.core')
    sys.modules['numba.core.types'] = ModuleType('numba.core.types')

try:
    import soxr
except (ImportError, OSError):
    sys.modules['soxr'] = ModuleType('soxr')

import librosa

# Parametros de audio
TASA_MUESTREO    = 16000
DURACION_OBJETIVO = 1.0
MUESTRAS_OBJETIVO = int(TASA_MUESTREO * DURACION_OBJETIVO)  # 16000 muestras

# Parametros MFCC
# n_fft=512 -> ventana de 32ms, suficiente para capturar fonemas del espanol
# hop_length=160 -> paso de 10ms, produce ~97 marcos por segundo de audio
N_FFT      = 512
HOP_LENGTH = 160
N_MFCC     = 40   # minimo exigido: 13; usamos 40 para mayor detalle espectral
N_MELS     = 128

# Parametros VAD
# El habla tiene energia RMS alta y ZCR bajo; el ruido blanco tiene ZCR alto
UMBRAL_ENERGIA    = 0.008
UMBRAL_ZCR        = 0.30
LONGITUD_MARCO_VAD = 512


def cargar_audio(ruta_archivo):
    audio, _ = librosa.load(ruta_archivo, sr=TASA_MUESTREO, mono=True)
    return audio.astype(np.float32)


def _energia_rms(marco):
    return float(np.sqrt(np.mean(marco ** 2)))


def _tasa_cruces_por_cero(marco):
    cruces = np.nonzero(np.diff(np.sign(marco)))[0]
    return len(cruces) / len(marco)


def detectar_voz_activa(audio,
                         longitud_marco=LONGITUD_MARCO_VAD,
                         umbral_energia=UMBRAL_ENERGIA,
                         umbral_zcr=UMBRAL_ZCR):
    # Marca como voz activa los marcos con energia suficiente y ZCR bajo
    mascara = np.zeros(len(audio), dtype=bool)
    paso = longitud_marco // 2

    for inicio in range(0, len(audio) - longitud_marco, paso):
        marco   = audio[inicio: inicio + longitud_marco]
        energia = _energia_rms(marco)
        zcr     = _tasa_cruces_por_cero(marco)
        if energia > umbral_energia and zcr < umbral_zcr:
            mascara[inicio: inicio + longitud_marco] = True

    return mascara


def segmentar_audio(audio):
    # Recorta el tramo con voz y ajusta la longitud a MUESTRAS_OBJETIVO
    mascara     = detectar_voz_activa(audio)
    indices_voz = np.where(mascara)[0]

    if len(indices_voz) == 0:
        segmento = audio
    else:
        margen = int(0.05 * TASA_MUESTREO)   # 50ms de margen en cada extremo
        inicio  = max(0, indices_voz[0] - margen)
        fin     = min(len(audio), indices_voz[-1] + 1 + margen)
        segmento = audio[inicio:fin]

    if len(segmento) < MUESTRAS_OBJETIVO:
        segmento = np.pad(segmento, (0, MUESTRAS_OBJETIVO - len(segmento)))
    else:
        segmento = segmento[:MUESTRAS_OBJETIVO]

    return segmento


def normalizar_amplitud(audio):
    maximo = np.max(np.abs(audio))
    if maximo > 1e-8:
        return audio / maximo
    return audio


def extraer_mfcc(audio, tasa=TASA_MUESTREO):
    # Extrae N_MFCC coeficientes con normalizacion cepstral (CMVN)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=tasa,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        window="hann"
    )
    media = np.mean(mfcc, axis=1, keepdims=True)
    std   = np.std(mfcc, axis=1, keepdims=True) + 1e-8
    mfcc  = (mfcc - media) / std
    return mfcc.T.astype(np.float32)   # shape: (num_marcos, N_MFCC)


def preprocesar_muestra(ruta_archivo):
    audio = cargar_audio(ruta_archivo)
    audio = segmentar_audio(audio)
    audio = normalizar_amplitud(audio)
    return extraer_mfcc(audio)


def cargar_dataset(directorio_dataset, clases):
    # Carga y preprocesa todos los WAV del dataset; retorna X e y
    X, y = [], []

    for idx_clase, clase in enumerate(clases):
        ruta_clase = os.path.join(directorio_dataset, clase)
        if not os.path.exists(ruta_clase):
            print(f"  Directorio no encontrado: {ruta_clase}")
            continue

        archivos = sorted([f for f in os.listdir(ruta_clase) if f.endswith(".wav")])
        print(f"  '{clase}': {len(archivos)} muestras...", end="", flush=True)

        errores = 0
        for archivo in archivos:
            ruta = os.path.join(ruta_clase, archivo)
            try:
                mfcc = preprocesar_muestra(ruta)
                X.append(mfcc)
                y.append(idx_clase)
            except Exception:
                errores += 1

        estado = f" {errores} errores" if errores else " OK"
        print(estado)

    if len(X) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


if __name__ == "__main__":
    import sys

    print("Modulo de preprocesamiento - prueba")
    print("-" * 40)

    if len(sys.argv) > 1:
        ruta = sys.argv[1]
        print(f"Archivo : {ruta}")
        mfcc = preprocesar_muestra(ruta)
        print(f"Shape MFCC : {mfcc.shape}")
        print(f"Min / Max  : {mfcc.min():.3f} / {mfcc.max():.3f}")
    else:
        print("Uso: python preprocess.py <ruta_audio.wav>")
        print("Ejecutando con audio sintetico...")
        audio_sintetico = (np.random.randn(MUESTRAS_OBJETIVO) * 0.3).astype(np.float32)
        segmento = segmentar_audio(audio_sintetico)
        segmento = normalizar_amplitud(segmento)
        mfcc     = extraer_mfcc(segmento)
        marcos_esperados = 1 + (MUESTRAS_OBJETIVO - N_FFT) // HOP_LENGTH
        print(f"Shape obtenido  : {mfcc.shape}")
        print(f"Shape esperado  : ({marcos_esperados}, {N_MFCC})")
        print(f"Coincide        : {mfcc.shape == (marcos_esperados, N_MFCC)}")
    print("-" * 40)
