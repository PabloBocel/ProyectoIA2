#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Aumento de datos para el corpus de audio.
# Tecnicas: desplazamiento temporal, cambio de tono, ruido gaussiano, estiramiento.
# Uso: python augment.py

import numpy as np
import librosa
import scipy.io.wavfile as wav_io
import os
import random

TASA_MUESTREO    = 16000
SEMILLA_ALEATORIA = 42


def desplazamiento_temporal(audio, max_fraccion=0.20):
    # Mueve la senal +/- hasta el 20% de su duracion; el hueco se rellena con ceros
    desplazamiento = int(random.uniform(-max_fraccion, max_fraccion) * len(audio))

    if desplazamiento > 0:
        aumentado = np.concatenate([np.zeros(desplazamiento, dtype=np.float32),
                                    audio[:-desplazamiento]])
    elif desplazamiento < 0:
        d = abs(desplazamiento)
        aumentado = np.concatenate([audio[d:],
                                    np.zeros(d, dtype=np.float32)])
    else:
        aumentado = audio.copy()

    return aumentado.astype(np.float32)


def cambio_de_tono(audio, tasa=TASA_MUESTREO, semitonos_max=3.0):
    # Modifica el pitch +/- 3 semitonos sin cambiar la duracion
    semitonos = random.uniform(-semitonos_max, semitonos_max)
    aumentado = librosa.effects.pitch_shift(audio, sr=tasa, n_steps=semitonos)
    return aumentado.astype(np.float32)


def inyeccion_ruido_gaussiano(audio, factor_min=0.002, factor_max=0.010):
    # Agrega ruido blanco gaussiano; normaliza para mantenerse en [-1, 1]
    factor   = random.uniform(factor_min, factor_max)
    ruido    = np.random.normal(0.0, factor, size=len(audio)).astype(np.float32)
    aumentado = audio + ruido

    maximo = np.max(np.abs(aumentado))
    if maximo > 1e-8:
        aumentado = aumentado / maximo

    return aumentado.astype(np.float32)


def estiramiento_temporal(audio, factor_min=0.80, factor_max=1.20):
    # Cambia la velocidad del habla sin alterar el tono; ajusta la longitud al original
    factor   = random.uniform(factor_min, factor_max)
    estirado = librosa.effects.time_stretch(audio, rate=factor)

    longitud_orig = len(audio)
    if len(estirado) < longitud_orig:
        estirado = np.pad(estirado, (0, longitud_orig - len(estirado)))
    else:
        estirado = estirado[:longitud_orig]

    return estirado.astype(np.float32)


def aplicar_aumento_aleatorio(audio, probabilidad=0.8):
    # Aplica un subconjunto aleatorio de las 4 tecnicas; garantiza al menos una
    tecnicas = [
        desplazamiento_temporal,
        cambio_de_tono,
        inyeccion_ruido_gaussiano,
        estiramiento_temporal,
    ]

    aumentado = audio.copy()
    aplicadas = 0

    for tecnica in tecnicas:
        if random.random() < probabilidad:
            aumentado = tecnica(aumentado)
            aplicadas += 1

    if aplicadas == 0:
        aumentado = random.choice(tecnicas)(aumentado)

    return aumentado


def aumentar_dataset(directorio_dataset, clases, factor_aumento=3,
                     semilla=SEMILLA_ALEATORIA):
    # Genera factor_aumento copias aumentadas por cada WAV original del dataset
    random.seed(semilla)
    np.random.seed(semilla)

    print(f"\nAumentando dataset (factor {factor_aumento}x)...")

    for clase in clases:
        ruta_clase = os.path.join(directorio_dataset, clase)
        if not os.path.exists(ruta_clase):
            print(f"  Directorio no encontrado: {ruta_clase}")
            continue

        originales = sorted([
            f for f in os.listdir(ruta_clase)
            if f.endswith(".wav") and "_aug" not in f
        ])

        if not originales:
            print(f"  '{clase}': sin muestras originales.")
            continue

        nuevos = 0
        for archivo in originales:
            audio, _ = librosa.load(
                os.path.join(ruta_clase, archivo),
                sr=TASA_MUESTREO, mono=True
            )
            audio = audio.astype(np.float32)

            for i in range(factor_aumento):
                nombre_aug = f"{os.path.splitext(archivo)[0]}_aug{i+1:02d}.wav"
                ruta_aug   = os.path.join(ruta_clase, nombre_aug)

                if os.path.exists(ruta_aug):
                    continue

                audio_aug   = aplicar_aumento_aleatorio(audio)
                audio_int16 = np.clip(audio_aug * 32767, -32768, 32767).astype(np.int16)
                wav_io.write(ruta_aug, TASA_MUESTREO, audio_int16)
                nuevos += 1

        total_ahora = len([f for f in os.listdir(ruta_clase) if f.endswith(".wav")])
        print(f"  '{clase}': {len(originales)} originales -> {total_ahora} total (+{nuevos} nuevos)")

    print("Aumento de datos completado.")


if __name__ == "__main__":
    random.seed(SEMILLA_ALEATORIA)
    np.random.seed(SEMILLA_ALEATORIA)

    print("Modulo de aumento de datos - prueba")
    print("-" * 40)

    duracion     = 1.0
    t            = np.linspace(0, duracion, int(TASA_MUESTREO * duracion))
    audio_prueba = (np.sin(2 * np.pi * 200 * t) * 0.5 +
                    np.random.randn(len(t)) * 0.05).astype(np.float32)

    print(f"Audio original  -> longitud: {len(audio_prueba)}  RMS: {np.sqrt(np.mean(audio_prueba**2)):.4f}")

    ds = desplazamiento_temporal(audio_prueba)
    print(f"Desp. temporal  -> longitud: {len(ds)}  RMS: {np.sqrt(np.mean(ds**2)):.4f}")

    ct = cambio_de_tono(audio_prueba)
    print(f"Cambio de tono  -> longitud: {len(ct)}  RMS: {np.sqrt(np.mean(ct**2)):.4f}")

    rn = inyeccion_ruido_gaussiano(audio_prueba)
    print(f"Ruido gaussiano -> longitud: {len(rn)}  RMS: {np.sqrt(np.mean(rn**2)):.4f}")

    et = estiramiento_temporal(audio_prueba)
    print(f"Estir. temporal -> longitud: {len(et)}  RMS: {np.sqrt(np.mean(et**2)):.4f}")

    aleat = aplicar_aumento_aleatorio(audio_prueba)
    print(f"Combinado aleat -> longitud: {len(aleat)}  RMS: {np.sqrt(np.mean(aleat**2)):.4f}")

    print("-" * 40)
    print("Todas las tecnicas funcionaron correctamente.")
