#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Script de grabacion del corpus de audio.
# Guia al usuario para grabar 200 muestras por clase con cuenta regresiva.
# Uso: python record_audio.py

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav_io
import os
import time
import sys

# Parametros de grabacion
TASA_MUESTREO       = 16000
DURACION_MUESTRA    = 1.0
MUESTRAS_POR_COMANDO = 200
CANALES             = 1

CLASES = [
    "AVANZA",
    "RETROCEDE",
    "IZQUIERDA",
    "DERECHA",
    "DETENTE",
    "RUIDO_FONDO",
]

DIR_DATASET = "dataset"


def barra_progreso(actual, total, ancho=45):
    progreso   = int(ancho * actual / total)
    barra      = "#" * progreso + "-" * (ancho - progreso)
    porcentaje = 100.0 * actual / total
    return f"[{barra}] {actual:>3}/{total} ({porcentaje:5.1f}%)"


def crear_estructura_directorios():
    for clase in CLASES:
        os.makedirs(os.path.join(DIR_DATASET, clase), exist_ok=True)
    print(f"Carpetas listas en '{DIR_DATASET}/'")


def contar_muestras(ruta_carpeta):
    if not os.path.exists(ruta_carpeta):
        return 0
    return len([f for f in os.listdir(ruta_carpeta) if f.endswith(".wav")])


def grabar_muestra(duracion=DURACION_MUESTRA, tasa=TASA_MUESTREO):
    muestras = int(duracion * tasa)
    audio    = sd.rec(muestras, samplerate=tasa, channels=CANALES, dtype="float32")
    sd.wait()
    return audio.flatten()


def guardar_wav(audio, ruta, tasa=TASA_MUESTREO):
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    wav_io.write(ruta, tasa, audio_int16)


def grabar_clase(clase, ruta_clase, objetivo, duracion=DURACION_MUESTRA):
    ya_grabadas = contar_muestras(ruta_clase)

    if ya_grabadas >= objetivo:
        print(f"\n  '{clase}' ya tiene {ya_grabadas} muestras. Saltando.")
        return

    print(f"\n{'='*55}")
    print(f"  CLASE: {clase}")
    print(f"  Grabadas: {ya_grabadas}/{objetivo}  -  Faltan: {objetivo - ya_grabadas}")
    print("-" * 55)

    if clase == "RUIDO_FONDO":
        print("  Graba silencio, ruido ambiente y habla aleatoria.")
        print("  Varia el ruido: ventilador, teclado, conversaciones.")
    else:
        print(f"  Di claramente '{clase}' cuando veas [GRABA].")
        print("  Varia tono, volumen y velocidad entre grabaciones.")

    input("\n  Presiona ENTER para comenzar...")

    for i in range(ya_grabadas, objetivo):
        numero        = i + 1
        nombre        = f"{clase}_{numero:04d}.wav"
        ruta_archivo  = os.path.join(ruta_clase, nombre)

        print(f"\n  {barra_progreso(i, objetivo)}")

        for cuenta in range(3, 0, -1):
            print(f"\r  Muestra {numero:>3}/{objetivo}  Preparate... {cuenta}", end="", flush=True)
            time.sleep(0.6)

        print(f"\r  Muestra {numero:>3}/{objetivo}  [GRABA]              ", end="", flush=True)

        audio     = grabar_muestra(duracion=duracion)
        guardar_wav(audio, ruta_archivo)

        nivel_rms = float(np.sqrt(np.mean(audio ** 2)))
        barras    = min(20, int(nivel_rms * 200))
        nivel_str = "#" * barras + "-" * (20 - barras)
        print(f"\r  Muestra {numero:>3}/{objetivo}  Guardada  Nivel: [{nivel_str}]  ", flush=True)

        time.sleep(0.25)

    print(f"\n  '{clase}' completada: {objetivo} muestras en '{ruta_clase}'")


def mostrar_resumen():
    print(f"\n{'='*55}")
    print("  RESUMEN DEL CORPUS")
    print("-" * 55)
    total = 0
    for clase in CLASES:
        ruta  = os.path.join(DIR_DATASET, clase)
        n     = contar_muestras(ruta)
        total += n
        estado = "OK" if n >= MUESTRAS_POR_COMANDO else f"faltan {MUESTRAS_POR_COMANDO - n}"
        print(f"  {clase:<16}: {n:>4}  [{estado}]")
    print("-" * 55)
    print(f"  Total: {total} muestras")
    print("=" * 55)


def main():
    print(f"\n{'='*55}")
    print("  GRABACION DE CORPUS DE VOZ")
    print("  Asistente Robotico - Universidad Rafael Landivar")
    print("-" * 55)
    print(f"  Tasa de muestreo : {TASA_MUESTREO} Hz")
    print(f"  Duracion/muestra : {DURACION_MUESTRA} s")
    print(f"  Meta por clase   : {MUESTRAS_POR_COMANDO} muestras")
    print("=" * 55)

    crear_estructura_directorios()

    print("\n  Opciones:")
    print("  1. Grabar todas las clases")
    print("  2. Grabar una clase especifica")
    print("  3. Ver resumen del corpus")
    opcion = input("\n  Opcion: ").strip()

    if opcion == "1":
        for clase in CLASES:
            ruta = os.path.join(DIR_DATASET, clase)
            grabar_clase(clase, ruta, MUESTRAS_POR_COMANDO)

    elif opcion == "2":
        print("\n  Clases disponibles:")
        for idx, clase in enumerate(CLASES):
            n = contar_muestras(os.path.join(DIR_DATASET, clase))
            print(f"    {idx+1}. {clase:<16} ({n}/{MUESTRAS_POR_COMANDO} grabadas)")

        try:
            sel = int(input("\n  Numero de clase: ")) - 1
            if 0 <= sel < len(CLASES):
                clase = CLASES[sel]
                ruta  = os.path.join(DIR_DATASET, clase)
                grabar_clase(clase, ruta, MUESTRAS_POR_COMANDO)
            else:
                print("  Opcion fuera de rango.")
        except ValueError:
            print("  Entrada invalida.")

    elif opcion == "3":
        mostrar_resumen()
        return

    else:
        print("  Opcion no reconocida.")
        return

    mostrar_resumen()


if __name__ == "__main__":
    main()
