#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Pipeline de inferencia en tiempo real: microfono -> VAD -> MFCC -> CNN -> Arduino.
# Uso: python inference.py [--puerto /dev/ttyUSB0]

import numpy as np
import sounddevice as sd
import time
import os
import json
import queue
import argparse

from preprocess import (TASA_MUESTREO, MUESTRAS_OBJETIVO,
                        detectar_voz_activa, normalizar_amplitud, extraer_mfcc,
                        segmentar_audio)
from arduino_comm import ComunicadorArduino

DIR_MODELOS       = "models"
RUTA_MODELO_TFLITE = os.path.join(DIR_MODELOS, "modelo_cnn_comandos.tflite")
RUTA_MODELO_H5    = os.path.join(DIR_MODELOS, "modelo_cnn_comandos.h5")
RUTA_CONFIG_CNN   = os.path.join(DIR_MODELOS, "config_cnn.json")

# 1024 muestras por bloque -> ~64ms por callback a 16kHz
TAMANO_BLOQUE = 1024

DURACION_VENTANA  = MUESTRAS_OBJETIVO / TASA_MUESTREO   # 1.0 segundo
UMBRAL_CONFIANZA  = 0.75
FRACCION_VOZ_MINIMA = 0.30
COOLDOWN_SEGUNDOS = 1.0

CLASES_DEFAULT = ["AVANZA", "RETROCEDE", "IZQUIERDA", "DERECHA", "DETENTE", "RUIDO_FONDO"]


def _cargar_configuracion():
    if os.path.exists(RUTA_CONFIG_CNN):
        with open(RUTA_CONFIG_CNN, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"clases": CLASES_DEFAULT}


def _cargar_modelo():
    # Prefiere TFLite (menos RAM, ideal para Raspberry Pi); cae a .h5 si no existe.
    # Retorna (backend, objeto) donde backend es "tflite", "keras" o None.
    import tensorflow as tf

    if os.path.exists(RUTA_MODELO_TFLITE):
        try:
            print(f"  Cargando TFLite: {RUTA_MODELO_TFLITE}", end="", flush=True)
            interprete = tf.lite.Interpreter(model_path=RUTA_MODELO_TFLITE)
            interprete.allocate_tensors()
            print("  OK")
            return "tflite", interprete
        except Exception as e:
            print(f"\n  Error TFLite: {e}. Intentando con .h5...")

    if os.path.exists(RUTA_MODELO_H5):
        try:
            print(f"  Cargando Keras: {RUTA_MODELO_H5}", end="", flush=True)
            modelo = tf.keras.models.load_model(RUTA_MODELO_H5)
            print(f"  OK ({modelo.count_params():,} parametros)")
            return "keras", modelo
        except Exception as e:
            print(f"\n  Error al cargar el modelo: {e}")

    print("\n  ADVERTENCIA: no se encontro ningun modelo en 'models/'")
    print("  Sistema en MODO DEMOSTRACION (predicciones aleatorias).\n")
    return None, None


class InferenciaVozRobot:

    def __init__(self, puerto_arduino=None):
        self._config   = _cargar_configuracion()
        self._clases   = self._config.get("clases", CLASES_DEFAULT)
        self._backend, self._modelo = _cargar_modelo()
        self._arduino  = ComunicadorArduino(puerto=puerto_arduino)

        self._buffer           = np.zeros(MUESTRAS_OBJETIVO, dtype=np.float32)
        self._cola_audio       = queue.Queue(maxsize=50)
        self._n_inferencias    = 0
        self._latencias_ms     = []
        self._tiempo_ultimo_cmd = 0.0
        self._activo           = False

    def _callback_microfono(self, entrada, num_muestras, info_tiempo, estado):
        # Solo encola el bloque; el procesamiento ocurre en el hilo principal
        bloque = entrada[:, 0].copy().astype(np.float32)
        try:
            self._cola_audio.put_nowait(bloque)
        except queue.Full:
            pass

    def _actualizar_buffer(self, nuevo_bloque):
        n = len(nuevo_bloque)
        self._buffer[:-n] = self._buffer[n:]
        self._buffer[-n:] = nuevo_bloque

    def _preprocesar(self):
        # Retorna None si no hay suficiente voz en el buffer
        audio      = self._buffer.copy()
        mascara_voz = detectar_voz_activa(audio)
        if np.mean(mascara_voz) < FRACCION_VOZ_MINIMA:
            return None
        audio = normalizar_amplitud(audio)
        mfcc  = extraer_mfcc(audio)
        return np.expand_dims(mfcc, axis=0).astype(np.float32)

    def _predecir(self, entrada):
        # En modo demostracion retorna clase aleatoria para probar el pipeline
        if self._modelo is None:
            import random
            idx  = random.randint(0, len(self._clases) - 1)
            conf = random.uniform(0.60, 0.98)
            return self._clases[idx], conf

        if self._backend == "tflite":
            det_entrada  = self._modelo.get_input_details()
            det_salida   = self._modelo.get_output_details()
            self._modelo.set_tensor(det_entrada[0]["index"], entrada)
            self._modelo.invoke()
            probs = self._modelo.get_tensor(det_salida[0]["index"])[0]
        else:
            probs = self._modelo.predict(entrada, verbose=0)[0]

        idx = int(np.argmax(probs))
        return self._clases[idx], float(probs[idx])

    def _actuar(self, clase, confianza):
        ahora = time.time()
        if confianza < UMBRAL_CONFIANZA:
            return
        if ahora - self._tiempo_ultimo_cmd < COOLDOWN_SEGUNDOS:
            return
        if clase == "RUIDO_FONDO":
            return
        print(f"\n  COMANDO: {clase:<20}  confianza: {confianza:.1%}")
        self._arduino.enviar_comando(clase)
        self._tiempo_ultimo_cmd = ahora

    def ejecutar(self):
        print("\n" + "=" * 55)
        print("  INFERENCIA EN TIEMPO REAL - ASISTENTE ROBOTICO")
        print("=" * 55)
        print(f"  Clases       : {', '.join(self._clases)}")
        print(f"  Umbral conf. : {UMBRAL_CONFIANZA:.0%}")
        print(f"  Cooldown     : {COOLDOWN_SEGUNDOS} s")
        backend_str = {"tflite": "TFLite (optimizado)", "keras": "Keras .h5"}.get(self._backend, "DEMOSTRACION")
        print(f"  Modelo       : {backend_str}")
        print(f"  Arduino      : {'Conectado' if self._arduino.conectado else 'Simulacion'}")
        print("\n  Habla un comando. Ctrl+C para salir.")
        print("-" * 55)

        self._activo = True

        with sd.InputStream(
            samplerate=TASA_MUESTREO,
            channels=1,
            blocksize=TAMANO_BLOQUE,
            dtype="float32",
            callback=self._callback_microfono,
        ):
            try:
                while self._activo:
                    try:
                        bloque = self._cola_audio.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    t0 = time.perf_counter()
                    self._actualizar_buffer(bloque)

                    entrada = self._preprocesar()
                    if entrada is None:
                        print(".", end="", flush=True)
                        continue

                    clase, confianza = self._predecir(entrada)
                    self._n_inferencias += 1

                    latencia_ms = (time.perf_counter() - t0) * 1000
                    self._latencias_ms.append(latencia_ms)

                    estado = "+" if confianza >= UMBRAL_CONFIANZA else "."
                    print(f"\r  [{self._n_inferencias:05d}] {clase:<14} "
                          f"{confianza:.1%} {estado}  lat={latencia_ms:5.1f}ms   ",
                          end="", flush=True)

                    self._actuar(clase, confianza)

            except KeyboardInterrupt:
                print("\n\n  Interrupcion recibida. Cerrando...")
            finally:
                self._activo = False
                self._arduino.cerrar()
                self._mostrar_estadisticas()

    def _mostrar_estadisticas(self):
        if not self._latencias_ms:
            return
        lats  = np.array(self._latencias_ms)
        cumple = np.mean(lats) < 500.0
        print(f"\n{'='*55}")
        print("  ESTADISTICAS DE LA SESION")
        print("-" * 55)
        print(f"  Inferencias totales : {self._n_inferencias}")
        print(f"  Latencia promedio   : {np.mean(lats):.1f} ms")
        print(f"  Latencia mediana    : {np.median(lats):.1f} ms")
        print(f"  Latencia maxima     : {np.max(lats):.1f} ms")
        print(f"  Objetivo < 500 ms   : {'CUMPLIDO' if cumple else 'SUPERA OBJETIVO'}")
        print("=" * 55)


def main():
    analizador = argparse.ArgumentParser(
        description="Inferencia en tiempo real de comandos de voz para robot movil"
    )
    analizador.add_argument(
        "--puerto", type=str, default=None,
        help="Puerto serial del Arduino (ej. /dev/ttyUSB0 o COM3)"
    )
    args    = analizador.parse_args()
    sistema = InferenciaVozRobot(puerto_arduino=args.puerto)
    sistema.ejecutar()


if __name__ == "__main__":
    main()
