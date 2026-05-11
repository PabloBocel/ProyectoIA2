#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Entrenamiento de la CNN 1D para clasificacion de comandos de voz.
# Uso: python train.py

import numpy as np
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from preprocess import cargar_dataset, N_MFCC
from augment import aumentar_dataset

CLASES   = ["AVANZA", "RETROCEDE", "IZQUIERDA", "DERECHA", "DETENTE", "RUIDO_FONDO"]
N_CLASES = len(CLASES)

DIR_DATASET = "dataset"
DIR_MODELOS = "models"

# Hiperparametros
EPOCAS              = 50
TAMANO_LOTE         = 32
TASA_APRENDIZAJE    = 0.001
PROPORCION_VALIDACION = 0.15
PROPORCION_PRUEBA   = 0.15
SEMILLA             = 42
NOMBRE_MODELO       = "modelo_cnn_comandos"
FACTOR_AUMENTO      = 3   # copias aumentadas por muestra original (0 = sin aumento)


def construir_modelo_cnn(forma_entrada, n_clases):
    # CNN 1D sobre MFCC: 3 bloques Conv+BN+Pool+Dropout, luego clasificador denso
    modelo = keras.Sequential(name="CNN_ComandosVoz")
    modelo.add(layers.Input(shape=forma_entrada))

    modelo.add(layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"))
    modelo.add(layers.BatchNormalization())
    modelo.add(layers.MaxPooling1D(pool_size=2))
    modelo.add(layers.Dropout(0.25))

    modelo.add(layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"))
    modelo.add(layers.BatchNormalization())
    modelo.add(layers.MaxPooling1D(pool_size=2))
    modelo.add(layers.Dropout(0.25))

    modelo.add(layers.Conv1D(256, kernel_size=3, activation="relu", padding="same"))
    modelo.add(layers.BatchNormalization())
    modelo.add(layers.GlobalAveragePooling1D())
    modelo.add(layers.Dropout(0.40))

    modelo.add(layers.Dense(128, activation="relu"))
    modelo.add(layers.BatchNormalization())
    modelo.add(layers.Dropout(0.30))
    modelo.add(layers.Dense(n_clases, activation="softmax"))

    return modelo


def graficar_historial(historial, directorio):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(historial.history["accuracy"],     label="Entrenamiento")
    ax1.plot(historial.history["val_accuracy"], label="Validacion")
    ax1.set_title("Precision - CNN Comandos de Voz")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(historial.history["loss"],     label="Entrenamiento")
    ax2.plot(historial.history["val_loss"], label="Validacion")
    ax2.set_title("Perdida - CNN Comandos de Voz")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Categorical Crossentropy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    ruta = os.path.join(directorio, "historial_cnn.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafica guardada: {ruta}")


def exportar_tflite(modelo, directorio, nombre):
    # Exporta a TFLite con cuantizacion dinamica para optimizar en Raspberry Pi
    convertidor = tf.lite.TFLiteConverter.from_keras_model(modelo)
    convertidor.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = convertidor.convert()

    ruta = os.path.join(directorio, f"{nombre}.tflite")
    with open(ruta, "wb") as f:
        f.write(tflite_bytes)

    kb = os.path.getsize(ruta) / 1024
    print(f"  Modelo TFLite: {ruta}  ({kb:.1f} KB)")


def main():
    print("\n" + "=" * 55)
    print("  ENTRENAMIENTO CNN 1D - COMANDOS DE VOZ")
    print("=" * 55)

    os.makedirs(DIR_MODELOS, exist_ok=True)

    print(f"\n[1/5] Aplicando Data Augmentation (factor {FACTOR_AUMENTO}x)...")
    if FACTOR_AUMENTO > 0:
        aumentar_dataset(DIR_DATASET, CLASES, factor_aumento=FACTOR_AUMENTO)

    print("\n[2/5] Cargando y preprocesando dataset...")
    X, y = cargar_dataset(DIR_DATASET, CLASES)

    if len(X) == 0:
        print("\nERROR: No se encontraron datos.")
        print("Ejecuta record_audio.py primero para grabar el corpus.")
        return

    print(f"\n  Muestras totales : {len(X)}")
    print(f"  Shape/muestra    : {X[0].shape}")
    print("  Distribucion:")
    for i, clase in enumerate(CLASES):
        n = int(np.sum(y == i))
        print(f"    {clase:<14}: {n:>5}")

    print("\n[3/5] Dividiendo dataset...")
    X_tv, X_prueba, y_tv, y_prueba = train_test_split(
        X, y, test_size=PROPORCION_PRUEBA,
        stratify=y, random_state=SEMILLA
    )
    prop_val = PROPORCION_VALIDACION / (1.0 - PROPORCION_PRUEBA)
    X_entreno, X_val, y_entreno, y_val = train_test_split(
        X_tv, y_tv, test_size=prop_val,
        stratify=y_tv, random_state=SEMILLA
    )
    print(f"  Entrenamiento : {len(X_entreno)}")
    print(f"  Validacion    : {len(X_val)}")
    print(f"  Prueba        : {len(X_prueba)}")

    np.save(os.path.join(DIR_MODELOS, "X_prueba.npy"), X_prueba)
    np.save(os.path.join(DIR_MODELOS, "y_prueba.npy"), y_prueba)

    print("\n[4/5] Construyendo modelo CNN 1D...")
    forma  = X_entreno.shape[1:]
    modelo = construir_modelo_cnn(forma, N_CLASES)
    modelo.summary()

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=TASA_APRENDIZAJE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\n[5/5] Entrenando {EPOCAS} epocas (EarlyStopping activo)...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(DIR_MODELOS, f"{NOMBRE_MODELO}_mejor.h5"),
            monitor="val_accuracy", save_best_only=True, verbose=0
        ),
    ]

    historial = modelo.fit(
        X_entreno, y_entreno,
        validation_data=(X_val, y_val),
        epochs=EPOCAS,
        batch_size=TAMANO_LOTE,
        callbacks=callbacks,
        verbose=1
    )

    print("\n  Exportando modelos...")
    ruta_h5 = os.path.join(DIR_MODELOS, f"{NOMBRE_MODELO}.h5")
    modelo.save(ruta_h5)
    print(f"  Modelo H5: {ruta_h5}")
    exportar_tflite(modelo, DIR_MODELOS, NOMBRE_MODELO)

    config = {
        "clases"        : CLASES,
        "n_mfcc"        : N_MFCC,
        "tasa_muestreo" : 16000,
        "forma_entrada" : list(forma),
        "epocas_reales" : len(historial.history["loss"]),
        "val_accuracy"  : float(max(historial.history["val_accuracy"])),
    }
    with open(os.path.join(DIR_MODELOS, "config_cnn.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Config guardada: {DIR_MODELOS}/config_cnn.json")

    graficar_historial(historial, DIR_MODELOS)

    loss_p, acc_p = modelo.evaluate(X_prueba, y_prueba, verbose=0)
    print(f"\n  Accuracy en prueba : {acc_p*100:.2f}%")
    print(f"  Loss en prueba     : {loss_p:.4f}")
    print("\n  Ejecuta evaluate.py para metricas detalladas.")
    print("Entrenamiento completado.")


if __name__ == "__main__":
    main()
