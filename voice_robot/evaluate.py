#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Evaluacion del modelo CNN: metricas, matrices de confusion y latencia.
# Uso: python evaluate.py

import numpy as np
import os
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

DIR_MODELOS  = "models"
DIR_REPORTES = "reports"

CLASES_CNN = ["AVANZA", "RETROCEDE", "IZQUIERDA", "DERECHA", "DETENTE", "RUIDO_FONDO"]


def cargar_datos_prueba(sufijo=""):
    # Carga los arrays guardados por train.py; retorna (None, None) si no existen
    ruta_X = os.path.join(DIR_MODELOS, f"X_prueba{sufijo}.npy")
    ruta_y = os.path.join(DIR_MODELOS, f"y_prueba{sufijo}.npy")

    if not os.path.exists(ruta_X) or not os.path.exists(ruta_y):
        print(f"  Datos de prueba no encontrados ({ruta_X}).")
        print("  Ejecuta train.py primero.")
        return None, None

    X = np.load(ruta_X)
    y = np.load(ruta_y)
    print(f"  Datos de prueba: {len(X)} muestras, shape {X[0].shape}")
    return X, y


def cargar_modelo_keras(ruta):
    import tensorflow as tf
    try:
        modelo = tf.keras.models.load_model(ruta)
        print(f"  Modelo cargado: {ruta}  ({modelo.count_params():,} parametros)")
        return modelo
    except Exception as e:
        print(f"  No se pudo cargar el modelo: {e}")
        return None


def evaluar_modelo(modelo, X, y, clases, etiqueta="Modelo"):
    print(f"\n  Generando predicciones ({len(X)} muestras)...")
    t0    = time.perf_counter()
    probs = modelo.predict(X, verbose=0)
    t_total = time.perf_counter() - t0

    y_pred      = np.argmax(probs, axis=1)
    latencia_ms = (t_total / len(X)) * 1000

    clases_presentes = [clases[i] for i in sorted(np.unique(y))]

    acc       = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y, y_pred, average="macro", zero_division=0)
    f1_macro  = f1_score(y, y_pred, average="macro", zero_division=0)

    print(f"\n  RESULTADOS - {etiqueta}")
    print("-" * 55)
    print(f"  Accuracy   : {acc*100:6.2f} %")
    print(f"  Precision  : {precision*100:6.2f} %  (macro-avg)")
    print(f"  Recall     : {recall*100:6.2f} %  (macro-avg)")
    print(f"  F1-score   : {f1_macro*100:6.2f} %  (macro-avg)")
    print(f"  Latencia   : {latencia_ms:6.2f} ms / muestra")
    print("-" * 55)

    print("\n  Reporte por clase:")
    reporte = classification_report(y, y_pred, target_names=clases_presentes, zero_division=0)
    for linea in reporte.splitlines():
        print(f"    {linea}")

    metricas = {
        "etiqueta"           : etiqueta,
        "muestras"           : int(len(X)),
        "accuracy"           : float(acc),
        "precision_macro"    : float(precision),
        "recall_macro"       : float(recall),
        "f1_macro"           : float(f1_macro),
        "latencia_ms_muestra": float(latencia_ms),
    }
    return y_pred, metricas


def graficar_matriz_confusion(y_real, y_pred, clases, titulo, ruta):
    indices   = sorted(np.unique(y_real))
    etiquetas = [clases[i] for i in indices]

    cm = confusion_matrix(y_real, y_pred, labels=indices)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(cm.sum(axis=1, keepdims=True) > 0,
                           cm / cm.sum(axis=1, keepdims=True), 0)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    im1 = ax1.imshow(cm, interpolation="nearest", cmap="Blues")
    ax1.set_title(f"Matriz de Confusion - {titulo}\n(conteos absolutos)")
    ax1.set_xlabel("Prediccion")
    ax1.set_ylabel("Clase Real")
    plt.colorbar(im1, ax=ax1)
    ax1.set_xticks(range(len(etiquetas)))
    ax1.set_yticks(range(len(etiquetas)))
    ax1.set_xticklabels(etiquetas, rotation=40, ha="right", fontsize=9)
    ax1.set_yticklabels(etiquetas, fontsize=9)
    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            ax1.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=10, fontweight="bold")

    im2 = ax2.imshow(cm_norm, interpolation="nearest", cmap="Greens", vmin=0, vmax=1)
    ax2.set_title(f"Matriz de Confusion - {titulo}\n(normalizada - recall por clase)")
    ax2.set_xlabel("Prediccion")
    ax2.set_ylabel("Clase Real")
    plt.colorbar(im2, ax=ax2)
    ax2.set_xticks(range(len(etiquetas)))
    ax2.set_yticks(range(len(etiquetas)))
    ax2.set_xticklabels(etiquetas, rotation=40, ha="right", fontsize=9)
    ax2.set_yticklabels(etiquetas, fontsize=9)
    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            ax2.text(j, i, f"{cm_norm[i, j]:.2f}",
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black",
                     fontsize=9)

    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Matriz guardada: {ruta}")


def medir_latencia_pipeline(modelo, n_repeticiones=100):
    # Mide el tiempo de preprocesamiento e inferencia con audio sintetico
    from preprocess import segmentar_audio, normalizar_amplitud, extraer_mfcc
    from preprocess import TASA_MUESTREO, MUESTRAS_OBJETIVO

    t_prepro, t_infer, t_total = [], [], []
    audio_base = np.random.randn(MUESTRAS_OBJETIVO).astype(np.float32) * 0.3

    for _ in range(n_repeticiones):
        t0    = time.perf_counter()
        audio = segmentar_audio(audio_base)
        audio = normalizar_amplitud(audio)
        mfcc  = extraer_mfcc(audio)
        entrada = np.expand_dims(mfcc, axis=0)
        t1    = time.perf_counter()
        t_prepro.append((t1 - t0) * 1000)

        modelo.predict(entrada, verbose=0)
        t2 = time.perf_counter()
        t_infer.append((t2 - t1) * 1000)
        t_total.append((t2 - t0) * 1000)

    def estadisticas(arr):
        return {"promedio": float(np.mean(arr)),
                "mediana" : float(np.median(arr)),
                "maximo"  : float(np.max(arr)),
                "minimo"  : float(np.min(arr))}

    resultados = {
        "preprocesamiento_ms": estadisticas(t_prepro),
        "inferencia_ms"      : estadisticas(t_infer),
        "total_pipeline_ms"  : estadisticas(t_total),
        "n_repeticiones"     : n_repeticiones,
    }

    print(f"\n  LATENCIA ({n_repeticiones} repeticiones)")
    print("-" * 55)
    print(f"  {'Componente':<22}  {'Prom':>8}  {'Med':>8}  {'Max':>8}")
    print("-" * 55)
    for nombre, clave in [("Preprocesamiento", "preprocesamiento_ms"),
                           ("Inferencia CNN",   "inferencia_ms"),
                           ("TOTAL (sin mic)",  "total_pipeline_ms")]:
        e = resultados[clave]
        print(f"  {nombre:<22}  {e['promedio']:>6.1f}ms  "
              f"{e['mediana']:>6.1f}ms  {e['maximo']:>6.1f}ms")

    margen = 500 - resultados["total_pipeline_ms"]["promedio"]
    cumple = margen > 0
    print("-" * 55)
    print(f"  Margen para 500 ms: {margen:+.1f} ms  ({'OK' if cumple else 'EXCEDE'})")

    lat_mic = (1024 / TASA_MUESTREO) * 1000
    print(f"  Latencia mic estimada : ~{lat_mic:.0f} ms")
    print(f"  TOTAL sistema estimado: ~{resultados['total_pipeline_ms']['promedio'] + lat_mic:.0f} ms")

    return resultados


def main():
    print("\n" + "=" * 55)
    print("  EVALUACION DEL SISTEMA DE RECONOCIMIENTO DE VOZ")
    print("=" * 55)

    os.makedirs(DIR_REPORTES, exist_ok=True)
    reporte_final = {}
    ruta_cnn  = os.path.join(DIR_MODELOS, "modelo_cnn_comandos.h5")

    print("\n[1/3] Evaluando modelo CNN...")
    X_cnn, y_cnn = cargar_datos_prueba(sufijo="")

    if X_cnn is not None and os.path.exists(ruta_cnn):
        modelo_cnn = cargar_modelo_keras(ruta_cnn)
        if modelo_cnn is not None:
            y_pred_cnn, met_cnn = evaluar_modelo(
                modelo_cnn, X_cnn, y_cnn, CLASES_CNN, "CNN Comandos Simples"
            )
            graficar_matriz_confusion(
                y_cnn, y_pred_cnn, CLASES_CNN,
                "CNN Comandos Simples",
                os.path.join(DIR_REPORTES, "confusion_cnn.png")
            )
            reporte_final["cnn"] = met_cnn
    else:
        print("  Modelo CNN o datos de prueba no disponibles.")

    print("\n[2/3] Analizando latencia del pipeline...")
    if os.path.exists(ruta_cnn):
        modelo_lat = cargar_modelo_keras(ruta_cnn)
        if modelo_lat:
            reporte_final["latencia"] = medir_latencia_pipeline(modelo_lat)
    else:
        print("  Modelo CNN no disponible para analisis de latencia.")

    print("\n[3/3] Guardando reporte JSON...")
    if reporte_final:
        ruta_reporte = os.path.join(DIR_REPORTES, "reporte_evaluacion.json")
        with open(ruta_reporte, "w", encoding="utf-8") as f:
            json.dump(reporte_final, f, indent=2, ensure_ascii=False)
        print(f"  Reporte guardado: {ruta_reporte}")

    print("\n" + "=" * 55)
    print("  EVALUACION COMPLETADA")
    print("=" * 55)


if __name__ == "__main__":
    main()
