#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Inferencia CNN 1D sin tensorflow — usa solo numpy + h5py.

import numpy as np
try:
    import h5py
except ImportError:
    import pyfive as h5py


# ── Operaciones de red neuronal ──────────────────────────────────────────────

def _relu(x):
    return np.maximum(0.0, x)


def _softmax(x):
    e = np.exp(x - np.max(x))
    return (e / e.sum()).astype(np.float32)


def _conv1d_same(x, W, b):
    """x:(T,Cin)  W:(k,Cin,Cout)  b:(Cout,) -> (T,Cout)"""
    T, Cin = x.shape
    k = W.shape[0]
    pad = k // 2
    xp = np.pad(x, ((pad, pad), (0, 0)))
    s = xp.strides
    windows = np.lib.stride_tricks.as_strided(
        xp, shape=(T, k, Cin), strides=(s[0], s[0], s[1])
    )
    return (np.einsum('tkc,kco->to', windows, W) + b).astype(np.float32)


def _bn(x, gamma, beta, mean, var, eps=1e-3):
    return (gamma * (x - mean) / np.sqrt(var + eps) + beta).astype(np.float32)


def _maxpool1d(x, pool=2):
    T, C = x.shape
    T2 = T // pool
    return x[:T2 * pool].reshape(T2, pool, C).max(axis=1)


def _gap1d(x):
    return x.mean(axis=0).astype(np.float32)


def _dense(x, W, b):
    return (x @ W + b).astype(np.float32)


# ── Lectura de pesos desde .h5 ───────────────────────────────────────────────

def _leer_pesos(grupo_h5, *nombres):
    """Busca datasets por nombre en cualquier nivel del grupo."""
    encontrados = {}

    def _buscar(g):
        for key in g.keys():
            item = g[key]
            if key in nombres and hasattr(item, 'shape') and not hasattr(item, 'keys'):
                try:
                    datos = item[()]
                except Exception:
                    datos = item[...]
                encontrados[key] = np.asarray(datos, dtype=np.float32)
            elif hasattr(item, 'keys'):
                _buscar(item)

    _buscar(grupo_h5)
    faltantes = [n for n in nombres if n not in encontrados]
    if faltantes:
        raise KeyError(f"No se encontraron: {faltantes} en {grupo_h5.name}")
    return [encontrados[n] for n in nombres]


def _grupo_capa(f, nombre_capa):
    """Devuelve el grupo h5 de una capa dado su nombre."""
    rutas = [
        f"model_weights/{nombre_capa}",
        nombre_capa,
    ]
    for ruta in rutas:
        if ruta in f:
            return f[ruta]
    raise KeyError(f"Capa '{nombre_capa}' no encontrada en el archivo h5")


# ── Modelo CNN ───────────────────────────────────────────────────────────────

class ModeloCNN:

    def __init__(self, ruta_h5):
        print(f"  Cargando pesos numpy desde {ruta_h5}...", end="", flush=True)
        with h5py.File(ruta_h5, 'r') as f:

            def _c(nombre):
                g = _grupo_capa(f, nombre)
                return _leer_pesos(g, 'kernel', 'bias')

            def _b(nombre):
                g = _grupo_capa(f, nombre)
                return _leer_pesos(g, 'gamma', 'beta',
                                   'moving_mean', 'moving_variance')

            self.c1_W, self.c1_b = _c('conv1d')
            self.c2_W, self.c2_b = _c('conv1d_1')
            self.c3_W, self.c3_b = _c('conv1d_2')

            self.bn1 = _b('batch_normalization')
            self.bn2 = _b('batch_normalization_1')
            self.bn3 = _b('batch_normalization_2')
            self.bn4 = _b('batch_normalization_3')

            self.d1_W, self.d1_b = _c('dense')
            self.d2_W, self.d2_b = _c('dense_1')

        n_clases = self.d2_W.shape[1]
        print(f"  OK  ({n_clases} clases)")

    def predecir(self, entrada):
        """entrada: (1, T, C) -> array de probabilidades shape (n_clases,)"""
        x = entrada[0]  # quita la dimension de batch -> (T, C)

        # Bloque 1: Conv(relu) -> BN -> MaxPool
        x = _relu(_conv1d_same(x, self.c1_W, self.c1_b))
        x = _bn(x, *self.bn1)
        x = _maxpool1d(x, 2)

        # Bloque 2: Conv(relu) -> BN -> MaxPool
        x = _relu(_conv1d_same(x, self.c2_W, self.c2_b))
        x = _bn(x, *self.bn2)
        x = _maxpool1d(x, 2)

        # Bloque 3: Conv(relu) -> BN -> GlobalAvgPool
        x = _relu(_conv1d_same(x, self.c3_W, self.c3_b))
        x = _bn(x, *self.bn3)
        x = _gap1d(x)

        # Dense 1: Dense(relu) -> BN
        x = _relu(_dense(x, self.d1_W, self.d1_b))
        x = _bn(x, *self.bn4)

        # Dense 2: Dense -> Softmax
        return _softmax(_dense(x, self.d2_W, self.d2_b))
