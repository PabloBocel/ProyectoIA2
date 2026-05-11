# Asistente Robótico por Comandos de Voz

**Universidad Rafael Landívar — Inteligencia Artificial — Primer Semestre 2026**

Robot móvil controlado por comandos de voz en español, con reconocimiento de voz
implementado desde cero usando CNN 1D entrenada con corpus propio.
Todo el sistema funciona sin conexión a internet.

---

## Arquitectura del sistema

```
Micrófono → VAD → MFCC → CNN 1D → Comando → Arduino → Motores DC
```

| Componente | Tecnología |
|---|---|
| Captura de audio | sounddevice (16 kHz, mono) |
| Detección de voz | Energía RMS + ZCR (implementación propia) |
| Features | MFCC (40 coeficientes, ventana 32 ms, hop 10 ms) |
| Modelo | CNN 1D — 6 clases (5 comandos + ruido de fondo) |
| Comunicación | Serial USB 9600 bps → Arduino |
| Hardware del robot | Modalidad A — Robot Móvil (tracción diferencial) |

---

## Comandos reconocidos

| Clase | Descripción |
|---|---|
| `AVANZA` | El robot avanza hacia adelante |
| `RETROCEDE` | El robot retrocede |
| `IZQUIERDA` | Giro a la izquierda |
| `DERECHA` | Giro a la derecha |
| `DETENTE` | El robot se detiene |
| `RUIDO_FONDO` | Silencio / habla no registrada (sin acción) |

> `RUIDO_FONDO` es obligatorio para que el modelo pueda "no hacer nada"
> cuando hay ruido ambiente. Sin esta clase, cualquier sonido activa un comando.

---

## Instalación

### 1. Requisitos de Python
```bash
pip install tensorflow librosa sounddevice pyserial scipy scikit-learn matplotlib numpy
```

> **Raspberry Pi:** instala primero las dependencias del sistema:
> ```bash
> sudo apt-get update
> sudo apt-get install python3-pip portaudio19-dev libsndfile1
> pip3 install tensorflow-aarch64  # versión ARM de TensorFlow
> ```

### 2. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd voice_robot
```

### 3. Verificar estructura de carpetas
```
voice_robot/
├── record_audio.py
├── preprocess.py
├── augment.py
├── train.py
├── inference.py
├── arduino_comm.py
├── evaluate.py
├── dataset/
│   ├── AVANZA/
│   ├── RETROCEDE/
│   ├── IZQUIERDA/
│   ├── DERECHA/
│   ├── DETENTE/
│   └── RUIDO_FONDO/
├── models/          ← modelos entrenados (.h5, .tflite)
└── reports/         ← métricas y gráficas generadas
```

---

## Paso 1 — Grabar el corpus de audio

```bash
python record_audio.py
```

El script te guía para grabar **200 muestras por comando** con cuenta regresiva visual.
Consejos:
- Graba en al menos **dos entornos** (sala silenciosa y laboratorio con ruido).
- Cada integrante del grupo debe grabar su propia voz.
- Invita **5 voluntarios externos** para mayor variabilidad.
- Para `RUIDO_FONDO`: captura silencio, conversaciones lejanas y ruido ambiental.

Mínimo requerido:
- 5 clases × 200 muestras = **1 000 muestras**
- 1 clase RUIDO_FONDO × 200 = **200 muestras**
- **Total: 1 200 muestras**

---

## Paso 2 — Entrenar el modelo CNN

```bash
python train.py
```

El script:
1. Aplica Data Augmentation (factor 3×) sobre las grabaciones originales.
2. Extrae MFCC de todas las muestras.
3. Divide en entrenamiento / validación / prueba (70/15/15).
4. Entrena la CNN 1D por hasta 50 épocas con EarlyStopping.
5. Exporta `models/modelo_cnn_comandos.h5` y `.tflite`.
6. Guarda `models/historial_cnn.png` con las curvas de entrenamiento.

> Si entrenás en una PC, copia `models/modelo_cnn_comandos.h5` y
> `models/config_cnn.json` a la Raspberry Pi antes de correr la inferencia.

---

## Paso 3 — Evaluar el modelo

```bash
python evaluate.py
```

Genera en `reports/`:
- `confusion_cnn.png` — Matriz de confusión (absoluta y normalizada)
- `reporte_evaluacion.json` — Accuracy, F1, latencia por componente

---

## Paso 4 — Inferencia en tiempo real

```bash
# Auto-detectar puerto Arduino
python inference.py

# Especificar puerto manualmente (Linux/Raspberry Pi)
python inference.py --puerto /dev/ttyUSB0

# Especificar puerto manualmente (Windows)
python inference.py --puerto COM3
```

El sistema:
1. Captura audio continuo del micrófono (bloques de 64 ms).
2. Aplica VAD para detectar voz activa en el buffer.
3. Cuando detecta voz, extrae MFCC y clasifica con la CNN.
4. Si la confianza supera el 75 %, envía el comando al Arduino por USB serial.
5. Imprime estadísticas de latencia al salir (Ctrl+C).

> **Sin modelo entrenado:** el sistema funciona en MODO DEMOSTRACIÓN
> (predicciones aleatorias) para probar el pipeline de comunicación con el Arduino.

---

## Código Arduino (sketch de referencia)

Copia este sketch al Arduino antes de la demostración:

```cpp
// sketch_robot_comandos.ino
// Motor izquierdo: pines 5 (IN1) y 6 (IN2)
// Motor derecho  : pines 9 (IN3) y 10 (IN4)

#define IN1 5
#define IN2 6
#define IN3 9
#define IN4 10

void setup() {
  Serial.begin(9600);
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  detener();
  Serial.println("Robot listo.");
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if      (cmd == "AVANZA")    avanzar();
    else if (cmd == "RETROCEDE") retroceder();
    else if (cmd == "IZQUIERDA") girarIzquierda();
    else if (cmd == "DERECHA")   girarDerecha();
    else if (cmd == "DETENTE")   detener();

    Serial.println("OK:" + cmd);
  }
}

void avanzar()        { analogWrite(IN1,200); digitalWrite(IN2,LOW);
                        analogWrite(IN3,200); digitalWrite(IN4,LOW); }
void retroceder()     { digitalWrite(IN1,LOW); analogWrite(IN2,200);
                        digitalWrite(IN3,LOW); analogWrite(IN4,200); }
void girarDerecha()   { analogWrite(IN1,180); digitalWrite(IN2,LOW);
                        digitalWrite(IN3,LOW); analogWrite(IN4,180); }
void girarIzquierda() { digitalWrite(IN1,LOW); analogWrite(IN2,180);
                        analogWrite(IN3,180); digitalWrite(IN4,LOW); }
void detener()        { digitalWrite(IN1,LOW); digitalWrite(IN2,LOW);
                        digitalWrite(IN3,LOW); digitalWrite(IN4,LOW); }
```

---

## Parámetros técnicos

| Parámetro | Valor | Justificación |
|---|---|---|
| Tasa de muestreo | 16 000 Hz | Estándar mínimo para reconocimiento de voz |
| Duración muestra | 1.0 s | Suficiente para comandos de una sola palabra |
| Tamaño ventana (n_fft) | 512 muestras ≈ 32 ms | Captura estacionaridad fonémica |
| Hop length | 160 muestras ≈ 10 ms | Resolución temporal de 10 ms, solapamiento 69 % |
| Coeficientes MFCC | 40 (mín. exigido: 13) | Mayor detalle espectral del habla en español |
| Bandas Mel | 128 | Resolución del banco de filtros |
| Umbral de energía VAD | 0.002 | Separa habla de silencio en ambiente típico |
| Umbral ZCR VAD | 0.30 | Descarta ruido de alta frecuencia |
| Umbral de confianza | 75 % | Rechaza predicciones ambiguas |
| Cooldown entre comandos | 1.0 s | Evita repetición involuntaria |

---

## Data Augmentation aplicada

| Técnica | Parámetro | Efecto |
|---|---|---|
| Desplazamiento temporal | ±20 % de la duración | Variación en inicio del habla |
| Cambio de tono | ±3 semitonos | Simula distintos registros vocales |
| Ruido gaussiano | σ ∈ [0.002, 0.010] | Robustez ante ruido ambiental |
| Estiramiento temporal | factor ∈ [0.80, 1.20] | Variación de velocidad del habla |

---

## Restricciones del proyecto

- Sin APIs externas de reconocimiento de voz (Google, Whisper, Azure, etc.)
- Sin modelos preentrenados de voz (Wav2Vec2, Vosk, etc.)
- El sistema funciona completamente sin conexión a internet
- Los modelos fueron entrenados desde cero con el corpus propio del grupo

---

## Equipo

| Nombre | Carné | Responsabilidad |
|---|---|---|
| (completar) | (completar) | (completar) |
| (completar) | (completar) | (completar) |
| (completar) | (completar) | (completar) |

**Entrega:** Lunes 18 de mayo de 2026  
**Presentación:** Miércoles 20 de mayo de 2026
