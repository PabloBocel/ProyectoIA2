# Asistente Robótico por Comandos de Voz

Robot móvil controlado por comandos de voz en español, con reconocimiento de voz
implementado desde cero usando CNN 1D entrenada con corpus propio.
Todo el sistema funciona sin conexión a internet.

---

## Arquitectura del sistema

```
Micrófono → VAD → MFCC → CNN 1D → Comando → Arduino → Motores DC
```

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
├── preprocess.py
├── augment.py
├── train.py
├── inference.py
├── arduino_comm.py
├── evaluate.py
├── models/          ← modelos entrenados (.h5, .tflite)
└── reports/         ← métricas y gráficas generadas
```

> La carpeta `dataset/` (1 456 archivos .wav, ~315 MB) no se incluye en el repositorio
> ni en la Raspberry Pi — solo es necesaria para reentrenar el modelo.

---

## Paso 1 — Obtener el corpus de audio

Los audios fueron grabados por separado (fuera del repositorio) usando dispositivos
de los integrantes del grupo y voluntarios externos, con parámetros de 16 kHz, mono
y 1 segundo por muestra.

El corpus completo se compartió mediante **Google Drive** y se descargó localmente.
Una vez descargados, los archivos `.wav` se colocaron manualmente en las carpetas
correspondientes del dataset:

```
dataset/
├── AVANZA/        ← archivos .wav de "avanza"
├── RETROCEDE/     ← archivos .wav de "retrocede"
├── IZQUIERDA/     ← archivos .wav de "izquierda"
├── DERECHA/       ← archivos .wav de "derecha"
├── DETENTE/       ← archivos .wav de "detente"
└── RUIDO_FONDO/   ← archivos .wav de ruido/silencio
```

Corpus recopilado:
- 5 clases × ~250 muestras = **~1 250 muestras**
- 1 clase RUIDO_FONDO × ~200 = **~200 muestras**
- **Total: 1 456 muestras**

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

## Data Augmentation aplicada

| Técnica | Parámetro | Efecto |
|---|---|---|
| Desplazamiento temporal | ±20 % de la duración | Variación en inicio del habla |
| Cambio de tono | ±3 semitonos | Simula distintos registros vocales |
| Ruido gaussiano | σ ∈ [0.002, 0.010] | Robustez ante ruido ambiental |
| Estiramiento temporal | factor ∈ [0.80, 1.20] | Variación de velocidad del habla |

---