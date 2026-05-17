#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Comunicacion serial con el Arduino via USB a 9600 baudios.
# Si no hay Arduino conectado, activa modo simulacion automaticamente.

import time
import threading

try:
    import serial
    import serial.tools.list_ports
    SERIAL_DISPONIBLE = True
except ImportError:
    SERIAL_DISPONIBLE = False
    print("pyserial no instalado. Instala con: pip install pyserial")

VELOCIDAD_BAUDIOS = 115200
TIEMPO_ESPERA_S   = 1.0
RETARDO_POST_CMD  = 0.05

# Mapeo clase del modelo -> string que recibe el Arduino
# None significa que esa clase no genera accion
MAPA_COMANDOS = {
    "AVANZA"      : "AVANZA",
    "RETROCEDE"   : "RETROCEDE",
    "IZQUIERDA"   : "IZQUIERDA",
    "DERECHA"     : "DERECHA",
    "DETENTE"     : "DETENTE",
    "RUIDO_FONDO" : None,
}


def detectar_puerto_arduino():
    # Busca el puerto COM donde esta conectado el Arduino
    if not SERIAL_DISPONIBLE:
        return None

    palabras_clave = ["Arduino", "arduino", "CH340", "CH341", "FTDI",
                      "CP210", "Silicon Labs", "wch.cn"]

    puertos = serial.tools.list_ports.comports()
    for puerto in puertos:
        info = f"{puerto.description} {puerto.manufacturer or ''}"
        if any(kw in info for kw in palabras_clave):
            return puerto.device

    # Fallback: usar primer ttyUSB o ttyACM disponible
    for puerto in puertos:
        if "ttyUSB" in puerto.device or "ttyACM" in puerto.device:
            return puerto.device

    return None


class ComunicadorArduino:

    def __init__(self, puerto=None, baudios=VELOCIDAD_BAUDIOS):
        self.baudios         = baudios
        self.conexion        = None
        self.modo_simulacion = False
        self._lock           = threading.Lock()
        self._puerto_nombre  = puerto
        self._inicializar_conexion()

    def _inicializar_conexion(self):
        if not SERIAL_DISPONIBLE:
            self._activar_simulacion("pyserial no instalado")
            return

        if self._puerto_nombre is None:
            self._puerto_nombre = detectar_puerto_arduino()

        if self._puerto_nombre is None:
            self._activar_simulacion("no se detecto ningun Arduino")
            return

        ultimo_error = None
        for intento in range(3):
            try:
                self.conexion = serial.Serial(
                    port=self._puerto_nombre,
                    baudrate=self.baudios,
                    timeout=TIEMPO_ESPERA_S,
                )
                # El Arduino reinicia al abrir el puerto; esperar a que termine el setup()
                time.sleep(4.0)
                print(f"Arduino conectado en {self._puerto_nombre} a {self.baudios} bps")
                self.modo_simulacion = False
                return
            except serial.SerialException as e:
                ultimo_error = e
                if intento < 2:
                    print(f"  Puerto ocupado, reintentando ({intento + 1}/3)...")
                    time.sleep(2.0)

        self._activar_simulacion(str(ultimo_error))

    def _activar_simulacion(self, razon):
        self.modo_simulacion = True
        print(f"Arduino no disponible ({razon}). Modo simulacion activo.")

    def enviar_comando(self, nombre_clase):
        # Envia "COMANDO\n" al Arduino; retorna False si la clase no tiene accion
        comando = MAPA_COMANDOS.get(nombre_clase)

        if comando is None:
            return False

        payload = f"{comando}\n"

        if self.modo_simulacion:
            print(f"  [SIM] Arduino recibe: {comando}")
            return True

        with self._lock:
            try:
                self.conexion.reset_input_buffer()
                self.conexion.write(payload.encode("utf-8"))
                self.conexion.flush()
                print(f"  [SERIAL] Enviado: {repr(payload)}")
                return True

            except serial.SerialException as e:
                print(f"Error serial: {e}. Cambiando a modo simulacion.")
                self.modo_simulacion = True
                return False

    def leer_respuesta(self, timeout_s=0.3):
        if self.modo_simulacion or self.conexion is None:
            return None
        try:
            self.conexion.timeout = timeout_s
            linea = self.conexion.readline().decode("utf-8", errors="ignore").strip()
            return linea or None
        except Exception:
            return None

    @property
    def conectado(self):
        return (not self.modo_simulacion
                and self.conexion is not None
                and self.conexion.is_open)

    def cerrar(self):
        if self.conexion and self.conexion.is_open:
            self.conexion.close()
            print("Conexion serial cerrada.")


if __name__ == "__main__":
    print("Prueba de comunicacion con Arduino")
    print("-" * 40)

    arduino = ComunicadorArduino()
    print(f"Puerto     : {arduino._puerto_nombre or 'N/A'}")
    print(f"Baudios    : {arduino.baudios}")
    print(f"Conectado  : {arduino.conectado}")
    print(f"Simulacion : {arduino.modo_simulacion}")
    print()

    secuencia_prueba = [
        ("AVANZA",      True),
        ("IZQUIERDA",   True),
        ("DERECHA",     True),
        ("RETROCEDE",   True),
        ("DETENTE",     True),
        ("RUIDO_FONDO", False),
    ]

    print("Enviando comandos de prueba:")
    for nombre, esperado_envio in secuencia_prueba:
        enviado = arduino.enviar_comando(nombre)
        estado  = "OK" if enviado == esperado_envio else "ERROR"
        print(f"  {nombre:<14} enviado={enviado}  [{estado}]")
        time.sleep(0.4)

    arduino.cerrar()
    print("Prueba completada.")
