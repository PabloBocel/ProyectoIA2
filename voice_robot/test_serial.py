#!/usr/bin/env python3
# Script de diagnostico: manda comandos y muestra en tiempo real lo que llega del Arduino.
import serial
import serial.tools.list_ports
import threading
import time

BAUDIOS = 9600
_detener_lector = threading.Event()

def lector_continuo(ser):
    """Hilo que imprime todo lo que manda el Arduino en tiempo real."""
    while not _detener_lector.is_set():
        try:
            if ser.in_waiting:
                linea = ser.readline().decode("utf-8", errors="ignore").strip()
                if linea:
                    print(f"\n  [ARDUINO] {linea}")
            else:
                time.sleep(0.01)
        except Exception:
            break

def listar_puertos():
    puertos = serial.tools.list_ports.comports()
    print("Puertos disponibles:")
    for p in puertos:
        print(f"  {p.device:10} | {p.description}")
    return puertos

def main():
    listar_puertos()
    puerto = input("\nEscribe el puerto (ej. COM7): ").strip()

    print(f"\nConectando a {puerto} a {BAUDIOS} bps...")
    try:
        ser = serial.Serial(puerto, BAUDIOS, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
        print("Conectado. Leyendo respuestas del Arduino en tiempo real...\n")

        hilo = threading.Thread(target=lector_continuo, args=(ser,), daemon=True)
        hilo.start()

        print("Comandos: AVANZA, RETROCEDE, IZQUIERDA, DERECHA, DETENTE")
        print("Escribe 'q' para salir.\n")

        while True:
            cmd = input("Comando: ").strip().upper()
            if cmd == "Q":
                break
            if not cmd:
                continue

            payload = cmd + "\n"
            n = ser.write(payload.encode("utf-8"))
            ser.flush()
            print(f"  -> Enviado {n} bytes: {repr(payload)}")

        _detener_lector.set()
        ser.close()
        print("Puerto cerrado.")

    except serial.SerialException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
