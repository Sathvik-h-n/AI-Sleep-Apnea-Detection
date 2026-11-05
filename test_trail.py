import serial

try:
    ser = serial.Serial("COM3", 115200, timeout=1)
    print("OK! Serial opened.")
    ser.close()
except Exception as e:
    print("ERROR:", e)
