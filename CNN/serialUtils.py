import time
import serial
import subprocess
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUDRATE = 7200

def getSerialPort():
    serialPort = ""
    while(len(serialPort) == 0):
        serial = subprocess.run("ls /dev/ttyUSB*", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        serialPort = serial.stdout.strip().decode()
    return serialPort
def serialInit(serial_port=SERIAL_PORT, baudrate=SERIAL_BAUDRATE):
    while(True):
        try:
            ser = serial.Serial(port=serial_port, baudrate=baudrate,
                                parity=serial.PARITY_NONE,
                                stopbits=serial.STOPBITS_ONE,
                                bytesize=serial.EIGHTBITS,
                                timeout=None)
            return ser
        except:
            print("device not found")

