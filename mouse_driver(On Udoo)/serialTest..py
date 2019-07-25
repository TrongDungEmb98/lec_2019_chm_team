import serial
import time
import subprocess
import ctypes as ct
SERIAL_PORT = "/dev/ttymxc5"
SERIAL_BAUDRATE = 7200
lib = ct.cdll.LoadLibrary('/home/udooer/LEC/libtest.so')
lib.open_file()
def getSerialPort():
    serialPort = ""
    while(len(serialPort) == 0):
        serial = subprocess.call("ls /dev/ttyGS*", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
       # serialPort = serial.stdout.strip().decode()
    return serial
def serialInit(serial_port=SERIAL_PORT, baudrate=SERIAL_BAUDRATE):
    #while(True):
        #try:
    ser = serial.Serial(port=serial_port, baudrate=baudrate, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS,timeout=None)
    return ser
        #except:
            #print("device not found")
if __name__ == '__main__':
    data = ""
    serial = serialInit("/dev/ttymxc5",7200)
    time.sleep(1)
    while(True):
        try:
            if(serial.in_waiting > 0):
                ser = serial.read(11)
                data = ser.decode('utf-8')
                data1 = data.split("/")
                #print(data,"--",len(data))
                print(data1)
                #if (len(data1[0])==4 and len(data1[1])==4):
                dx= int(data1[0])
                dy= int(data1[1])
                event= int(data1[2])
                #else:
                #dx = dy = event = 0
                if (event == 1) :
                        lib.write_value(0,0,1)
                        lib.write_value(0,0,2)
                        time.sleep(0.1)
                        lib.write_value(0,0,1)
                        lib.write_value(0,0,2)
                elif (event == 2) :
                        lib.write_value(0,0,1)
                        lib.write_value(0,0,2)
                elif (event == 3) :
                        lib.write_value(0,0,3)
                        lib.write_value(0,0,4)
                else :            
                        lib.write_value(dx,dy,0)
           # time.sleep(1)
        except KeyboardInterrupt:
            print("exitting...")
            serial.close()
            lib.close_file
            exit()
