import serial

PORT = "/dev/cu.usbserial-OMC02001"

ser = serial.Serial(
    PORT,
    baudrate=9600,   # ← RX-450Kはこの可能性が高い
    timeout=1
)

print("connected")

while True:
    data = ser.readline()
    if data:
        print(data)
