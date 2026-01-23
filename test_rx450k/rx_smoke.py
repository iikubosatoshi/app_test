import time
import serial

PORT = "/dev/cu.usbserial-OMC02001"

with serial.Serial(PORT, baudrate=9600, timeout=0.1) as ser:
    ser.dtr = True
    ser.rts = True
    time.sleep(0.2)
    ser.reset_input_buffer()

    # 何も送らずに 2秒だけ受信を覗く
    buf = bytearray()
    t0 = time.time()
    while time.time() - t0 < 2.0:
        buf += ser.read(4096)

    print("received bytes:", len(buf))
    if buf:
        print("raw:", buf[:200])

    # 何か送って 2秒覗く（改行に依存しない）
    ser.write(b"\r")
    ser.flush()

    buf2 = bytearray()
    t0 = time.time()
    while time.time() - t0 < 2.0:
        buf2 += ser.read(4096)

    print("received after CR bytes:", len(buf2))
    if buf2:
        print("raw2:", buf2[:200])
