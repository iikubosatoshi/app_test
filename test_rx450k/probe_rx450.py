import time
import serial

PORT = "/dev/cu.usbserial-OMC02001"

BAUDS = [2400, 4800, 9600, 19200, 38400, 57600, 115200]

# よくある「起きる」きっかけになりがちな送信（改行/NULL/ENQなど）
PROBES = [
    b"\r\n",
    b"\n",
    b"\r",
    b"\x00",
    b"\x05",      # ENQ
    b"?",         # 汎用
    b"ID\r\n",    # 汎用
]

def try_baud(baud: int):
    print(f"\n=== baud={baud} ===")
    with serial.Serial(PORT, baudrate=baud, timeout=0.3) as ser:
        # ありがちな制御線（機器によって効く）
        ser.dtr = True
        ser.rts = True
        time.sleep(0.2)

        # 受信バッファを空に
        ser.reset_input_buffer()

        for p in PROBES:
            ser.write(p)
            ser.flush()
            time.sleep(0.25)
            data = ser.read(256)
            if data:
                print("sent:", p, "recv:", data)
                return True

        # 送信しなくても吐くタイプの可能性も一応見る
        time.sleep(0.5)
        data = ser.read(256)
        if data:
            print("recv(no probe):", data)
            return True

    return False

def main():
    for b in BAUDS:
        if try_baud(b):
            print("\n>>> got response! baud may be", b)
            break
    else:
        print("\n(no response at common baudrates/probes)")

if __name__ == "__main__":
    main()
