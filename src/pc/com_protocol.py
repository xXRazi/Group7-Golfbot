#!/usr/bin/env python3
import socket
import struct
import sys


HOST = "ev3dev.local"
PORT = 5000

ERROR = 0x0
CALIBRATE = 0x1
SENDMAP = 0x2
HANDSHAKE = 0xA
GOTO = 0xB
POSSYNC = 0xC
TURN = 0xD
SETSPEED = 0xE
FINISH = 0xF


def encode_signed_byte(value):
    """Convert signed int -128..127 to one byte 0..255."""
    if not (-128 <= value <= 127):
        raise ValueError("value must be between -128 and 127")
    return value % 256


def validate_signed_byte(value, name):
    if not (-128 <= value <= 127):
        raise ValueError("{} must be between -128 and 127".format(name))
    return value


def validate_signed_short(value, name):
    if not (-32768 <= value <= 32767):
        raise ValueError("{} must be between -32768 and 32767".format(name))
    return value


def validate_signed_int(value, name):
    if not (-2147483648 <= value <= 2147483647):
        raise ValueError("{} must be between -2147483648 and 2147483647".format(name))
    return value


def validate_unsigned_short(value, name):
    if not (0 <= value <= 65535):
        raise ValueError("{} must be between 0 and 65535".format(name))
    return value


def validate_unsigned_byte(value, name):
    if not (0 <= value <= 255):
        raise ValueError("{} must be between 0 and 255".format(name))
    return value


def packet_preview(packet, max_bytes=40):
    preview = list(packet[:max_bytes])
    if len(packet) > max_bytes:
        return "{} ... total {} bytes".format(preview, len(packet))
    return "{} total {} bytes".format(preview, len(packet))


def recv_response(sock):
    try:
        data = sock.recv(1024)
        if not data:
            print("Robot closed the connection")
            return False

        print("Robot:", data.decode("utf-8", errors="replace").strip())
        return True

    except OSError as exc:
        print("Receive error:", exc)
        return False


def send_command(sock, packet):
    try:
        sock.sendall(packet)
        print("Sent:", packet_preview(packet))
        return recv_response(sock)

    except OSError as exc:
        print("Send error:", exc)
        return False


def build_handshake():
    return bytes([HANDSHAKE])


def build_calibrate(left_trim, right_trim):
    left_trim = validate_signed_byte(left_trim, "left_trim")
    right_trim = validate_signed_byte(right_trim, "right_trim")

    return bytes([
        CALIBRATE,
        encode_signed_byte(left_trim),
        encode_signed_byte(right_trim),
    ])


def build_goto(x, y):
    """
    GOTO packet:
    [CMD][X:int32][Y:int32]

    Total length: 9 bytes.
    """
    x = validate_signed_int(x, "x")
    y = validate_signed_int(y, "y")

    return struct.pack(">Bii", GOTO, x, y)


def build_possync(x, y, heading_tenths):
    """
    POSSYNC packet:
    [CMD][X:int32][Y:int32][HEADING:int16]

    Total length: 11 bytes.
    """
    x = validate_signed_int(x, "x")
    y = validate_signed_int(y, "y")
    heading_tenths = validate_signed_short(heading_tenths, "heading_tenths")

    return struct.pack(">Biih", POSSYNC, x, y, heading_tenths)


def build_turn(angle, speed):
    """
    TURN packet:
    [CMD][ANGLE:int16][SPEED:int8]

    Total length: 4 bytes.
    """
    angle = validate_signed_short(angle, "angle")
    speed = validate_signed_byte(speed, "speed")

    return struct.pack(">Bhb", TURN, angle, speed)


def build_setspeed(left, right):
    left = validate_signed_byte(left, "left")
    right = validate_signed_byte(right, "right")

    return bytes([
        SETSPEED,
        encode_signed_byte(left),
        encode_signed_byte(right),
    ])


def build_sendmap(rows, cols, cells):
    """
    SENDMAP packet:
    [CMD][ROWS:uint16][COLS:uint16][MAP...]

    This supports a 480 x 640 map.
    The cell values are bytes 0..255.
    """
    rows = validate_unsigned_short(rows, "rows")
    cols = validate_unsigned_short(cols, "cols")

    expected = rows * cols

    if len(cells) != expected:
        raise ValueError(
            "sendmap needs exactly {} cell values for a {}x{} map".format(
                expected, rows, cols
            )
        )

    validated_cells = [validate_unsigned_byte(cell, "cell") for cell in cells]

    return struct.pack(">BHH", SENDMAP, rows, cols) + bytes(validated_cells)


def build_sendmap_fill(rows, cols, value):
    """
    Convenience command for testing large maps without typing all cells.

    Example:
      sendmap_fill 480 640 0
    """
    rows = validate_unsigned_short(rows, "rows")
    cols = validate_unsigned_short(cols, "cols")
    value = validate_unsigned_byte(value, "value")

    cells = bytes([value]) * (rows * cols)

    return struct.pack(">BHH", SENDMAP, rows, cols) + cells


def build_finish():
    return bytes([FINISH])


def print_help():
    print()
    print("Commands:")
    print("  handshake")
    print("  calibrate LEFT_TRIM RIGHT_TRIM")
    print("  goto X Y")
    print("  possync X Y")
    print("  possync X Y HEADING_TENTHS  heading is accepted but ignored")
    print("  turn ANGLE SPEED")
    print("  setspeed LEFT RIGHT")
    print("  sendmap ROWS COLS CELL1 CELL2 ...")
    print("  sendmap_fill ROWS COLS VALUE")
    print("  finish")
    print("  help")
    print("  quit")
    print()
    print("Notes:")
    print("  goto uses signed 32-bit x/y")
    print("  possync sends only x/y to match the current EV3 protocol")
    print("  possync packet length is 9 bytes, not 11")
    print("  heading is not sent unless the EV3 protocol is updated too")
    print("  turn angle uses signed 16-bit integer")
    print("  turn speed uses signed byte: -128..127")
    print("  setspeed/calibrate use signed bytes: -128..127")
    print("  sendmap rows/cols use unsigned 16-bit integers")
    print("  sendmap cell values are bytes: 0..255")
    print()


def interactive_loop(sock, host, port):
    print("Connected to EV3 at {}:{}".format(host, port))
    print_help()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd == "handshake":
                packet = build_handshake()

            elif cmd == "calibrate":
                if len(parts) != 3:
                    print("Usage: calibrate LEFT_TRIM RIGHT_TRIM")
                    continue

                left_trim = int(parts[1])
                right_trim = int(parts[2])

                packet = build_calibrate(left_trim, right_trim)

            elif cmd == "goto":
                if len(parts) != 3:
                    print("Usage: goto X Y")
                    continue

                x = int(parts[1])
                y = int(parts[2])

                packet = build_goto(x, y)

            elif cmd == "possync":
                if len(parts) not in (3, 4):
                    print("Usage: possync X Y")
                    print("   or: possync X Y HEADING_TENTHS")
                    continue

                x = int(parts[1])
                y = int(parts[2])

                # Optional heading is accepted for convenience but ignored.
                packet = build_possync(x, y)

            elif cmd == "turn":
                if len(parts) != 3:
                    print("Usage: turn ANGLE SPEED")
                    continue

                angle = int(parts[1])
                speed = int(parts[2])

                packet = build_turn(angle, speed)

            elif cmd == "setspeed":
                if len(parts) != 3:
                    print("Usage: setspeed LEFT RIGHT")
                    continue

                left = int(parts[1])
                right = int(parts[2])

                packet = build_setspeed(left, right)

            elif cmd == "sendmap":
                if len(parts) < 4:
                    print("Usage: sendmap ROWS COLS CELL1 CELL2 ...")
                    continue

                rows = int(parts[1])
                cols = int(parts[2])
                cells = [int(value) for value in parts[3:]]

                packet = build_sendmap(rows, cols, cells)

            elif cmd == "sendmap_fill":
                if len(parts) != 4:
                    print("Usage: sendmap_fill ROWS COLS VALUE")
                    continue

                rows = int(parts[1])
                cols = int(parts[2])
                value = int(parts[3])

                packet = build_sendmap_fill(rows, cols, value)

            elif cmd == "finish":
                packet = build_finish()

            elif cmd == "help":
                print_help()
                continue

            elif cmd in ("quit", "exit"):
                break

            else:
                print("Unknown command. Type 'help'.")
                continue

            if not send_command(sock, packet):
                break

        except ValueError as exc:
            print("Invalid input:", exc)


def main():
    host = HOST
    port = PORT

    if len(sys.argv) >= 2:
        host = sys.argv[1]

    if len(sys.argv) >= 3:
        port = int(sys.argv[2])

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            interactive_loop(sock, host, port)

    except OSError as exc:
        print("Connection error:", exc)


if __name__ == "__main__":
    main()