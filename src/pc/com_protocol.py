#!/usr/bin/env python3
import socket
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
    """Convert signed int (-128..127) to one byte (0..255)."""
    if not (-128 <= value <= 127):
        raise ValueError("value must be between -128 and 127")
    return value % 256


def validate_unsigned_byte(value, name):
    if not (0 <= value <= 255):
        raise ValueError("{} must be between 0 and 255".format(name))
    return value


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
        print("Sent bytes:", list(packet))
        return recv_response(sock)
    except OSError as exc:
        print("Send error:", exc)
        return False


def build_handshake():
    return bytes([HANDSHAKE])


def build_calibrate(left_trim, right_trim):
    return bytes([
        CALIBRATE,
        encode_signed_byte(left_trim),
        encode_signed_byte(right_trim),
    ])


def build_goto(x, y):
    x = validate_unsigned_byte(x, "x")
    y = validate_unsigned_byte(y, "y")
    return bytes([GOTO, x, y])


def build_possync(x, y):
    """
    Matches the current robot code:
    POSSYNC = [CMD][X][Y]
    """
    x = validate_unsigned_byte(x, "x")
    y = validate_unsigned_byte(y, "y")
    return bytes([POSSYNC, x, y])


def build_turn(angle, speed):
    return bytes([
        TURN,
        encode_signed_byte(angle),
        encode_signed_byte(speed),
    ])


def build_setspeed(left, right):
    return bytes([
        SETSPEED,
        encode_signed_byte(left),
        encode_signed_byte(right),
    ])


def build_sendmap(rows, cols, cells):
    rows = validate_unsigned_byte(rows, "rows")
    cols = validate_unsigned_byte(cols, "cols")

    expected = rows * cols
    if len(cells) != expected:
        raise ValueError(
            "sendmap needs exactly {} cell values for a {}x{} map".format(
                expected, rows, cols
            )
        )

    validated_cells = [validate_unsigned_byte(cell, "cell") for cell in cells]
    return bytes([SENDMAP, rows, cols] + validated_cells)


def build_finish():
    return bytes([FINISH])


def print_help():
    print()
    print("Commands:")
    print("  handshake")
    print("  calibrate LEFT_TRIM RIGHT_TRIM")
    print("  goto X Y")
    print("  possync X Y")
    print("  turn ANGLE SPEED")
    print("  setspeed LEFT RIGHT")
    print("  sendmap ROWS COLS CELL1 CELL2 ...")
    print("  finish")
    print("  help")
    print("  quit")
    print()
    print("Notes:")
    print("  calibrate/turn/setspeed use signed values: -128..127")
    print("  goto/possync use unsigned values: 0..255")
    print("  sendmap needs exactly ROWS*COLS cell values")
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
                if len(parts) != 3:
                    print("Usage: possync X Y")
                    continue
                x = int(parts[1])
                y = int(parts[2])
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