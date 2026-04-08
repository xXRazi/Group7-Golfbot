#!/usr/bin/env python3
import socket
import sys


HOST = "ev3dev.local"   # Change to your EV3 IP
PORT = 5000


ERROR = 0x0
HANDSHAKE = 0xA
GOTO = 0xB
POSSYNC = 0xC
TURN = 0xD
SETSPEED = 0xE
FINISH = 0xF


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


def build_goto(x, y):
    if not (0 <= x <= 255 and 0 <= y <= 255):
        raise ValueError("x and y must be between 0 and 255")
    return bytes([GOTO, x, y])


def build_possync(x, y, heading=0):
    if not (0 <= x <= 255 and 0 <= y <= 255 and 0 <= heading <= 255):
        raise ValueError("x, y and heading must be between 0 and 255")
    return bytes([POSSYNC, x, y, heading])


def build_turn(angle, speed):
    if not (0 <= angle <= 255 and 0 <= speed <= 255):
        raise ValueError("angle and speed must be between 0 and 255")
    return bytes([TURN, angle, speed])


def build_setspeed(left, right):
    if not (0 <= left <= 255 and 0 <= right <= 255):
        raise ValueError("left and right must be between 0 and 255")
    return bytes([SETSPEED, left, right])


def build_finish():
    return bytes([FINISH])


def print_help():
    print()
    print("Commands:")
    print("  handshake")
    print("  goto X Y")
    print("  possync X Y HEADING")
    print("  turn ANGLE SPEED")
    print("  setspeed LEFT RIGHT")
    print("  finish")
    print("  help")
    print("  quit")
    print()


def interactive_loop(sock):
    print("Connected to EV3 at {}:{}".format(HOST, PORT))
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

            elif cmd == "goto":
                if len(parts) != 3:
                    print("Usage: goto X Y")
                    continue
                x = int(parts[1])
                y = int(parts[2])
                packet = build_goto(x, y)

            elif cmd == "possync":
                if len(parts) != 4:
                    print("Usage: possync X Y HEADING")
                    continue
                x = int(parts[1])
                y = int(parts[2])
                heading = int(parts[3])
                packet = build_possync(x, y, heading)

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
            interactive_loop(sock)
    except OSError as exc:
        print("Connection error:", exc)


if __name__ == "__main__":
    main()