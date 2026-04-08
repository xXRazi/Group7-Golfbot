#!/usr/bin/env python3
import socket

from motor_control import MotorController
from command_handling import CommandHandler

HOST = "0.0.0.0"
PORT = 5000


def loop(conn, command_handler):
    while True:
        data_buf = conn.recv(16)

        if not data_buf:
            print("Client disconnected")
            command_handler.motor_controller.stop()
            break

        if not command_handler.handle_command(data_buf):
            break

        print("Received raw bytes:", list(data_buf))

        response = b"EV3 got command\n"
        conn.sendall(response)


def main():
    motor_controller = MotorController()
    command_handler = CommandHandler(motor_controller)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)

        print("Listening on {}:{}...".format(HOST, PORT))

        conn, addr = server.accept()
        with conn:
            print("Connected by {}".format(addr))
            loop(conn, command_handler)


if __name__ == "__main__":
    main()