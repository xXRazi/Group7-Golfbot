from src.ev3.modules.state_machine.state_machine import StateMachine


def main():
    sm = StateMachine()
    while True:
        sm.update()


if __name__ == "__main__":
    main()
