#!/usr/bin/env python3
import math
import sys
import time

from piper_sdk import C_PiperInterface


JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']


def read_loop(can_port: str, rate_hz: float = 10.0):
    piper = C_PiperInterface(can_name=can_port)
    piper.ConnectPort()

    print(f'Conectado a {can_port}. Leyendo posiciones (sin energizar). Ctrl-C para salir.\n')
    period = 1.0 / rate_hz

    while True:
        try:
            joints = piper.GetArmJointMsgs()
            gripper = piper.GetArmGripperMsgs()
        except Exception as exc:
            print(f'Error leyendo SDK: {exc}')
            time.sleep(period)
            continue

        factor = math.pi / 180000.0  # milli-degrees → radians
        positions = [
            joints.joint_state.joint_1 * factor,
            joints.joint_state.joint_2 * factor,
            joints.joint_state.joint_3 * factor,
            joints.joint_state.joint_4 * factor,
            joints.joint_state.joint_5 * factor,
            joints.joint_state.joint_6 * factor,
            gripper.gripper_state.grippers_angle / 1_000_000,
        ]

        lines = ['--- Piper joint states ---']
        for name, pos in zip(JOINT_NAMES, positions):
            lines.append(f'  {name:<8} {pos:+.4f} rad')
        lines.append('')
        # Move cursor up to overwrite previous block
        print('\n'.join(lines))
        up = len(lines)
        sys.stdout.write(f'\033[{up}A')
        sys.stdout.flush()

        time.sleep(period)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Lee posiciones del brazo Piper sin energizar motores.')
    parser.add_argument('--can-port', default='can2', help='Interfaz CAN (default: can2)')
    parser.add_argument('--rate', type=float, default=10.0, help='Hz de lectura (default: 10)')
    args = parser.parse_args()

    try:
        read_loop(args.can_port, args.rate)
    except KeyboardInterrupt:
        print('\nDetenido.')


if __name__ == '__main__':
    main()
