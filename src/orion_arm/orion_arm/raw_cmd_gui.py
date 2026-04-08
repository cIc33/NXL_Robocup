#!/usr/bin/env python3
import tkinter as tk

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int8MultiArray


JOINT_NAMES = ["E1", "E2", "E3", "E4"]
CMD_VALUE = 10


class RawCmdGui(Node):
    def __init__(self):
        super().__init__('raw_cmd_gui')

        qos_cmds = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.publisher = self.create_publisher(Int8MultiArray, '/brazo/raw_cmd', qos_cmds)
        self.current_cmd = [0, 0, 0, 0]
        self.manual_active = False
        self.create_timer(0.05, self.publish_current_cmd)

    def set_joint_cmd(self, joint_idx, value):
        self.current_cmd[joint_idx] = int(value)

    def stop_all(self):
        self.current_cmd = [0, 0, 0, 0]
        self.publish_msg(self.current_cmd)
        self.manual_active = False

    def publish_msg(self, values):
        msg = Int8MultiArray()
        msg.data = [int(v) for v in values]
        self.publisher.publish(msg)

    def publish_current_cmd(self):
        is_moving = any(value != 0 for value in self.current_cmd)

        if is_moving:
            self.publish_msg(self.current_cmd)
            self.manual_active = True
        elif self.manual_active:
            self.publish_msg([0, 0, 0, 0])
            self.manual_active = False


class RawCmdWindow:
    def __init__(self, node: RawCmdGui):
        self.node = node
        self.root = tk.Tk()
        self.root.title('Orion Raw Cmd GUI')
        self.root.geometry('460x320')
        self.root.configure(bg='#16181d')
        self.value_labels = []

        title = tk.Label(
            self.root,
            text='Control Manual /brazo/raw_cmd',
            bg='#16181d',
            fg='white',
            font=('Segoe UI', 16, 'bold'),
        )
        title.pack(pady=(16, 12))

        content = tk.Frame(self.root, bg='#16181d')
        content.pack(fill='both', expand=True, padx=16, pady=8)

        for joint_idx, joint_name in enumerate(JOINT_NAMES):
            self._build_joint_controls(content, joint_idx, joint_name)

        stop_button = tk.Button(
            self.root,
            text='STOP',
            command=self.on_close_request,
            bg='#c0392b',
            fg='white',
            activebackground='#e74c3c',
            activeforeground='white',
            font=('Segoe UI', 13, 'bold'),
            bd=0,
            padx=12,
            pady=8,
        )
        stop_button.pack(fill='x', padx=16, pady=(8, 16))

        self.root.protocol('WM_DELETE_WINDOW', self.on_close_request)
        self.root.after(20, self.spin_ros)

    def _build_joint_controls(self, parent, joint_idx, joint_name):
        row = tk.Frame(parent, bg='#20232a')
        row.pack(fill='x', pady=6)

        name_label = tk.Label(
            row,
            text=joint_name,
            width=6,
            bg='#20232a',
            fg='white',
            font=('Segoe UI', 12, 'bold'),
        )
        name_label.pack(side='left', padx=(12, 8), pady=10)

        negative_button = self._make_hold_button(row, '-10', joint_idx, -CMD_VALUE)
        negative_button.pack(side='left', padx=4)

        positive_button = self._make_hold_button(row, '+10', joint_idx, CMD_VALUE)
        positive_button.pack(side='left', padx=4)

        value_label = tk.Label(
            row,
            text='0',
            width=4,
            bg='#20232a',
            fg='#5dade2',
            font=('Consolas', 12, 'bold'),
        )
        value_label.pack(side='right', padx=(8, 12))
        self.value_labels.append(value_label)

    def _make_hold_button(self, parent, text, joint_idx, value):
        button = tk.Button(
            parent,
            text=text,
            bg='#2d3436',
            fg='white',
            activebackground='#5dade2',
            activeforeground='black',
            font=('Segoe UI', 11, 'bold'),
            width=8,
            bd=0,
            pady=8,
        )
        button.bind('<ButtonPress-1>', lambda _event: self.on_button_press(joint_idx, value))
        button.bind('<ButtonRelease-1>', lambda _event: self.on_button_release(joint_idx))
        button.bind('<Leave>', lambda _event: self.on_button_release(joint_idx))
        return button

    def on_button_press(self, joint_idx, value):
        self.node.set_joint_cmd(joint_idx, value)
        self.value_labels[joint_idx].config(text=str(value))

    def on_button_release(self, joint_idx):
        self.node.set_joint_cmd(joint_idx, 0)
        self.value_labels[joint_idx].config(text='0')

    def on_close_request(self):
        self.node.stop_all()
        self.root.after(50, self.root.destroy)

    def spin_ros(self):
        if not self.root.winfo_exists():
            return
        rclpy.spin_once(self.node, timeout_sec=0.0)
        self.root.after(20, self.spin_ros)

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = RawCmdGui()
    window = RawCmdWindow(node)

    try:
        window.run()
    finally:
        node.stop_all()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
