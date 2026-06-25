#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32


class PiperVelocityGuiNode(Node):
    def __init__(self):
        super().__init__('piper_velocity_gui')

        self.declare_parameter('topic_name', '/piper/test_velocity_cmd')
        self.declare_parameter('switch_mode_topic', '/piper/switch_mode')
        self.declare_parameter('publish_rate', 20.0)

        self.topic_name = str(self.get_parameter('topic_name').value)
        self.switch_mode_topic = str(self.get_parameter('switch_mode_topic').value)
        self.publish_rate = max(1.0, float(self.get_parameter('publish_rate').value))

        self.publisher = self.create_publisher(Float32MultiArray, self.topic_name, 10)
        self.switch_mode_publisher = self.create_publisher(Int32, self.switch_mode_topic, 10)

        self.get_logger().info(
            f'Piper velocity GUI publishing to {self.topic_name} at {self.publish_rate:.1f} Hz.'
        )

    def publish_velocities(self, velocities: list):
        msg = Float32MultiArray()
        msg.data = velocities
        self.publisher.publish(msg)

    def publish_switch_mode(self, cartesian: bool):
        msg = Int32()
        msg.data = 1 if cartesian else 0
        self.switch_mode_publisher.publish(msg)


class VelocityGui:
    def __init__(self, node: PiperVelocityGuiNode):
        self.node = node
        self.root = tk.Tk()
        self.root.title('Piper Velocity GUI')
        self.root.protocol('WM_DELETE_WINDOW', self.close)

        self.command_vars = [tk.DoubleVar(value=0.0) for _ in range(6)]
        self.gripper_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value='Publishing Joint Mode')
        self.is_cartesian = tk.BooleanVar(value=False)
        self.closed = False

        self._build_layout()
        
        # Calcular el intervalo en milisegundos basado en los Hz del nodo
        self.update_interval_ms = int(1000.0 / self.node.publish_rate)
        # Iniciar el bucle de ROS dentro de Tkinter
        self._ros_loop()

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky='nsew')
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        title = ttk.Label(main, text='Piper Velocity Command', font=('TkDefaultFont', 14, 'bold'))
        title.grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 10))

        ttk.Label(main, text='Topic:').grid(row=1, column=0, sticky='w')
        ttk.Label(main, text=self.node.topic_name).grid(row=1, column=1, columnspan=2, sticky='w')
        
        # Switch de Modo (Fila 2)
        ttk.Label(main, text='Switch:').grid(row=2, column=0, sticky='w')
        self.mode_checkbox = ttk.Checkbutton(
            main, 
            text='Cartesian Mode (TCP)', 
            variable=self.is_cartesian,
            command=self._update_mode_status
        )
        self.mode_checkbox.grid(row=2, column=1, columnspan=2, sticky='w', pady=3)

        labels = [
            'Cmd 1 / Joint 1 / X TCP',
            'Cmd 2 / Joint 2 / Y TCP',
            'Cmd 3 / Joint 3 / Z TCP',
            'Cmd 4 / Joint 4 / Roll TCP',
            'Cmd 5 / Joint 5 / Pitch TCP',
            'Cmd 6 / Joint 6 / Yaw TCP',
        ]
        for index, label in enumerate(labels):
            row = index + 4
            ttk.Label(main, text=label).grid(row=row, column=0, sticky='w')
            scale = ttk.Scale(
                main,
                from_=-100.0,
                to=100.0,
                orient='horizontal',
                variable=self.command_vars[index],
                command=lambda _value, idx=index: self._update_value_label(idx),
            )
            scale.grid(row=row, column=1, sticky='ew', padx=8, pady=3)
            value_label = ttk.Label(main, text='0.0', width=7)
            value_label.grid(row=row, column=2, sticky='e')
            setattr(self, f'command_label_{index}', value_label)

        gripper_row = 10
        ttk.Label(main, text='Gripper').grid(row=gripper_row, column=0, sticky='w')
        gripper_scale = ttk.Scale(
            main,
            from_=-1.0,
            to=1.0,
            orient='horizontal',
            variable=self.gripper_var,
            command=lambda _value: self._update_gripper_label(),
        )
        gripper_scale.grid(row=gripper_row, column=1, sticky='ew', padx=8, pady=3)
        self.gripper_label = ttk.Label(main, text='0.00', width=7)
        self.gripper_label.grid(row=gripper_row, column=2, sticky='e')

        controls = ttk.Frame(main)
        controls.grid(row=11, column=0, columnspan=3, sticky='ew', pady=(12, 6))
        ttk.Button(controls, text='Detener / Ceros', command=self.zero_commands).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(controls, text='Cerrar', command=self.close).grid(row=0, column=1)

        ttk.Label(main, textvariable=self.status_var).grid(row=12, column=0, columnspan=3, sticky='w')
        main.columnconfigure(1, weight=1)

    def _update_value_label(self, index: int):
        label = getattr(self, f'command_label_{index}')
        label.configure(text=f'{self.command_vars[index].get():.1f}')

    def _update_gripper_label(self):
        self.gripper_label.configure(text=f'{self.gripper_var.get():.2f}')

    def _update_mode_status(self):
        cartesian = self.is_cartesian.get()
        self.status_var.set('Publishing Cartesian mode' if cartesian else 'Publishing Joint mode')
        self.node.publish_switch_mode(cartesian)

    def zero_commands(self):
        for var in self.command_vars:
            var.set(0.0)
        self.gripper_var.set(0.0)
        for index in range(len(self.command_vars)):
            self._update_value_label(index)
        self._update_gripper_label()

    def _ros_loop(self):
        """Bucle periódico que procesa ROS 2 y publica los datos actuales."""
        if self.closed:
            return

        # 1. Hacer un spin_once para que el nodo procese eventos si es necesario
        rclpy.spin_once(self.node, timeout_sec=0)

        # 2. Recopilar datos de la GUI: 6 articulaciones/TCP + 1 gripper
        current_velocities = [var.get() for var in self.command_vars]
        current_velocities.append(self.gripper_var.get())

        # 3. Publicar mediante el nodo
        self.node.publish_velocities(current_velocities)

        # 4. Programar la siguiente ejecución para cumplir los Hz configurados
        self.root.after(self.update_interval_ms, self._ros_loop)

    def run(self):
        self.root.mainloop()

    def close(self):
        self.zero_commands()
        self.closed = True
        # Publicar una última ráfaga de ceros antes de destruir todo
        current_velocities = [0.0] * 7
        self.node.publish_velocities(current_velocities)
        self.root.destroy()


def main(args=None):
    rclpy.init(args=args)
    node = PiperVelocityGuiNode()
    gui = VelocityGui(node)
    try:
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()