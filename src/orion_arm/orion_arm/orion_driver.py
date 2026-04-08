import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8MultiArray, Float32MultiArray, Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time


class OrionDriver(Node):
    def __init__(self):
        super().__init__('orion_driver')

        qos_cmds = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_sensors = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.tx_seq = 1
        self.KP = 1.5
        self.MAX_CMD = 10
        self.TOLERANCIA = 1.0
        self.DIR_MULTS = {"E1": 1, "E2": 1, "E3": -1, "E4": 1}
        self.mode = "IDLE"
        self.current_pos = [0.0, 0.0, 0.0, 0.0]
        self.target_pos = [None, None, None, None]
        self.raw_cmd = [0, 0, 0, 0]
        self.last_sensor_time = 0.0
        self.emergency_stop = False

        self.declare_parameter(
            'joint_limits',
            [0.0, 0.0, -120.0, 120.0, -145.0, 140.0, -100.0, 100.0]
        )
        self._load_joint_limits()

        self.create_subscription(
            Float32MultiArray,
            '/brazo/angulos_sensores',
            self.cb_sensors,
            qos_sensors
        )
        self.create_subscription(
            Int8MultiArray,
            '/brazo/raw_cmd',
            self.cb_raw,
            10
        )
        self.create_subscription(
            Float32MultiArray,
            '/brazo/set_joint_angles',
            self.cb_pos,
            10
        )
        self.create_subscription(
            Bool,
            '/brazo/emergency_stop',
            self.cb_stop,
            10
        )

        # Ahora publica Int8MultiArray — sin empaquetado, valores directos
        self.pub_joint_cmds = self.create_publisher(
            Int8MultiArray,
            '/brazo/joint_cmds',
            qos_cmds
        )

        self.create_timer(0.05, self.control_loop)
        self.get_logger().info('Orion Driver listo → /brazo/joint_cmds como Int8MultiArray')

    def cb_sensors(self, msg):
        data = list(msg.data)
        if len(data) >= 4:
            self.current_pos = data[:4]
            self.last_sensor_time = time.time()

    def cb_raw(self, msg):
        if self.emergency_stop:
            return
        data = list(msg.data)
        if len(data) < 4:
            return
        self.mode = "RAW"
        self.raw_cmd = [int(x) for x in data[:4]]
        self.target_pos = [None] * 4
        self.get_logger().info(f'RAW recibido: {self.raw_cmd}')

    def cb_pos(self, msg):
        if self.emergency_stop:
            return
        data = list(msg.data)
        if len(data) < 4:
            return
        self.mode = "POS"
        self.target_pos = [float(x) for x in data[:4]]
        self.raw_cmd = [0, 0, 0, 0]
        self.get_logger().info(f'POS recibido: {self.target_pos}')

    def cb_stop(self, msg):
        self.emergency_stop = bool(msg.data)
        if self.emergency_stop:
            self.mode = "IDLE"
            self.get_logger().warning('EMERGENCY STOP activado')
            self.publish_joint_cmds(0, 0, 0, 0)
        else:
            self.get_logger().info('EMERGENCY STOP liberado')

    def control_loop(self):
        final_cmd = [0, 0, 0, 0]

        if self.mode == "RAW":
            final_cmd = list(self.raw_cmd)

        elif self.mode == "POS":
            if time.time() - self.last_sensor_time > 0.5:
                self.get_logger().warning('Sin sensores recientes: enviando 0,0,0,0')
                self.publish_joint_cmds(0, 0, 0, 0)
                return

            for i in range(4):
                if self.target_pos[i] is not None:
                    target = self.clamp_to_limits(self.target_pos[i], self.JOINT_LIMITS[i], i)
                    err = self.get_error(target, self.current_pos[i], i)
                    if abs(err) > self.TOLERANCIA:
                        val = int(err * self.KP)
                        val = max(-self.MAX_CMD, min(self.MAX_CMD, val))
                        final_cmd[i] = val * self.DIR_MULTS[f"E{i+1}"] * (-1)

        self.publish_joint_cmds(
            final_cmd[0], final_cmd[1],
            final_cmd[2], final_cmd[3]
        )

    def publish_joint_cmds(self, e1, e2, e3, e4):
        msg = Int8MultiArray()
        # data = [seq, e1, e2, e3, e4] — valores directos, sin empaquetar
        msg.data = [self.tx_seq, int(e1), int(e2), int(e3), int(e4)]
        self.pub_joint_cmds.publish(msg)

        self.get_logger().info(
            f'mode={self.mode} seq={self.tx_seq} e1={e1} e2={e2} e3={e3} e4={e4}'
        )

        self.tx_seq += 1
        if self.tx_seq > 127:
            self.tx_seq = 1

    def _load_joint_limits(self):
        jl = self.get_parameter('joint_limits').value
        self.JOINT_LIMITS = [
            (jl[0], jl[1]),
            (jl[2], jl[3]),
            (jl[4], jl[5]),
            (jl[6], jl[7])
        ]

    def get_error(self, target, current, axis_idx):
        if axis_idx == 0:
            return ((target - current + 180) % 360) - 180
        return target - current

    def clamp_to_limits(self, angle, limits, axis_idx):
        if axis_idx == 0:
            return angle
        return max(limits[0], min(limits[1], angle))


def main(args=None):
    rclpy.init(args=args)
    node = OrionDriver()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()