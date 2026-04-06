import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from orion_arm.orion_kinematics import OrionIK 

class OrionIKNode(Node):
    def __init__(self):
        super().__init__('orion_ik_node')
        
        # 1. Instanciamos el Cerebro Matemático
        self.ik_solver = OrionIK()
        
        # 2. ESCUCHA (Input): Comandos Cartesianos
        # Espera recibir [x, y, z, pitch]
        self.sub_cartesian = self.create_subscription(
            Float32MultiArray,
            '/brazo/cmd_cartesian',
            self.callback_cartesian,
            10
        )
        
        # 3. HABLA (Output): Comandos para los Motores (Joints)
        # Publica directamente al tópico que escucha el Driver
        self.pub_joints = self.create_publisher(
            Float32MultiArray,
            '/brazo/set_joint_angles',
            10
        )
        
        self.get_logger().info("✅ Nodo IK Listo: Escuchando en '/brazo/cmd_cartesian'...")

    def callback_cartesian(self, msg):
        """
        Recibe una coordenada (X, Y, Z, Pitch) y, si es válida,
        le ordena al Driver que mueva el robot.
        """
        data = msg.data
        if len(data) < 4:
            self.get_logger().warn("⚠️ Datos incompletos. Se requieren [x, y, z, pitch]")
            return

        x, y, z, pitch = data[0], data[1], data[2], data[3]
        
        # Usamos la librería matemática para resolver
        # Esto NO bloquea el driver ni la GUI, porque es un proceso aparte.
        resultado = self.ik_solver.calcular_ik(x, y, z, pitch)
        
        if resultado:
            t1, t2, t3, t4 = resultado
            
            # Preparamos el mensaje para el Driver
            msg_joints = Float32MultiArray()
            msg_joints.data = [float(t1), float(t2), float(t3), float(t4)]
            
            # Publicamos la orden
            self.pub_joints.publish(msg_joints)
            # self.get_logger().info(f"Moviendo a: {t1:.1f}, {t2:.1f}, {t3:.1f}, {t4:.1f}")
            
        else:
            self.get_logger().warn(f"❌ Punto inalcanzable: X={x:.1f}, Y={y:.1f}, Z={z:.1f}")

def main(args=None):
    rclpy.init(args=args)
    node = OrionIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()