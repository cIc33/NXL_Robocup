
import rclpy
from rclpy.node import Node

from sensor_msgs.msg   import Image
from std_msgs.msg      import Bool
from geometry_msgs.msg import Point

import cv2
import numpy as np
from cv_bridge import CvBridge

# ── Colores BGR ─────────────────────────────────────────────────────────────────
VERDE    = (  0, 220,   0)
ROJO_BGR = (  0,   0, 220)
AMARILLO = (  0, 200, 255)
NEGRO    = (  0,   0,   0)


class ParoDetectorNode(Node):

    def __init__(self):
        super().__init__('paro_detector')

        # ── Parámetros ──────────────────────────────────────────────────────────
        self.declare_parameter('cam_index',     0)
        self.declare_parameter('ancho',         1280)
        self.declare_parameter('alto',          720)
        self.declare_parameter('dp',            1.2)
        self.declare_parameter('min_dist',      60)
        self.declare_parameter('param1',        105)
        self.declare_parameter('param2',        80)
        self.declare_parameter('min_radio',     20)
        self.declare_parameter('max_radio',     250)
        self.declare_parameter('blur_k',        5)
        self.declare_parameter('cobertura_min', 0.35)

        # ── Visión ──────────────────────────────────────────────────────────────
        self.bridge      = CvBridge()
        self.clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.kernel_morf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Rojo en HSV está partido en dos rangos (cerca de 0° y 180°)
        self.rojo_bajo1 = np.array([  0,  80,  60], dtype=np.uint8)
        self.rojo_alto1 = np.array([ 10, 255, 255], dtype=np.uint8)
        self.rojo_bajo2 = np.array([165,  80,  60], dtype=np.uint8)
        self.rojo_alto2 = np.array([180, 255, 255], dtype=np.uint8)

        # ── Publishers ──────────────────────────────────────────────────────────
        self.pub_imagen    = self.create_publisher(Image,  '/paro/imagen',    10)
        self.pub_detectado = self.create_publisher(Bool,   '/paro/detectado', 10)
        self.pub_centro    = self.create_publisher(Point,  '/paro/centro',    10)

        # ── Webcam ──────────────────────────────────────────────────────────────
        cam_idx = self.get_parameter('cam_index').value
        ancho   = self.get_parameter('ancho').value
        alto    = self.get_parameter('alto').value

        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened():
            self.get_logger().fatal(f"No se pudo abrir la cámara {cam_idx}")
            raise RuntimeError(f"Cámara {cam_idx} no disponible")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ancho)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
        self.cap.set(cv2.CAP_PROP_FPS,          30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        # Timer a 30 Hz — cada tick lee un frame y procesa
        self.timer = self.create_timer(1.0 / 30.0, self._tick)

        self.get_logger().info(f"Detector iniciado — cámara {cam_idx}  {ancho}x{alto}")

    # ── Timer callback ───────────────────────────────────────────────────────────

    def _tick(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("Frame no disponible, reintentando…")
            return
        self._procesar(frame)

    # ── Pipeline de visión ───────────────────────────────────────────────────────

    def _mascara_roja(self, hsv: np.ndarray) -> np.ndarray:
        m1 = cv2.inRange(hsv, self.rojo_bajo1, self.rojo_alto1)
        m2 = cv2.inRange(hsv, self.rojo_bajo2, self.rojo_alto2)
        return cv2.morphologyEx(cv2.bitwise_or(m1, m2),
                                cv2.MORPH_CLOSE, self.kernel_morf)

    def _preprocesar(self, frame: np.ndarray) -> np.ndarray:
        bk   = self.get_parameter('blur_k').value
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gris = self.clahe.apply(gris)
        return cv2.GaussianBlur(gris, (bk, bk), 0)

    def _es_rojo(self, mascara: np.ndarray, x: int, y: int, r: int) -> bool:
        h, w  = mascara.shape
        disco = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(disco, (x, y), r, 255, -1)
        px_disco = cv2.countNonZero(disco)
        if px_disco == 0:
            return False
        px_rojo = cv2.countNonZero(cv2.bitwise_and(mascara, mascara, mask=disco))
        return (px_rojo / px_disco) >= self.get_parameter('cobertura_min').value

    def _procesar(self, frame: np.ndarray):
        hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mascara = self._mascara_roja(hsv)
        suave   = self._preprocesar(frame)

        candidatos = cv2.HoughCircles(
            suave,
            cv2.HOUGH_GRADIENT,
            dp        = self.get_parameter('dp').value,
            minDist   = self.get_parameter('min_dist').value,
            param1    = self.get_parameter('param1').value,
            param2    = self.get_parameter('param2').value,
            minRadius = self.get_parameter('min_radio').value,
            maxRadius = self.get_parameter('max_radio').value,
        )

        vis         = frame.copy()
        confirmados = []

        if candidatos is not None:
            for x, y, r in np.round(candidatos[0]).astype(int):
                if not self._es_rojo(mascara, x, y, r):
                    continue
                confirmados.append((x, y, r))
                cv2.circle(vis, (x, y), r, VERDE,    2,  cv2.LINE_AA)
                cv2.circle(vis, (x, y), 4, ROJO_BGR, -1, cv2.LINE_AA)
                label = f"r={r}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                lx = max(x - r, 2)
                ly = max(y - r - 4, th + 2)
                cv2.rectangle(vis, (lx-2, ly-th-2), (lx+tw+2, ly+2), NEGRO, -1)
                cv2.putText(vis, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMARILLO, 1, cv2.LINE_AA)

        n = len(confirmados)
        cv2.putText(vis, f"Paros: {n}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, VERDE, 2, cv2.LINE_AA)

        # ── Publicar imagen anotada ──────────────────────────────────────────────
        try:
            self.pub_imagen.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
        except Exception as e:
            self.get_logger().error(f"cv_bridge: {e}")

        # ── Publicar detección ───────────────────────────────────────────────────
        msg_bool      = Bool()
        msg_bool.data = n > 0
        self.pub_detectado.publish(msg_bool)

        # ── Publicar centro (x, y px ; z = radio px) ────────────────────────────
        if confirmados:
            x, y, r  = confirmados[0]
            msg_pt   = Point()
            msg_pt.x = float(x)
            msg_pt.y = float(y)
            msg_pt.z = float(r)
            self.pub_centro.publish(msg_pt)

    # ── Cleanup ──────────────────────────────────────────────────────────────────

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            self.get_logger().info("Cámara liberada.")
        super().destroy_node()


# ── Entrypoint ───────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    try:
        nodo = ParoDetectorNode()
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"[ERROR] {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()