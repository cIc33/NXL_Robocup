import time
from collections import deque
from rclpy.qos import QoSProfile, ReliabilityPolicy
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.lifecycle import LifecycleNode, Publisher, State, TransitionCallbackReturn
from sensor_msgs.msg import Image
from ultralytics import YOLO


# ===========================================================================
# Constantes de configuración
# ===========================================================================

# ── YOLO ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH  = "/home/angel/NXL_Robocup/src/nixito_perception/modelos/Robocup_NXL.pt"
YOLO_MIN_PERIOD  = 0.05
YOLO_CONF        = 0.5
YOLO_IMGSZ       = 480


QR_MODEL_DIR      = "/home/angel/NXL_Robocup/src/nixito_perception/drivers/qr_models"
QR_HOLD_SECS      = 0.1
QR_MIN_AREA       = 100
QR_MAX_RATIO      = 0.50
QR_MIN_PERIOD     = 0.05   # s → ~20 fps máximo para detección activa
QR_DETECT_WIDTH   = 640    # El modelo WeChat trabaja bien a esta resolución

# ── Detección de movimiento ──────────────────────────────────────────────────
MOV_HISTORY        = 500
MOV_VAR_THRESHOLD  = 8
MOV_LEARNING_RATE  = 0.002
MOV_DIFF_THRESHOLD = 6
MOV_MIN_AREA       = 500
MOV_GROUP_EPS      = 0.3
MOV_WARMUP_LEN     = 15

# ── Buffer de frames ─────────────────────────────────────────────────────────
FRAME_BUFFER_MAXLEN = 15

# ── Topics ROS ───────────────────────────────────────────────────────────────
TOPIC_INPUT_RGB = "/camera/camera/color/image_raw"

TOPIC_OUTPUTS = {
    "yolo":     ("vision/yolo",     10),
    "qr":       ("vision/qr",       10),
    "movement": ("vision/movement", 10),
}

TIMER_CHECK_SUBS = 2.0
TIMER_LOG_STATS  = 10.0


# ===========================================================================
# Nodo principal
# ===========================================================================

class VisionNode(LifecycleNode):

    def __init__(self):
        super().__init__("vision_node")

        self.bridge      = CvBridge()
        self.pubs:       dict[str, Publisher] = {}
        self.active_mode: str | None = None

        self.model        = None
        self.model_loaded = False
        self.qr_detector  = None   # cv2.wechat_qrcode_WeChatQRCode
        self.mog2         = None

        self.frame_buffer = deque(maxlen=FRAME_BUFFER_MAXLEN)
        self.kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=40, detectShadows=False
        )

        self.last_yolo_time = 0.0
        self._last_qr_time  = 0.0
        self.tamano_roi     = 150

        self._qr_last_seen  = 0.0
        self._qr_last_value = ""
        self._qr_last_pts   = None  # coords en el frame original (full-res)

        self._timer_check_subs = None
        self._timer_stats      = None

        self.get_logger().info("VisionNode creado (UNCONFIGURED)")

    # -----------------------------------------------------------------------
    # Ciclo de vida
    # -----------------------------------------------------------------------

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configurando nodo …")
        for mode, (topic, qos) in TOPIC_OUTPUTS.items():
            self.pubs[mode] = self.create_lifecycle_publisher(Image, topic, qos)
            self.get_logger().info(f"  Publicador listo: {topic}")
        self.sub_rgb = self.create_subscription(
            Image, TOPIC_INPUT_RGB, self._main_loop, 1
        )
        self._timer_check_subs = self.create_timer(TIMER_CHECK_SUBS, self._check_subscribers)
        self._timer_stats      = self.create_timer(TIMER_LOG_STATS,  self._log_stats)
        self.get_logger().info("Nodo configurado — listo para activar")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Nodo ACTIVO — esperando suscriptores …")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Nodo INACTIVO — liberando recursos pesados …")
        self._unload_models()
        return super().on_deactivate(state)

    # -----------------------------------------------------------------------
    # Modelos
    # -----------------------------------------------------------------------

    def _load_models_for_mode(self, mode: str) -> None:
        if mode == "yolo" and not self.model_loaded:
            self.get_logger().info("Cargando modelo YOLO …")
            t0 = time.time()
            self.model = YOLO(YOLO_MODEL_PATH)
            self.model_loaded = True
            self.get_logger().info(f"YOLO cargado en {time.time() - t0:.1f} s")

        elif mode == "qr" and self.qr_detector is None:
            self.get_logger().info("Inicializando WeChatQRCode …")
            t0 = time.time()
            try:
                # Con modelos CNN (detección robusta + super-resolución interna)
                d = f"{QR_MODEL_DIR}/detect.prototxt"
                dc = f"{QR_MODEL_DIR}/detect.caffemodel"
                s = f"{QR_MODEL_DIR}/sr.prototxt"
                sc = f"{QR_MODEL_DIR}/sr.caffemodel"
                self.qr_detector = cv2.wechat_qrcode_WeChatQRCode(d, dc, s, sc)
                self.get_logger().info(
                    f"WeChatQRCode cargado con modelos CNN en {time.time()-t0:.1f} s"
                )
            except Exception as e:
                # Fallback: sin modelos (funciona pero sin super-resolución)
                self.get_logger().warn(
                    f"No se pudieron cargar modelos WeChat ({e}). "
                    "Usando modo sin modelos — descarga los .prototxt/.caffemodel "
                    f"en {QR_MODEL_DIR} para mejor detección."
                )
                self.qr_detector = cv2.wechat_qrcode_WeChatQRCode()

        elif mode == "movement" and self.mog2 is None:
            self.get_logger().info("Inicializando sustractor MOG2 …")
            self.mog2 = cv2.createBackgroundSubtractorMOG2(
                history=MOV_HISTORY, varThreshold=MOV_VAR_THRESHOLD, detectShadows=False,
            )

    def _unload_models(self) -> None:
        if self.model_loaded:
            self.get_logger().info("Descargando modelo YOLO …")
            self.model = None
            self.model_loaded = False
        self.qr_detector = None
        self.mog2 = None

    # -----------------------------------------------------------------------
    # Suscriptores / stats
    # -----------------------------------------------------------------------

    def _check_subscribers(self) -> None:
        if not self._is_active():
            return
        for mode, pub in self.pubs.items():
            if pub.get_subscription_count() > 0:
                if self.active_mode != mode:
                    self.get_logger().info(f"Suscriptor detectado para modo: {mode}")
                    self._load_models_for_mode(mode)
                    self.active_mode = mode
                return
        if self.active_mode is not None:
            self.get_logger().info(f"Sin suscriptores — desactivando modo: {self.active_mode}")
            self.active_mode = None

    def _log_stats(self) -> None:
        if self.active_mode:
            self.get_logger().info(f"Modo activo: {self.active_mode}")
        else:
            self.get_logger().info("Idle — sin procesamiento activo")

    # -----------------------------------------------------------------------
    # Bucle principal
    # -----------------------------------------------------------------------

    def _main_loop(self, msg: Image) -> None:
        if not self._is_active() or self.active_mode is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        processors = {
            "yolo":     (self.model_loaded,            self._process_yolo),
            "qr":       (self.qr_detector is not None, self._process_qr),
            "movement": (self.mog2 is not None,        self._process_movement),
        }
        ready, process_fn = processors.get(self.active_mode, (False, None))
        if not ready or process_fn is None:
            return
        result  = process_fn(frame)
        out_msg = self.bridge.cv2_to_imgmsg(result, "bgr8")
        self.pubs[self.active_mode].publish(out_msg)

    # -----------------------------------------------------------------------
    # Pipeline YOLO
    # -----------------------------------------------------------------------

    def _process_yolo(self, frame: np.ndarray) -> np.ndarray:
        t0      = time.time()
        results = self.model(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False)[0]
        frame   = results.plot()
        cv2.putText(frame, f"YOLO Active | Latency: {(time.time()-t0)*1000:.1f} ms",
                    (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
        return frame

    # -----------------------------------------------------------------------
    # Pipeline QR — WeChatQRCode
    # -----------------------------------------------------------------------

    def _validate_qr(self, value: str, pts: np.ndarray, frame_shape: tuple) -> bool:
        """Filtra detecciones espurias (sin cambios respecto al original)."""
        if not value or not value.strip():
            return False
        pts_i        = np.int32(pts).reshape(-1, 2)
        x, y, bw, bh = cv2.boundingRect(pts_i)
        area         = bw * bh
        fh, fw       = frame_shape[:2]
        if area < QR_MIN_AREA or area / (fw * fh) > QR_MAX_RATIO:
            return False
        if bh == 0 or abs(bw / bh - 1.0) > 0.3:
            return False
        if value.count('\x00') / len(value) > 0.1:
            return False
        return True

    def _draw_qr_result(self, frame: np.ndarray, age: float, t0: float) -> None:
        """Dibuja el overlay QR persistente."""
        if age < QR_HOLD_SECS and self._qr_last_pts is not None:
            pts_i        = np.int32(self._qr_last_pts).reshape(-1, 2)
            x, y, bw, bh = cv2.boundingRect(pts_i)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            label = self._qr_last_value[:40] + ("…" if len(self._qr_last_value) > 40 else "")
            cv2.putText(frame, label, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.65, (255, 255, 255), 2)
        status = "QR Active" if age < QR_HOLD_SECS else "Scanning"
        cv2.putText(frame, f"{status} | {(time.time()-t0)*1000:.1f} ms",
                    (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 0, 255), 1)

    def _process_qr(self, frame: np.ndarray) -> np.ndarray:
        t0  = time.time()
        now = t0
        age = now - self._qr_last_seen

        # Rate limiting: si el hold está vigente y no pasó QR_MIN_PERIOD,
        # solo dibujamos el resultado cacheado.
        if age < QR_HOLD_SECS and (now - self._last_qr_time) < QR_MIN_PERIOD:
            self._draw_qr_result(frame, age, t0)
            return frame

        self._last_qr_time = now

        # Downscale a QR_DETECT_WIDTH con INTER_AREA (rápido para reducción).
        # WeChat internamente aplica super-resolución cuando el QR es pequeño,
        # por lo que no necesitamos las múltiples variantes del detector anterior.
        h, w  = frame.shape[:2]
        scale = QR_DETECT_WIDTH / w
        if scale < 1.0:
            small = cv2.resize(frame, (QR_DETECT_WIDTH, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame

        # detectAndDecode de WeChat devuelve (lista_de_textos, lista_de_puntos).
        # Si no detecta nada, ambas listas están vacías.
        try:
            texts, points = self.qr_detector.detectAndDecode(small)
        except cv2.error as e:
            self.get_logger().warn(f"WeChat detectAndDecode error: {e}")
            texts, points = [], []

        value, pts = "", None
        for text, pt in zip(texts, points):
            if not text:
                continue
            # pt viene como array (4,2); proyectar a coords del frame original
            pts_full = pt / scale
            if self._validate_qr(text, pts_full, frame.shape):
                value = text
                pts   = pts_full
                break   # tomar el primer QR válido

        if pts is not None:
            self._qr_last_value = value
            self._qr_last_pts   = pts
            self._qr_last_seen  = now

        age = now - self._qr_last_seen
        self._draw_qr_result(frame, age, t0)
        return frame

    # -----------------------------------------------------------------------
    # Pipeline Movimiento
    # -----------------------------------------------------------------------

    def _process_movement(self, frame: np.ndarray) -> np.ndarray:
        start_time = time.time()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)
        self.frame_buffer.append(gray_frame)

        if len(self.frame_buffer) < self.frame_buffer.maxlen:
            cv2.putText(frame, f"Warming up... {len(self.frame_buffer)}/{self.frame_buffer.maxlen}",
                        (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 2)
            return frame

        oldest_frame     = self.frame_buffer[0]
        temporal_delta   = cv2.absdiff(gray_frame, oldest_frame)
        _, combined_mask = cv2.threshold(temporal_delta, 10, 255, cv2.THRESH_BINARY)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,  self.kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        combined_mask = cv2.dilate(combined_mask, self.kernel, iterations=2)

        if self.tamano_roi > 0:
            alto_img, ancho_img = frame.shape[:2]
            cx, cy = ancho_img // 2, alto_img // 2
            m      = self.tamano_roi
            x1 = max(0, cx - m);         y1 = max(0, cy - m)
            x2 = min(ancho_img, cx + m); y2 = min(alto_img, cy + m)
            mascara_roi = np.zeros(combined_mask.shape, dtype=np.uint8)
            cv2.rectangle(mascara_roi, (x1, y1), (x2, y2), 255, -1)
            combined_mask = cv2.bitwise_and(combined_mask, mascara_roi)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        latency = (time.time() - start_time) * 1000
        status  = "MOTION DETECTED" if motion_detected else "Monitoring..."
        color   = (0, 100, 255) if motion_detected else (255, 255, 255)
        cv2.putText(frame, f"{status} | Latency: {latency:.1f} ms",
                    (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color, 2)
        return frame

    # -----------------------------------------------------------------------
    # Auxiliares
    # -----------------------------------------------------------------------

    def _is_active(self) -> bool:
        return self._state_machine.current_state[1] == "active"

    def _virtual_zoom(self, frame: np.ndarray, factor: float) -> np.ndarray:
        if factor <= 1:
            return frame
        h, w   = frame.shape[:2]
        nw, nh = int(w / factor), int(h / factor)
        x1, y1 = (w - nw) // 2, (h - nh) // 2
        return cv2.resize(frame[y1:y1 + nh, x1:x1 + nw], (w, h))


# ===========================================================================
# Punto de entrada
# ===========================================================================

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()