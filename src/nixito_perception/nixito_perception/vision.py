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
YOLO_MIN_PERIOD  = 0.05    # s  → throughput máximo ~20 fps
YOLO_CONF        = 0.5     # Umbral de confianza mínima
YOLO_IMGSZ       = 480     # Tamaño de imagen de entrada para inferencia

# ── QR ──────────────────────────────────────────────────────────────────────
QR_HOLD_SECS = 0.8         # Segundos que persiste el último resultado detectado

# ── Detección de movimiento ──────────────────────────────────────────────────
MOV_HISTORY        = 500   # Historial de frames del sustractor MOG2
MOV_VAR_THRESHOLD  = 8     # Sensibilidad del sustractor MOG2
MOV_LEARNING_RATE  = 0.002 # Tasa de aprendizaje del modelo de fondo
MOV_DIFF_THRESHOLD = 6     # Umbral de diferencia de pixel para máscara temporal
MOV_MIN_AREA       = 500   # px² – área mínima de contorno a considerar
MOV_GROUP_EPS      = 0.3   # Tolerancia de solapamiento para groupRectangles
MOV_WARMUP_LEN     = 15    # Frames de calentamiento antes de iniciar detección

# ── Buffer de frames ─────────────────────────────────────────────────────────
FRAME_BUFFER_MAXLEN = 15   # Longitud máxima del buffer circular de frames grises

# ── Topics ROS ───────────────────────────────────────────────────────────────
TOPIC_INPUT_RGB     = "/principal/image_raw"

# Mapa: nombre_modo → (topic_de_salida, qos)
TOPIC_OUTPUTS = {
    "yolo":     ("vision/yolo",     10),
    "qr":       ("vision/qr",       10),
    "movement": ("vision/movement", 10),
}

# ── Periodos de los timers internos (segundos) ───────────────────────────────
TIMER_CHECK_SUBS = 2.0     # Revisión de suscriptores activos
TIMER_LOG_STATS  = 10.0    # Log de estado periódico


# ===========================================================================
# Nodo principal
# ===========================================================================

class VisionNode(LifecycleNode):

    # -----------------------------------------------------------------------
    # Construcción
    # -----------------------------------------------------------------------

    def __init__(self):
        super().__init__("vision_node")

        # Conversor entre mensajes ROS Image y arrays NumPy / OpenCV
        self.bridge = CvBridge()

        # Diccionario de publicadores lifecycle, indexado por nombre de modo
        self.pubs: dict[str, Publisher] = {}

        # Modo activo actualmente; None indica que el nodo está inactivo
        self.active_mode: str | None = None

        # ── Recursos pesados (carga perezosa) ────────────────────────────────
        self.model        = None   # Modelo YOLO
        self.model_loaded = False
        self.qr_detector  = None   # Detector de QR de OpenCV
        self.mog2         = None   # Sustractor de fondo MOG2 (movimiento)

        # ── Recursos ligeros (siempre disponibles) ───────────────────────────
        self.frame_buffer  = deque(maxlen=FRAME_BUFFER_MAXLEN)  # Buffer circular de frames
        self.kernel        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._clahe        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # LUT para corrección gamma (γ = 2.0) aplicada a frames oscuros
        self._gamma_lut = np.array(
            [int(((i / 255.0) ** 2.0) * 255) for i in range(256)], dtype=np.uint8
        )

        # Sustractor de fondo auxiliar (no es el MOG2 del pipeline de movimiento)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=40, detectShadows=False
        )

        # ── Control de tasa YOLO ─────────────────────────────────────────────
        self.last_yolo_time = 0.0  # Timestamp de la última inferencia YOLO

        # ── ROI de movimiento ────────────────────────────────────────────────
        # Semianchura (px) de la región de interés centrada. 0 = frame completo.
        self.tamano_roi = 150

        # ── Estado de confirmación QR ────────────────────────────────────────
        self._qr_last_seen  = 0.0   # Timestamp de la última detección válida
        self._qr_last_value = ""    # Texto del último QR detectado
        self._qr_last_pts   = None  # Puntos del contorno del último QR

        # ── Handles de timers (se crean en on_configure) ─────────────────────
        self._timer_check_subs = None
        self._timer_stats      = None

        self.get_logger().info("VisionNode creado (UNCONFIGURED)")

    # -----------------------------------------------------------------------
    # Callbacks del ciclo de vida
    # -----------------------------------------------------------------------

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configurando nodo …")

        # Publicadores lifecycle (se activan/desactivan junto con el nodo)
        for mode, (topic, qos) in TOPIC_OUTPUTS.items():
            self.pubs[mode] = self.create_lifecycle_publisher(Image, topic, qos)
            self.get_logger().info(f"  Publicador listo: {topic}")

        # Suscripciones a las cámaras de entrada
        self.sub_rgb = self.create_subscription(
            Image, TOPIC_INPUT_RGB, self._main_loop, 1
        )
        qos_thermal = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Timers de monitorización
        self._timer_check_subs = self.create_timer(TIMER_CHECK_SUBS, self._check_subscribers)
        self._timer_stats      = self.create_timer(TIMER_LOG_STATS,  self._log_stats)

        self.get_logger().info("Nodo configurado — listo para activar")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """
        Transición INACTIVE → ACTIVE.
        Los publicadores lifecycle se habilitan automáticamente en la superclase.
        """
        self.get_logger().info("Nodo ACTIVO — esperando suscriptores …")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """
        Transición ACTIVE → INACTIVE.
        Libera los modelos pesados para recuperar memoria.
        """
        self.get_logger().info("Nodo INACTIVO — liberando recursos pesados …")
        self._unload_models()
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """
        Transición INACTIVE → UNCONFIGURED.
        Destruye suscripciones y timers.
        """
        self.get_logger().info("Limpiando recursos …")
        self.destroy_subscription(self.sub_rgb)
        self.destroy_subscription(self.sub_thermal)
        self.destroy_timer(self._timer_check_subs)
        self.destroy_timer(self._timer_stats)
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """Apagado del nodo; libera todos los modelos antes de finalizar."""
        self.get_logger().info("Apagando nodo …")
        self._unload_models()
        return TransitionCallbackReturn.SUCCESS

    # -----------------------------------------------------------------------
    # Gestión de modelos pesados
    # -----------------------------------------------------------------------

    def _load_models_for_mode(self, mode: str) -> None:
        if mode == "yolo" and not self.model_loaded:
            self.get_logger().info("Cargando modelo YOLO …")
            t0 = time.time()
            self.model = YOLO(YOLO_MODEL_PATH)
            self.model_loaded = True
            self.get_logger().info(f"YOLO cargado en {time.time() - t0:.1f} s")

        elif mode == "qr" and self.qr_detector is None:
            self.get_logger().info("Inicializando detector QR …")
            self.qr_detector = cv2.QRCodeDetector()

        elif mode == "movement" and self.mog2 is None:
            self.get_logger().info("Inicializando sustractor MOG2 …")
            self.mog2 = cv2.createBackgroundSubtractorMOG2(
                history=MOV_HISTORY,
                varThreshold=MOV_VAR_THRESHOLD,
                detectShadows=False,
            )

    def _unload_models(self) -> None:
        """Libera la memoria ocupada por los modelos pesados."""
        if self.model_loaded:
            self.get_logger().info("Descargando modelo YOLO …")
            self.model = None
            self.model_loaded = False

        self.qr_detector = None
        self.mog2 = None

    # -----------------------------------------------------------------------
    # Monitorización de suscriptores
    # -----------------------------------------------------------------------

    def _check_subscribers(self) -> None:
        if not self._is_active():
            return

        for mode, pub in self.pubs.items():
            if pub.get_subscription_count() > 0:
                # Cambio de modo: carga el modelo si hace falta
                if self.active_mode != mode:
                    self.get_logger().info(f"Suscriptor detectado para modo: {mode}")
                    self._load_models_for_mode(mode)
                    self.active_mode = mode
                return  # Solo el primer topic con suscriptor gana

        # Sin suscriptores en ningún topic → modo idle
        if self.active_mode is not None:
            self.get_logger().info(f"Sin suscriptores — desactivando modo: {self.active_mode}")
            self.active_mode = None

    def _log_stats(self) -> None:
        """Timer callback de heartbeat: registra el modo activo cada TIMER_LOG_STATS segundos."""
        if self.active_mode:
            self.get_logger().info(f"Modo activo: {self.active_mode}")
        else:
            self.get_logger().info("Idle — sin procesamiento activo")

    # -----------------------------------------------------------------------
    # Bucle principal de procesamiento
    # -----------------------------------------------------------------------

    def _main_loop(self, msg: Image) -> None:
        if not self._is_active() or self.active_mode is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Tabla de despacho: modo → (listo, función_de_proceso)
        processors = {
            "yolo":     (self.model_loaded,             self._process_yolo),
            "qr":       (self.qr_detector is not None,  self._process_qr),
            "thermal":  (True,                           self._process_thermal),
            "movement": (self.mog2 is not None,          self._process_movement),
        }

        ready, process_fn = processors.get(self.active_mode, (False, None))
        if not ready or process_fn is None:
            return  # Modelo aún no cargado o modo desconocido

        result  = process_fn(frame)
        out_msg = self.bridge.cv2_to_imgmsg(result, "bgr8")
        self.pubs[self.active_mode].publish(out_msg)

    # -----------------------------------------------------------------------
    # Pipelines de procesamiento
    # -----------------------------------------------------------------------

    # ── Pipeline YOLO ───────────────────────────────────────────────────────

    def _process_yolo(self, frame: np.ndarray) -> np.ndarray:
        t0      = time.time()
        results = self.model(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False)[0]
        frame   = results.plot()   # Dibuja cajas/máscaras sobre el frame
        latency = (time.time() - t0) * 1000

        cv2.putText(
            frame,
            f"YOLO Active | Latency: {latency:.1f} ms",
            (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1,
        )
        return frame

    # ── Pipeline QR ─────────────────────────────────────────────────────────

    def _preprocess_for_qr(self, frame: np.ndarray):

        h, w  = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2))
        scale = (w / (w // 2), h / (h // 2))

        # Preprocesamiento base: CLAHE + filtro bilateral + unsharp mask
        gray      = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray      = self._clahe.apply(gray)
        gray      = cv2.bilateralFilter(gray, 5, 75, 75)
        blur      = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpened = cv2.addWeighted(gray, 2.5, blur, -1.5, 0)

        # Umbralización adaptativa (útil con iluminación no uniforme)
        thresh = cv2.adaptiveThreshold(
            sharpened, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            11, 2,
        )

        # Corrección gamma para frames oscuros (media < 80)
        gamma = cv2.LUT(gray, self._gamma_lut) if np.mean(gray) < 80 else sharpened

        return [frame, thresh, cv2.bitwise_not(thresh), gamma], scale

    def _validate_qr(self, value: str, pts, frame_shape) -> bool:
 
        if not value or not value.strip():
            return False

        pts_i    = np.int32(pts).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(pts_i)
        area     = w * h
        fh, fw   = frame_shape[:2]

        if area < 400 or area / (fw * fh) > 0.5:
            return False
        if h == 0 or abs(w / h - 1.0) > 0.3:
            return False
        if value.count('\x00') / len(value) > 0.1:
            return False

        return True

    def _process_qr(self, frame: np.ndarray) -> np.ndarray:
        t0 = time.time()

        variants, (sx, sy) = self._preprocess_for_qr(frame)
        value, pts = "", None

        # Prueba cada variante hasta obtener una detección válida
        for i, img in enumerate(variants):
            try:
                v, p, _ = self.qr_detector.detectAndDecode(img)
            except cv2.error:
                continue
            if p is None or len(p) == 0:
                continue
            if cv2.contourArea(np.int32(p).reshape(-1, 2)) <= 0:
                continue
            if v:
                # Escalar puntos al sistema de coordenadas del frame original
                value, pts = v, p * ([sx, sy] if i > 0 else [1, 1])
                break

        # Actualizar estado si la detección supera la validación
        if pts is not None and self._validate_qr(value, pts, frame.shape):
            self._qr_last_value = value
            self._qr_last_pts   = pts
            self._qr_last_seen  = time.time()

        # Dibujar el resultado persistente mientras esté dentro del tiempo de hold
        age = time.time() - self._qr_last_seen
        if age < QR_HOLD_SECS and self._qr_last_pts is not None:
            pts_i        = np.int32(self._qr_last_pts).reshape(-1, 2)
            x, y, bw, bh = cv2.boundingRect(pts_i)

            # Rectángulo semitransparente de fondo verde
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

            # Borde sólido y etiqueta de texto
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            label = self._qr_last_value[:40] + ("…" if len(self._qr_last_value) > 40 else "")
            cv2.putText(
                frame, label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_TRIPLEX, 0.65, (255, 255, 255), 2,
            )

        status = "QR Active" if age < QR_HOLD_SECS else "Scanning"
        cv2.putText(
            frame,
            f"{status} | {(time.time() - t0) * 1000:.1f} ms",
            (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 0, 255), 1,
        )
        return frame

    # ── Pipeline de Movimiento ───────────────────────────────────────────────

    def _process_movement(self, frame: np.ndarray) -> np.ndarray:
        start_time = time.time()

        # ── Preprocesamiento ─────────────────────────────────────────────────
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)
        self.frame_buffer.append(gray_frame)

        # Esperar a que el buffer esté lleno antes de detectar
        if len(self.frame_buffer) < self.frame_buffer.maxlen:
            latency = (time.time() - start_time) * 1000
            cv2.putText(
                frame,
                f"Warming up... {len(self.frame_buffer)}/{self.frame_buffer.maxlen}",
                (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 2,
            )
            return frame

        # ── Diferencia temporal ──────────────────────────────────────────────
        # Comparar el frame actual con el más antiguo del buffer (ventana temporal)
        oldest_frame   = self.frame_buffer[0]
        temporal_delta = cv2.absdiff(gray_frame, oldest_frame)

        # Umbralización: umbral bajo para capturar movimientos lentos
        _, combined_mask = cv2.threshold(temporal_delta, 10, 255, cv2.THRESH_BINARY)

        # ── Limpieza morfológica ─────────────────────────────────────────────
        # Open  → elimina pequeños artefactos de ruido
        # Close → rellena huecos en regiones de movimiento
        # Dilate → engrosa las regiones para unir fragmentos cercanos
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,  self.kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        combined_mask = cv2.dilate(combined_mask, self.kernel, iterations=2)

        # ── Máscara ROI (opcional) ───────────────────────────────────────────
        if self.tamano_roi > 0:
            alto_img, ancho_img = frame.shape[:2]
            cx, cy = ancho_img // 2, alto_img // 2
            m      = self.tamano_roi
            x1 = max(0, cx - m);         y1 = max(0, cy - m)
            x2 = min(ancho_img, cx + m); y2 = min(alto_img, cy + m)

            # Crear máscara binaria con el rectángulo ROI y aplicarla
            mascara_roi = np.zeros(combined_mask.shape, dtype=np.uint8)
            cv2.rectangle(mascara_roi, (x1, y1), (x2, y2), 255, -1)
            combined_mask = cv2.bitwise_and(combined_mask, mascara_roi)

            # Visualizar ROI en el frame de salida
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # ── Detección de contornos ───────────────────────────────────────────
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 200:   # Ignorar contornos muy pequeños
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # ── Overlay de estado ────────────────────────────────────────────────
        latency = (time.time() - start_time) * 1000
        status  = "MOTION DETECTED" if motion_detected else "Monitoring..."
        color   = (0, 100, 255)    if motion_detected else (255, 255, 255)
        cv2.putText(
            frame,
            f"{status} | Latency: {latency:.1f} ms",
            (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color, 2,
        )
        return frame

    # -----------------------------------------------------------------------
    # Métodos auxiliares
    # -----------------------------------------------------------------------

    def _is_active(self) -> bool:
        """Devuelve True si la máquina de estados del ciclo de vida está en 'active'."""
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
