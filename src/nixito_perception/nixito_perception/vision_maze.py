import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path
import threading
import csv
import cv2
import numpy as np
import time


# ── Clases ────────────────────────────────────────────────────────────────────

HAZMAT_CLASSES = {
    '1.1 Explosives', '1.5 Blasting Agents', '2 Flamable gas',
    '2 Non-Flamable gas', '2 Oxygen', '3 Fuel Oil',
    '4 Dangerous when wet', '4 Flammable solid', '4 Spontaneously Combustible',
    '5.1 Oxidizer', '5.2 Organic Peroxide', '6 Infectious Substance',
    '6 Inhalation hazard', '6 Poison', '7 Radioactive', '8 Corrosive',
}

REAL_OBJECT_CLASSES = {
    'Backpack', 'Fire Extinguisher', 'gas tank', 'Helmet',
}

# ── Configuración general ─────────────────────────────────────────────────────

YEAR               = '2026'
TEAM_NAME          = 'NIXITO'
COUNTRY            = 'Mexico'
ROBOT              = 'Nixito'
MODE               = 'T'
QR_HOLD_SECS       = 0.1
QR_MIN_PERIOD      = 0.05
QR_DETECT_WIDTH    = 640
QR_MIN_AREA        = 100
QR_MAX_RATIO       = 0.50
DETECTION_COOLDOWN = 20.0

# ── Configuración AprilTag ────────────────────────────────────────────────────

# Diccionarios AprilTag a detectar (se pueden activar/desactivar según la arena)
APRILTAG_DICTS = {
    'DICT_APRILTAG_36H11': cv2.aruco.DICT_APRILTAG_36H11,
    'DICT_APRILTAG_36H10': cv2.aruco.DICT_APRILTAG_36H10,
    'DICT_APRILTAG_25H9':  cv2.aruco.DICT_APRILTAG_25H9,
    'DICT_APRILTAG_16H5':  cv2.aruco.DICT_APRILTAG_16H5,
}

# Tamaño lateral real del marcador en metros (ajustar según el marcador físico)
APRILTAG_MARKER_SIZE_M = 0.15

AT_MIN_AREA        = 400    # área mínima del bounding box en px² para validar
AT_HOLD_SECS       = 0.15   # segundos que se mantiene el overlay tras perder detección
AT_MIN_PERIOD      = 0.05   # intervalo mínimo entre llamadas al detector
AT_DETECT_WIDTH    = 640    # resolución de trabajo para el detector

QR_MODEL_DIR = '/home/angel/NXL_Robocup/src/nixito_perception/drivers/qr_models'

PKG_DIR      = Path('/home/angel/NXL_Robocup/src/nixito_perception/nixito_perception/csv')
MISSION_FILE = PKG_DIR / 'mission.txt'
CSV_DIR      = PKG_DIR / 'csv'

TIMER_CHECK_SUBS = 2.0


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('model_path',
            '/home/angel/NXL_Robocup/src/nixito_perception/modelos/Robocup_NXL.pt')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('confidence_threshold', 0.75)
        self.declare_parameter('min_confirmations', 5)

        model_path                = self.get_parameter('model_path').value
        image_topic               = self.get_parameter('image_topic').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.min_confirmations    = self.get_parameter('min_confirmations').value

        # ── Modelo YOLO ───────────────────────────────────────────────────────
        self.model = YOLO(model_path)
        self.get_logger().info(f'Modelo cargado: {model_path}')

        # ── Estado interno ────────────────────────────────────────────────────
        self.bridge               = CvBridge()
        self.detection_counts     = {}
        self.detection_counter    = 0
        self._last_published_time = {}

        self.latest_frame   = None
        self.latest_header  = None
        self.last_annotated = None
        self.lock           = threading.Lock()

        # ── Profundidad ───────────────────────────────────────────────────────
        self.depth_image = None
        self.camera_info = None
        self.fx = self.fy = self.cx = self.cy = None

        # ── WeChatQRCode ──────────────────────────────────────────────────────
        self.get_logger().info('Inicializando WeChatQRCode …')
        try:
            d  = f'{QR_MODEL_DIR}/detect.prototxt'
            dc = f'{QR_MODEL_DIR}/detect.caffemodel'
            s  = f'{QR_MODEL_DIR}/sr.prototxt'
            sc = f'{QR_MODEL_DIR}/sr.caffemodel'
            self.qr_detector = cv2.wechat_qrcode_WeChatQRCode(d, dc, s, sc)
            self.get_logger().info('WeChatQRCode cargado con modelos CNN.')
        except Exception as e:
            self.get_logger().warn(
                f'No se pudieron cargar modelos WeChat CNN ({e}). '
                f'Usando modo sin modelos — descarga los .prototxt/.caffemodel en {QR_MODEL_DIR}.'
            )
            self.qr_detector = cv2.wechat_qrcode_WeChatQRCode()

        self._last_qr_time  = 0.0
        self._qr_last_value = ''
        self._qr_last_pts   = None
        self._qr_last_seen  = 0.0
        self._qr_published  = set()

        # ── AprilTag ──────────────────────────────────────────────────────────
        self.get_logger().info('Inicializando detectores AprilTag …')
        self._at_detectors = self._build_apriltag_detectors()
        self._at_params     = self._build_apriltag_params()
        self._at_last_time  = 0.0
        self._at_published  = set()          # "DICT_NAME:id" ya publicados (cooldown propio)
        self._at_last_detections = {}        # "DICT_NAME:id" → {corners, center, seen_time}
        self.get_logger().info(
            f'AprilTag: {len(self._at_detectors)} diccionario(s) activos.'
        )

        # ── CSV (se inicializa lazy al detectar el primer suscriptor) ─────────
        self.csv_file   = None
        self.csv_writer = None
        self._csv_ready = False

        # ── Publicadores ──────────────────────────────────────────────────────
        self.pub       = self.create_publisher(String, 'detection/name',      10)
        self.image_pub = self.create_publisher(Image,  'detection/annotated', 10)

        # ── Suscripciones de sensor ───────────────────────────────────────────
        self.create_subscription(Image,      image_topic,                                       self.image_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',                self.info_callback,  10)
        self.create_subscription(Image,      '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)

        # ── Timer: revisión de suscriptores ───────────────────────────────────
        self._processing_active = False
        self.create_timer(TIMER_CHECK_SUBS, self._check_subscribers)

        # ── Hilo de inferencia ────────────────────────────────────────────────
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()

        self.get_logger().info('Nodo iniciado — esperando suscriptores …')

    # ── Helpers AprilTag ──────────────────────────────────────────────────────

    def _build_apriltag_detectors(self) -> dict:
        """Crea un cv2.aruco.ArucoDetector por cada diccionario AprilTag."""
        detectors = {}
        for name, dict_id in APRILTAG_DICTS.items():
            dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            detectors[name] = cv2.aruco.ArucoDetector(dictionary, self._build_apriltag_params())
        return detectors

    @staticmethod
    def _build_apriltag_params() -> cv2.aruco.DetectorParameters:
        params = cv2.aruco.DetectorParameters()
        # Refinamiento de esquinas con el algoritmo propio de AprilTag
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        params.minMarkerPerimeterRate  = 0.02
        params.maxMarkerPerimeterRate  = 4.0
        params.polygonalApproxAccuracyRate = 0.05
        params.minCornerDistanceRate   = 0.05
        params.minDistanceToBorder     = 3
        return params

    # ── Revisión de suscriptores ──────────────────────────────────────────────

    def _check_subscribers(self) -> None:
        has_subs = (
            self.pub.get_subscription_count()       > 0 or
            self.image_pub.get_subscription_count() > 0
        )

        if has_subs and not self._processing_active:
            self._processing_active = True
            if not self._csv_ready:
                self.csv_file, self.csv_writer = self._init_csv()
                self._csv_ready = True
                self.get_logger().info('CSV creado al detectar el primer suscriptor.')
            self.get_logger().info('Suscriptor detectado — procesamiento ACTIVO')

        elif not has_subs and self._processing_active:
            self._processing_active = False
            self.detection_counts.clear()
            self.get_logger().info('Sin suscriptores — procesamiento INACTIVO')

    # ── Profundidad ───────────────────────────────────────────────────────────

    def info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.camera_info = msg

    def depth_callback(self, msg: Image):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        with self.lock:
            self.depth_image = depth

    def pixel_to_3d(self, u: int, v: int):
        with self.lock:
            depth = self.depth_image
            info  = self.camera_info

        if depth is None or info is None:
            return None

        h, w = depth.shape[:2]
        u = int(np.clip(u, 5, w - 6))
        v = int(np.clip(v, 5, h - 6))

        roi   = depth[v - 5:v + 5, u - 5:u + 5]
        valid = roi[roi > 0]

        if len(valid) == 0:
            return None

        z = np.median(valid) / 1000.0
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return float(x), float(y), float(z)

    # ── CSV ───────────────────────────────────────────────────────────────────

    def _read_mission_number(self) -> int:
        if not MISSION_FILE.exists():
            MISSION_FILE.write_text('1')
        return int(MISSION_FILE.read_text().strip())

    def _increment_mission_number(self, current: int):
        MISSION_FILE.write_text(str(current + 1))

    def _init_csv(self):
        mission_num = self._read_mission_number()
        self._increment_mission_number(mission_num)
        self.current_mission_num = mission_num

        now               = datetime.now()
        start_date        = now.strftime('%Y-%m-%d')
        start_time_file   = now.strftime('%H-%M-%S')
        start_time_header = now.strftime('%H:%M:%S')

        CSV_DIR.mkdir(parents=True, exist_ok=True)

        filename = (
            f'RoboCup{YEAR}-{TEAM_NAME}-Prelim{mission_num}'
            f'-{start_date}-{start_time_file}-pois.csv'
        )
        filepath = CSV_DIR / filename

        f      = open(filepath, 'w', newline='')
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

        f.write('"pois"\n')
        f.write('"1.3"\n')
        f.write(f'"{TEAM_NAME}"\n')
        f.write(f'"{COUNTRY}"\n')
        f.write(f'"{start_date}"\n')
        f.write(f'"{start_time_header}"\n')
        f.write(f'"{mission_num}"\n')
        f.write('\n')
        f.write('detection,time,type,name,x,y,z,robot,mode\n')
        f.flush()

        self.get_logger().info(f'CSV creado: {filepath}')
        return f, writer

    def write_csv_row(self, detection_type: str, name: str,
                      x: float = None, y: float = None, z: float = None):
        if not self._csv_ready or self.csv_writer is None:
            self.get_logger().warn(
                f'CSV no listo — detección ignorada en disco: {name}'
            )
            return

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.csv_writer.writerow([
            self.detection_counter,
            timestamp,
            detection_type,
            name,
            f'{x:.3f}' if x is not None else '',
            f'{y:.3f}' if y is not None else '',
            f'{z:.3f}' if z is not None else '',
            ROBOT,
            MODE,
        ])
        self.csv_file.flush()

    # ── Detección ─────────────────────────────────────────────────────────────

    def get_detection_type(self, name: str) -> str:
        if name in HAZMAT_CLASSES:
            return 'hazmat_sign'
        elif name in REAL_OBJECT_CLASSES:
            return 'real_object'
        return 'unknown'

    def _publish_detection(self, detection_type: str, name: str,
                           confidence: float = 0.0,
                           x: float = None, y: float = None, z: float = None):
        now  = time.time()
        last = self._last_published_time.get(name, 0.0)

        if now - last < DETECTION_COOLDOWN:
            remaining = DETECTION_COOLDOWN - (now - last)
            self.get_logger().debug(
                f'Ignorado {name} — faltan {remaining:.1f}s para cooldown'
            )
            return

        self._last_published_time[name] = now
        self.detection_counter += 1
        self.write_csv_row(detection_type, name, x, y, z)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        x_str = f'{x:.3f}' if x is not None else ' '
        y_str = f'{y:.3f}' if y is not None else ' '
        z_str = f'{z:.3f}' if z is not None else ' '

        if self.pub.get_subscription_count() > 0:
            out      = String()
            out.data = (
                f'{self.detection_counter},'
                f'{timestamp},'
                f'{detection_type},'
                f'"{name}",'
                f'{x_str},{y_str},{z_str},'
                f'{ROBOT},'
                f'{MODE}'
            )
            self.pub.publish(out)
            self.get_logger().info(f'Publicado: {out.data}')
        else:
            self.get_logger().debug(
                f'Detección registrada en CSV pero sin suscriptores: {name}'
            )

    # ── Pipeline QR ───────────────────────────────────────────────────────────

    def _validate_qr(self, value: str, pts: np.ndarray, frame_shape: tuple) -> bool:
        if not value or not value.strip():
            return False
        pts_i       = np.int32(pts).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(pts_i)
        area        = w * h
        fh, fw      = frame_shape[:2]
        if area < QR_MIN_AREA or area / (fw * fh) > QR_MAX_RATIO:
            return False
        if h == 0 or abs(w / h - 1.0) > 0.3:
            return False
        if value.count('\x00') / len(value) > 0.1:
            return False
        return True

    def _process_qr(self, frame: np.ndarray, scale_x: float, scale_y: float) -> tuple:
        t0  = time.time()
        now = t0
        age = now - self._qr_last_seen

        if age < QR_HOLD_SECS and (now - self._last_qr_time) < QR_MIN_PERIOD:
            self._draw_qr_overlay(frame, age, scale_x, scale_y)
            return frame, None, None

        self._last_qr_time = now

        h, w      = frame.shape[:2]
        wechat_w  = QR_DETECT_WIDTH
        wc_scale  = wechat_w / w

        if wc_scale < 1.0:
            small_qr = cv2.resize(frame, (wechat_w, int(h * wc_scale)),
                                  interpolation=cv2.INTER_AREA)
        else:
            small_qr = frame

        try:
            texts, points = self.qr_detector.detectAndDecode(small_qr)
        except cv2.error as e:
            self.get_logger().warn(f'WeChat detectAndDecode error: {e}')
            texts, points = [], []

        value, pts_ann = '', None
        for text, pt in zip(texts, points):
            if not text:
                continue
            pts_in_frame = pt / wc_scale
            if self._validate_qr(text, pts_in_frame, frame.shape):
                value   = text
                pts_ann = pts_in_frame
                break

        new_detection = None
        qr_coords     = None

        if pts_ann is not None:
            self._qr_last_value = value
            self._qr_last_pts   = pts_ann
            self._qr_last_seen  = now

            if value not in self._qr_published:
                self._qr_published.add(value)
                new_detection = value

                pts_i      = np.int32(pts_ann).reshape(-1, 2)
                qr_cx_ann  = int(pts_i[:, 0].mean())
                qr_cy_ann  = int(pts_i[:, 1].mean())
                qr_cx_orig = int(qr_cx_ann * scale_x)
                qr_cy_orig = int(qr_cy_ann * scale_y)
                qr_coords  = self.pixel_to_3d(qr_cx_orig, qr_cy_orig)

        age = now - self._qr_last_seen
        self._draw_qr_overlay(frame, age, scale_x, scale_y)
        return frame, new_detection, qr_coords

    def _draw_qr_overlay(self, frame: np.ndarray, age: float,
                         scale_x: float, scale_y: float) -> None:
        if age >= QR_HOLD_SECS or self._qr_last_pts is None:
            return

        pts_i        = np.int32(self._qr_last_pts).reshape(-1, 2)
        x, y, bw, bh = cv2.boundingRect(pts_i)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        label = self._qr_last_value[:40] + ('…' if len(self._qr_last_value) > 40 else '')
        cv2.putText(frame, label, (x, max(y - 22, 20)),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.65, (255, 255, 255), 2)

        qr_cx_ann  = x + bw // 2
        qr_cy_ann  = y + bh // 2
        qr_cx_orig = int(qr_cx_ann * scale_x)
        qr_cy_orig = int(qr_cy_ann * scale_y)
        rt_coords  = self.pixel_to_3d(qr_cx_orig, qr_cy_orig)

        cv2.circle(frame, (qr_cx_ann, qr_cy_ann), 6, (0, 0, 255), -1)
        dist_text = f'{rt_coords[2]:.2f}m' if rt_coords is not None else '--'
        cv2.putText(frame, dist_text,
                    (qr_cx_ann + 8, qr_cy_ann),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ── Pipeline AprilTag ─────────────────────────────────────────────────────

    def _process_apriltag(self, frame: np.ndarray, scale_x: float, scale_y: float) -> tuple:
        """
        Detecta AprilTags en `frame` (resolución anotada 640×360).
        Retorna el frame anotado y una lista de nuevas detecciones:
            [(tag_key, tag_label, coords_3d), ...]
        Respeta:
          - AT_MIN_PERIOD  : intervalo mínimo entre llamadas al detector
          - AT_HOLD_SECS   : mantiene el overlay aunque el tag ya no se vea
          - DETECTION_COOLDOWN: heredado de _publish_detection
        """
        now = time.time()

        # ── Frecuencia de detección ───────────────────────────────────────────
        if now - self._at_last_time < AT_MIN_PERIOD:
            # Solo dibujar el overlay de la última detección conocida
            self._draw_apriltag_overlays(frame, now, scale_x, scale_y)
            return frame, []

        self._at_last_time = now

        # ── Preparar imagen de detección ──────────────────────────────────────
        h, w     = frame.shape[:2]
        at_scale = AT_DETECT_WIDTH / w
        if at_scale < 1.0:
            small_at = cv2.resize(frame, (AT_DETECT_WIDTH, int(h * at_scale)),
                                  interpolation=cv2.INTER_AREA)
        else:
            small_at = frame
            at_scale = 1.0

        gray = cv2.cvtColor(small_at, cv2.COLOR_BGR2GRAY)

        # ── Detección en todos los diccionarios ───────────────────────────────
        new_detections = []
        detected_keys  = set()

        for dict_name, detector in self._at_detectors.items():
            try:
                corners_list, ids, _ = detector.detectMarkers(gray)
            except cv2.error as e:
                self.get_logger().warn(f'AprilTag detect error ({dict_name}): {e}')
                continue

            if ids is None:
                continue

            for corners, tag_id in zip(corners_list, ids.flatten()):
                # Escalar esquinas de vuelta a resolución anotada
                corners_frame = corners[0] / at_scale      # shape (4, 2)

                # ── Validar por área ──────────────────────────────────────────
                x_c, y_c, bw, bh = cv2.boundingRect(np.int32(corners_frame))
                area = bw * bh
                if area < AT_MIN_AREA:
                    continue

                tag_key   = f'{dict_name}:{tag_id}'   # clave interna (cooldown/dedup)
                tag_label = str(int(tag_id))            # solo el número → va al CSV y topic
                detected_keys.add(tag_key)

                # Centro en frame anotado → pixel original
                cx_ann  = int(corners_frame[:, 0].mean())
                cy_ann  = int(corners_frame[:, 1].mean())
                cx_orig = int(cx_ann * scale_x)
                cy_orig = int(cy_ann * scale_y)

                coords_3d = self.pixel_to_3d(cx_orig, cy_orig)

                # ── Actualizar overlay ────────────────────────────────────────
                self._at_last_detections[tag_key] = {
                    'corners':   corners_frame,
                    'center':    (cx_ann, cy_ann),
                    'seen_time': now,
                    'coords':    coords_3d,
                    'label':     tag_label,
                }

                # ── Nueva detección (para publicar) ───────────────────────────
                if tag_key not in self._at_published:
                    self._at_published.add(tag_key)
                    new_detections.append((tag_key, tag_label, coords_3d))
                    self.get_logger().info(
                        f'AprilTag detectado: {tag_label}  '
                        f'dist={coords_3d[2]:.2f}m' if coords_3d else
                        f'AprilTag detectado: {tag_label}  dist=--'
                    )

        # Limpiar de _at_last_detections los tags que llevan mucho sin verse
        for key in list(self._at_last_detections.keys()):
            if now - self._at_last_detections[key]['seen_time'] > AT_HOLD_SECS * 10:
                del self._at_last_detections[key]

        # ── Dibujar overlays ──────────────────────────────────────────────────
        self._draw_apriltag_overlays(frame, now, scale_x, scale_y)

        return frame, new_detections

    def _draw_apriltag_overlays(self, frame: np.ndarray, now: float,
                                scale_x: float, scale_y: float) -> None:
        """Dibuja el overlay de todos los AprilTags visibles recientemente."""
        for tag_key, info in self._at_last_detections.items():
            age = now - info['seen_time']
            if age > AT_HOLD_SECS:
                continue

            corners = np.int32(info['corners'])
            cx, cy  = info['center']
            label   = info['label']
            coords  = info['coords']

            # ── Relleno semitransparente ──────────────────────────────────────
            overlay = frame.copy()
            cv2.fillPoly(overlay, [corners], (255, 128, 0))
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

            # ── Contorno del marcador ─────────────────────────────────────────
            cv2.polylines(frame, [corners], isClosed=True, color=(255, 128, 0), thickness=2)

            # ── Ejes de las esquinas ──────────────────────────────────────────
            for i, pt in enumerate(corners):
                cv2.circle(frame, tuple(pt), 4, (0, 128, 255), -1)

            # ── Punto central ─────────────────────────────────────────────────
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

            # ── Etiqueta: solo el ID numérico ─────────────────────────────────
            x_b, y_b, bw, bh = cv2.boundingRect(corners)
            cv2.putText(frame, f'AprilTag {label}',
                        (x_b, max(y_b - 22, 20)),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.55, (255, 255, 255), 2)

            # ── Distancia ─────────────────────────────────────────────────────
            # Obtener distancia en tiempo real (no solo la del momento de detección)
            cx_orig = int(cx * scale_x)
            cy_orig = int(cy * scale_y)
            rt_coords = self.pixel_to_3d(cx_orig, cy_orig)

            dist_text = f'{rt_coords[2]:.2f}m' if rt_coords is not None else '--'
            cv2.putText(frame, dist_text,
                        (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        with self.lock:
            self.latest_frame  = frame
            self.latest_header = msg.header
            annotated          = self.last_annotated

        if self.image_pub.get_subscription_count() > 0:
            annotated_msg        = self.bridge.cv2_to_imgmsg(
                annotated if annotated is not None else frame, encoding='bgr8')
            annotated_msg.header = msg.header
            self.image_pub.publish(annotated_msg)

    # ── Hilo de inferencia ────────────────────────────────────────────────────

    def inference_loop(self):
        while rclpy.ok():
            if not self._processing_active:
                time.sleep(0.1)
                continue

            with self.lock:
                frame = self.latest_frame

            if frame is None:
                continue

            orig_h, orig_w = frame.shape[:2]
            scale_x = orig_w / 640.0
            scale_y = orig_h / 360.0

            small       = cv2.resize(frame, (640, 360))
            results     = self.model(small, conf=self.confidence_threshold, imgsz=320, verbose=False)[0]
            annotated   = results.plot()

            detected_this_frame = set()

            # ── YOLO ──────────────────────────────────────────────────────────
            for box in results.boxes:
                class_id   = int(box.cls.item())
                confidence = float(box.conf.item())
                name       = results.names[class_id]
                detected_this_frame.add(name)

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx_ann  = int((x1 + x2) / 2)
                cy_ann  = int((y1 + y2) / 2)
                cx_orig = int(cx_ann * scale_x)
                cy_orig = int(cy_ann * scale_y)

                coords_3d = self.pixel_to_3d(cx_orig, cy_orig)
                dist_m    = coords_3d[2] if coords_3d is not None else None

                cv2.circle(annotated, (cx_ann, cy_ann), 6, (0, 0, 255), -1)
                cv2.putText(annotated,
                            f'{dist_m:.2f}m' if dist_m is not None else '--',
                            (cx_ann + 8, cy_ann),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                self.detection_counts[name] = self.detection_counts.get(name, 0) + 1
                count = self.detection_counts[name]

                if count >= self.min_confirmations:
                    detection_type = self.get_detection_type(name)
                    z3d = coords_3d[2] if coords_3d else None

                    if detection_type != 'unknown':
                        self._publish_detection(
                            detection_type, name, confidence,
                            x=0, y=0, z=z3d)
                    else:
                        self.get_logger().warn(f'Clase sin mapear: {name}')
                    self.detection_counts[name] = 0

            for name in list(self.detection_counts.keys()):
                if name not in detected_this_frame:
                    self.detection_counts[name] = 0

            # ── QR ────────────────────────────────────────────────────────────
            annotated, qr_value, qr_coords = self._process_qr(annotated, scale_x, scale_y)

            if qr_value:
                z3d = qr_coords[2] if qr_coords else None
                self._publish_detection('ar_code', qr_value, x=0, y=0, z=z3d)
                self.get_logger().info(f'QR detectado: {qr_value}')

            # ── AprilTag ──────────────────────────────────────────────────────
            annotated, at_detections = self._process_apriltag(annotated, scale_x, scale_y)

            for tag_key, tag_label, at_coords in at_detections:
                z3d = at_coords[2] if at_coords else None
                self._publish_detection('ar_code', tag_label, x=0, y=0, z=z3d)

            with self.lock:
                self.last_annotated = annotated


# ── Punto de entrada ──────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    finally:
        if node.csv_file is not None:
            node.csv_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()