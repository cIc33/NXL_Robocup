import rclpy
from rclpy.node import Node
from rclpy.lifecycle import LifecycleNode, Publisher, State, TransitionCallbackReturn
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path
from nixito_msgs.msg import Detection
import threading
import csv
import cv2
import numpy as np
import time
import math


# ── Clases ────────────────────────────────────────────────────────────────────

HAZMAT_CLASSES = {
    '1.1 Explosives', '1.5 Blasting Agents', '2 Flamable gas',
    '2 Non-Flamable gas', '2 Oxygen', '3 Fuel Oil',
    '4 Dangerous when wet', '4 Flammable solid', '4 Spontaneously Combustible',
    '5.1 Oxidizer', '5.2 Organic Peroxide', '6 Infectious Substance',
    '6 Inhalation hazard', '6 Poison', '7 Radioactive', '8 Corrosive',
}

REAL_OBJECT_CLASSES = {
    'Backpack', 'Fire Extinguisher', 'Gas tank', 'Helmet', 'Baby',
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

APRILTAG_DICTS = {
    'DICT_APRILTAG_36H11': cv2.aruco.DICT_APRILTAG_36H11,
    'DICT_APRILTAG_36H10': cv2.aruco.DICT_APRILTAG_36H10,
    'DICT_APRILTAG_25H9':  cv2.aruco.DICT_APRILTAG_25H9,
    'DICT_APRILTAG_16H5':  cv2.aruco.DICT_APRILTAG_16H5,
}

APRILTAG_MARKER_SIZE_M = 0.15
AT_MIN_AREA            = 400
AT_HOLD_SECS           = 0.15
AT_MIN_PERIOD          = 0.05
AT_DETECT_WIDTH        = 640

QR_MODEL_DIR = '/home/nixito/NXL_Robocup/src/nixito_perception/drivers/qr_models'
MODEL_PATH   = '/home/nixito/NXL_Robocup/src/nixito_perception/modelos/Robocup_NXL.pt'

PKG_DIR      = Path('/home/nixito/NXL_Robocup/src/nixito_perception/nixito_perception/csv')
MISSION_FILE = PKG_DIR / 'mission.txt'
CSV_DIR      = PKG_DIR / 'csv'


class MazeNode(LifecycleNode):
    def __init__(self):
        super().__init__('detector_node')

        self.bridge               = CvBridge()
        self.model                = None
        self.qr_detector          = None
        self.detection_counts     = {}
        self.detection_counter    = 0
        self._last_published_time = {}

        self.latest_frame   = None
        self.latest_header  = None

        # Anotaciones por topico (independientes)
        self.last_annotated_yolo = None
        self.last_annotated_qr   = None
        self.last_annotated_at   = None

        self.lock           = threading.Lock()
        self._frame_event   = threading.Event()

        self.depth_image = None
        self.camera_info = None
        self.fx = self.fy = self.cx = self.cy = None

        self._last_qr_time  = 0.0
        self._qr_last_value = ''
        self._qr_last_pts   = None
        self._qr_last_seen  = 0.0
        self._qr_published  = set()

        self.csv_file   = None
        self.csv_writer = None
        self._csv_ready = False

        self.get_logger().info('Inicializando detectores AprilTag …')
        self._at_detectors       = self._build_apriltag_detectors()
        self._at_last_time       = 0.0
        self._at_published       = set()
        self._at_last_detections = {}

        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._inference_thread.start()

        self.get_logger().info('DetectorNode creado (UNCONFIGURED)')

    # =========================================================================
    # Ciclo de vida
    # =========================================================================

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Configurando …')

        self.declare_parameter('model_path',           MODEL_PATH)
        self.declare_parameter('image_topic',          '/camera/camera/color/image_raw')
        self.declare_parameter('confidence_threshold', 0.75)
        self.declare_parameter('min_confirmations',    5)

        self._model_path          = self.get_parameter('model_path').value
        self._image_topic         = self.get_parameter('image_topic').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.min_confirmations    = self.get_parameter('min_confirmations').value

        # ── Publicador de detecciones (común) ───────────────────────────────
        self._pub_detection = self.create_lifecycle_publisher(
            Detection, 'detection', 10
        )

        # ── Suscripciones de entrada ─────────────────────────────────────────
        self._sub_image = self.create_subscription(
            Image, self._image_topic, self._image_callback, 10)
        self._sub_info  = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self._info_callback, 10)
        self._sub_depth = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw',
            self._depth_callback, 10)

        # Los publicadores de imagen anotada por topico se crean en on_activate
        self._pub_annotated_yolo = None
        self._pub_annotated_qr   = None
        self._pub_annotated_at   = None

        self.get_logger().info('Configurado — listo para activar')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Activando — cargando modelos …')
        self._load_models()
        if not self._csv_ready:
            self.csv_file, self.csv_writer = self._init_csv()
            self._csv_ready = True

        # ── Crear los topicos de imagen anotada SOLO al activar ──────────────
        self._pub_annotated_yolo = self.create_lifecycle_publisher(
            Image, 'detection/yolo/annotated', 10
        )
        self._pub_annotated_qr = self.create_lifecycle_publisher(
            Image, 'detection/qr/annotated', 10
        )
        self._pub_annotated_at = self.create_lifecycle_publisher(
            Image, 'detection/apriltag/annotated', 10
        )

        self.get_logger().info('ACTIVO')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Desactivando …')
        self._frame_event.clear()
        self._unload_models()
        self.detection_counts.clear()

        # ── Destruir los topicos de imagen anotada al desactivar ─────────────
        if self._pub_annotated_yolo is not None:
            self.destroy_lifecycle_publisher(self._pub_annotated_yolo)
            self._pub_annotated_yolo = None
        if self._pub_annotated_qr is not None:
            self.destroy_lifecycle_publisher(self._pub_annotated_qr)
            self._pub_annotated_qr = None
        if self._pub_annotated_at is not None:
            self.destroy_lifecycle_publisher(self._pub_annotated_at)
            self._pub_annotated_at = None

        with self.lock:
            self.last_annotated_yolo = None
            self.last_annotated_qr   = None
            self.last_annotated_at   = None

        self.get_logger().info('INACTIVO')
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        if self.csv_file is not None:
            self.csv_file.close()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        if self.csv_file is not None:
            self.csv_file.close()
        return TransitionCallbackReturn.SUCCESS

    # =========================================================================
    # Modelos
    # =========================================================================

    def _load_models(self) -> None:
        if self.model is None:
            self.get_logger().info(f'Cargando YOLO: {self._model_path} …')
            t0 = time.time()
            self.model = YOLO(self._model_path)
            self.get_logger().info(f'YOLO listo en {time.time()-t0:.1f}s')

        if self.qr_detector is None:
            self.get_logger().info('Cargando WeChatQRCode …')
            t0 = time.time()
            try:
                self.qr_detector = cv2.wechat_qrcode_WeChatQRCode(
                    f'{QR_MODEL_DIR}/detect.prototxt',
                    f'{QR_MODEL_DIR}/detect.caffemodel',
                    f'{QR_MODEL_DIR}/sr.prototxt',
                    f'{QR_MODEL_DIR}/sr.caffemodel',
                )
                self.get_logger().info(f'WeChatQRCode listo en {time.time()-t0:.1f}s')
            except Exception as e:
                self.get_logger().warn(f'WeChat sin modelos CNN ({e})')
                self.qr_detector = cv2.wechat_qrcode_WeChatQRCode()

    def _unload_models(self) -> None:
        self.model        = None
        self.qr_detector  = None

    def _is_active(self) -> bool:
        return self._state_machine.current_state[1] == 'active'

    # =========================================================================
    # AprilTag
    # =========================================================================

    def _build_apriltag_detectors(self) -> dict:
        detectors = {}
        for name, dict_id in APRILTAG_DICTS.items():
            dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            detectors[name] = cv2.aruco.ArucoDetector(
                dictionary, self._build_apriltag_params()
            )
        return detectors

    @staticmethod
    def _build_apriltag_params() -> cv2.aruco.DetectorParameters:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod      = cv2.aruco.CORNER_REFINE_APRILTAG
        params.minMarkerPerimeterRate      = 0.02
        params.maxMarkerPerimeterRate      = 4.0
        params.polygonalApproxAccuracyRate = 0.05
        params.minCornerDistanceRate       = 0.05
        params.minDistanceToBorder         = 3
        return params

    # =========================================================================
    # Profundidad e intrínsecos
    # =========================================================================

    def _info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0];  self.fy = msg.k[4]
        self.cx = msg.k[2];  self.cy = msg.k[5]
        self.camera_info = msg

    def _depth_callback(self, msg: Image):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        with self.lock:
            self.depth_image = depth

    def pixel_to_3d(self, u: int, v: int):
        with self.lock:
            depth = self.depth_image
            info  = self.camera_info
        if depth is None or info is None:
            return None
        h, w  = depth.shape[:2]
        u     = int(np.clip(u, 5, w - 6))
        v     = int(np.clip(v, 5, h - 6))
        roi   = depth[v-5:v+5, u-5:u+5]
        valid = roi[roi > 0]
        if len(valid) == 0:
            return None
        z = np.median(valid) / 1000.0
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return float(x), float(y), float(z)

    def pixel_to_yaw(self, u: int) -> float:
        """
        Ángulo horizontal (yaw) en grados desde el centro óptico de la cámara.

        Retorna:
            yaw_deg > 0  →  objeto a la DERECHA del centro óptico
            yaw_deg < 0  →  objeto a la IZQUIERDA del centro óptico
            0.0          →  si los intrínsecos aún no están disponibles
        """
        if self.fx is None or self.cx is None:
            return 0.0
        return math.degrees(math.atan2(u - self.cx, self.fx))

    # =========================================================================
    # CSV
    # =========================================================================

    def _read_mission_number(self) -> int:
        if not MISSION_FILE.exists():
            MISSION_FILE.write_text('1')
        return int(MISSION_FILE.read_text().strip())

    def _init_csv(self):
        mission_num = self._read_mission_number()
        MISSION_FILE.write_text(str(mission_num + 1))
        self.current_mission_num = mission_num
        now             = datetime.now()
        start_date      = now.strftime('%Y-%m-%d')
        start_time_file = now.strftime('%H-%M-%S')
        start_time_hdr  = now.strftime('%H:%M:%S')
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        filename = (
            f'RoboCup{YEAR}-{TEAM_NAME}-Prelim{mission_num}'
            f'-{start_date}-{start_time_file}-pois.csv'
        )
        filepath = CSV_DIR / filename
        f      = open(filepath, 'w', newline='')
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        f.write('"pois"\n"1.3"\n')
        f.write(f'"{TEAM_NAME}"\n"{COUNTRY}"\n')
        f.write(f'"{start_date}"\n"{start_time_hdr}"\n"{mission_num}"\n\n')
        f.write('detection,time,type,name,x,y,z,robot,mode\n')
        f.flush()
        self.get_logger().info(f'CSV creado: {filepath}')
        return f, writer

    def _write_csv_row(self, msg: Detection):
        """Escribe directamente desde el mensaje — sin duplicar lógica."""
        if not self._csv_ready or self.csv_writer is None:
            return
        x = ''
        y = ''
        z = f'{msg.z:.3f}' if msg.z != -1.0 else ''
        self.csv_writer.writerow([
            msg.detection_id,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            msg.detection_type, msg.label,
            x, y, z,
            ROBOT, MODE,
        ])
        self.csv_file.flush()

    # =========================================================================
    # Publicación
    # =========================================================================

    def get_detection_type(self, name: str) -> str:
        if name in HAZMAT_CLASSES:      return 'hazmat_sign'
        if name in REAL_OBJECT_CLASSES: return 'real_object'
        return 'unknown'

    def _publish_detection(self, detection_type: str, label: str,
                           confidence: float = 1.0,
                           coords_3d=None,
                           yaw_deg: float = 0.0,
                           header=None):
        """
        Mantiene un único contador (self.detection_counter) y un único
        diccionario de cooldown (self._last_published_time) compartidos
        entre los tres detectores, para garantizar consistencia de IDs
        y de tiempos entre tópicos.
        """
        now  = time.time()
        last = self._last_published_time.get(label, 0.0)
        if now - last < DETECTION_COOLDOWN:
            return
        self._last_published_time[label] = now
        self.detection_counter += 1

        # ── Construir mensaje ─────────────────────────────────────────────────
        msg                = Detection()
        msg.detection_type = detection_type
        msg.label          = label
        msg.detection_id   = self.detection_counter
        msg.z              = float(coords_3d[2]) if coords_3d else -1.0
        msg.yaw            = float(yaw_deg)

        # ── Publicar y guardar ────────────────────────────────────────────────
        self._pub_detection.publish(msg)
        self._write_csv_row(msg)

        dist = f'{msg.z:.2f}m' if msg.z != -1.0 else 'sin profundidad'
        self.get_logger().info(
            f'[{detection_type}] {label} | dist={dist} | yaw={yaw_deg:+.1f}° | conf={confidence:.2f}'
        )

    # =========================================================================
    # Pipeline QR
    # =========================================================================

    def _validate_qr(self, value, pts, frame_shape):
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

    def _process_qr(self, frame, scale_x, scale_y):
        if self.qr_detector is None:
            return frame, None, None, 0.0
        now = time.time()
        age = now - self._qr_last_seen
        if age < QR_HOLD_SECS and (now - self._last_qr_time) < QR_MIN_PERIOD:
            self._draw_qr_overlay(frame, age, scale_x, scale_y)
            return frame, None, None, 0.0
        self._last_qr_time = now
        h, w     = frame.shape[:2]
        wc_scale = QR_DETECT_WIDTH / w
        small_qr = cv2.resize(frame, (QR_DETECT_WIDTH, int(h * wc_scale)),
                              interpolation=cv2.INTER_AREA) if wc_scale < 1.0 else frame
        try:
            texts, points = self.qr_detector.detectAndDecode(small_qr)
        except cv2.error as e:
            self.get_logger().warn(f'WeChat error: {e}')
            texts, points = [], []
        value, pts_ann = '', None
        for text, pt in zip(texts, points):
            if not text:
                continue
            pts_in_frame = pt / wc_scale
            if self._validate_qr(text, pts_in_frame, frame.shape):
                value, pts_ann = text, pts_in_frame
                break
        new_detection = qr_coords = None
        qr_yaw = 0.0
        if pts_ann is not None:
            self._qr_last_value = value
            self._qr_last_pts   = pts_ann
            self._qr_last_seen  = now
            if value not in self._qr_published:
                self._qr_published.add(value)
                new_detection = value
                pts_i  = np.int32(pts_ann).reshape(-1, 2)
                cx_qr  = int(pts_i[:, 0].mean() * scale_x)
                cy_qr  = int(pts_i[:, 1].mean() * scale_y)
                qr_coords = self.pixel_to_3d(cx_qr, cy_qr)
                qr_yaw    = self.pixel_to_yaw(cx_qr)
        self._draw_qr_overlay(frame, now - self._qr_last_seen, scale_x, scale_y)
        return frame, new_detection, qr_coords, qr_yaw

    def _draw_qr_overlay(self, frame, age, scale_x, scale_y):
        if age >= QR_HOLD_SECS or self._qr_last_pts is None:
            return
        pts_i        = np.int32(self._qr_last_pts).reshape(-1, 2)
        x, y, bw, bh = cv2.boundingRect(pts_i)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x+bw, y+bh), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
        label = self._qr_last_value[:40] + ('…' if len(self._qr_last_value) > 40 else '')
        cv2.putText(frame, label, (x, max(y-22, 20)),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.65, (255, 255, 255), 2)
        cx_a, cy_a = x + bw//2, y + bh//2
        rt = self.pixel_to_3d(int(cx_a * scale_x), int(cy_a * scale_y))
        cv2.circle(frame, (cx_a, cy_a), 6, (0, 0, 255), -1)
        cv2.putText(frame, f'{rt[2]:.2f}m' if rt else '--',
                    (cx_a+8, cy_a), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # =========================================================================
    # Pipeline AprilTag
    # =========================================================================

    def _process_apriltag(self, frame, scale_x, scale_y):
        now = time.time()
        if now - self._at_last_time < AT_MIN_PERIOD:
            self._draw_apriltag_overlays(frame, now, scale_x, scale_y)
            return frame, []
        self._at_last_time = now
        h, w     = frame.shape[:2]
        at_scale = AT_DETECT_WIDTH / w
        small_at = cv2.resize(frame, (AT_DETECT_WIDTH, int(h * at_scale)),
                              interpolation=cv2.INTER_AREA) if at_scale < 1.0 else frame
        if at_scale >= 1.0:
            at_scale = 1.0
        gray = cv2.cvtColor(small_at, cv2.COLOR_BGR2GRAY)
        new_detections = []
        for dict_name, detector in self._at_detectors.items():
            try:
                corners_list, ids, _ = detector.detectMarkers(gray)
            except cv2.error as e:
                self.get_logger().warn(f'AprilTag error ({dict_name}): {e}')
                continue
            if ids is None:
                continue
            for corners, tag_id in zip(corners_list, ids.flatten()):
                corners_frame = corners[0] / at_scale
                x_c, y_c, bw, bh = cv2.boundingRect(np.int32(corners_frame))
                if bw * bh < AT_MIN_AREA:
                    continue
                tag_key   = f'{dict_name}:{tag_id}'
                tag_label = str(int(tag_id))
                cx_ann    = int(corners_frame[:, 0].mean())
                cy_ann    = int(corners_frame[:, 1].mean())
                coords_3d = self.pixel_to_3d(int(cx_ann*scale_x), int(cy_ann*scale_y))
                self._at_last_detections[tag_key] = {
                    'corners': corners_frame, 'center': (cx_ann, cy_ann),
                    'seen_time': now, 'coords': coords_3d, 'label': tag_label,
                }
                if tag_key not in self._at_published:
                    self._at_published.add(tag_key)
                    at_yaw = self.pixel_to_yaw(int(cx_ann * scale_x))
                    new_detections.append((tag_key, tag_label, coords_3d, at_yaw))
        for key in list(self._at_last_detections.keys()):
            if now - self._at_last_detections[key]['seen_time'] > AT_HOLD_SECS * 10:
                del self._at_last_detections[key]
        self._draw_apriltag_overlays(frame, now, scale_x, scale_y)
        return frame, new_detections

    def _draw_apriltag_overlays(self, frame, now, scale_x, scale_y):
        for tag_key, info in self._at_last_detections.items():
            if now - info['seen_time'] > AT_HOLD_SECS:
                continue
            corners = np.int32(info['corners'])
            cx, cy  = info['center']
            overlay = frame.copy()
            cv2.fillPoly(overlay, [corners], (255, 128, 0))
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.polylines(frame, [corners], True, (255, 128, 0), 2)
            for pt in corners:
                cv2.circle(frame, tuple(pt), 4, (0, 128, 255), -1)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            x_b, y_b, bw, bh = cv2.boundingRect(corners)
            cv2.putText(frame, f'AprilTag {info["label"]}',
                        (x_b, max(y_b-22, 20)),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.55, (255, 255, 255), 2)
            rt = self.pixel_to_3d(int(cx*scale_x), int(cy*scale_y))
            cv2.putText(frame, f'{rt[2]:.2f}m' if rt else '--',
                        (cx+8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # =========================================================================
    # Callbacks de imagen
    # =========================================================================

    def _image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self.lock:
            self.latest_frame  = frame
            self.latest_header = msg.header
        self._frame_event.set()

    def _publish_annotated(self, publisher, frame, header):
        """Publica `frame` en `publisher` solo si hay suscriptores."""
        if publisher is None:
            return
        if publisher.get_subscription_count() == 0:
            return
        out_msg        = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = header
        publisher.publish(out_msg)

    # =========================================================================
    # Hilo de inferencia
    # =========================================================================

    def _inference_loop(self):
        while rclpy.ok():
            got = self._frame_event.wait(timeout=0.5)
            if not got:
                continue
            self._frame_event.clear()

            if not self._is_active():
                continue

            with self.lock:
                frame  = self.latest_frame
                header = self.latest_header

            if frame is None:
                continue

            orig_h, orig_w = frame.shape[:2]
            scale_x = orig_w / 640.0
            scale_y = orig_h / 360.0

            small = cv2.resize(frame, (640, 360))

            run_yolo = (self._pub_annotated_yolo is not None
                        and self._pub_annotated_yolo.get_subscription_count() > 0)
            run_qr   = (self._pub_annotated_qr is not None
                        and self._pub_annotated_qr.get_subscription_count() > 0)
            run_at   = (self._pub_annotated_at is not None
                        and self._pub_annotated_at.get_subscription_count() > 0)

            # ── YOLO ──────────────────────────────────────────────────────────
            if run_yolo and self.model is not None:
                results = self.model(
                    small, conf=self.confidence_threshold, imgsz=640, verbose=False
                )[0]
                annotated_yolo = results.plot()

                detected_this_frame = set()

                for box in results.boxes:
                    class_id   = int(box.cls.item())
                    confidence = float(box.conf.item())
                    name       = results.names[class_id]
                    detected_this_frame.add(name)

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx_ann = int((x1+x2)/2)
                    cy_ann = int((y1+y2)/2)

                    coords_3d = self.pixel_to_3d(int(cx_ann*scale_x), int(cy_ann*scale_y))
                    yaw = self.pixel_to_yaw(int(cx_ann * scale_x))

                    cv2.circle(annotated_yolo, (cx_ann, cy_ann), 6, (0, 0, 255), -1)
                    dist_text = f'{coords_3d[2]:.2f}m' if coords_3d else '--'
                    cv2.putText(annotated_yolo, dist_text,
                                (cx_ann+8, cy_ann),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    self.detection_counts[name] = self.detection_counts.get(name, 0) + 1
                    if self.detection_counts[name] >= self.min_confirmations:
                        det_type = self.get_detection_type(name)
                        if det_type != 'unknown':
                            self._publish_detection(
                                det_type, name, confidence, coords_3d,
                                yaw_deg=yaw,
                                header=header
                            )
                        else:
                            self.get_logger().warn(f'Clase sin mapear: {name}')
                        self.detection_counts[name] = 0

                for name in list(self.detection_counts.keys()):
                    if name not in detected_this_frame:
                        self.detection_counts[name] = 0

                self._publish_annotated(self._pub_annotated_yolo, annotated_yolo, header)
                with self.lock:
                    self.last_annotated_yolo = annotated_yolo

            # ── QR ────────────────────────────────────────────────────────────
            if run_qr and self.qr_detector is not None:
                annotated_qr = small.copy()
                annotated_qr, qr_value, qr_coords, qr_yaw = self._process_qr(
                    annotated_qr, scale_x, scale_y
                )
                if qr_value:
                    self._publish_detection(
                        'ar_code', qr_value, 1.0, qr_coords,
                        yaw_deg=qr_yaw,
                        header=header,
                    )

                self._publish_annotated(self._pub_annotated_qr, annotated_qr, header)
                with self.lock:
                    self.last_annotated_qr = annotated_qr

            # ── AprilTag ──────────────────────────────────────────────────────
            if run_at:
                annotated_at = small.copy()
                annotated_at, at_detections = self._process_apriltag(
                    annotated_at, scale_x, scale_y
                )
                for _, tag_label, at_coords, at_yaw in at_detections:
                    self._publish_detection(
                        'ar_code', tag_label, 1.0, at_coords,
                        yaw_deg=at_yaw,
                        header=header,
                    )

                self._publish_annotated(self._pub_annotated_at, annotated_at, header)
                with self.lock:
                    self.last_annotated_at = annotated_at

# ── Punto de entrada ──────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MazeNode()
    try:
        rclpy.spin(node)
    finally:
        if node.csv_file is not None:
            node.csv_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()