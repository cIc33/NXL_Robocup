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

QR_MODEL_DIR = '/home/angel/NXL_Robocup/src/nixito_perception/drivers/qr_models'

PKG_DIR      = Path('/home/angel/NXL_Robocup/src/nixito_perception/nixito_perception/csv')
MISSION_FILE = PKG_DIR / 'mission.txt'
CSV_DIR      = PKG_DIR / 'csv'

# Intervalo (s) con el que se revisan los conteos de suscriptores
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

        model_path               = self.get_parameter('model_path').value
        image_topic              = self.get_parameter('image_topic').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.min_confirmations   = self.get_parameter('min_confirmations').value

        # ── Modelo YOLO ───────────────────────────────────────────────────────
        self.model = YOLO(model_path)
        self.get_logger().info(f'Modelo cargado: {model_path}')

        # ── Estado interno ────────────────────────────────────────────────────
        self.bridge              = CvBridge()
        self.detection_counts    = {}
        self.detection_counter   = 0
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

        # ── CSV ───────────────────────────────────────────────────────────────
        self.csv_file, self.csv_writer = self._init_csv()

        # ── Publicadores ──────────────────────────────────────────────────────
        self.pub       = self.create_publisher(String, 'detection/name',      10)
        self.image_pub = self.create_publisher(Image,  'detection/annotated', 10)

        # ── Suscripciones de sensor ───────────────────────────────────────────
        self.create_subscription(Image,      image_topic,                                  self.image_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',           self.info_callback,  10)
        self.create_subscription(Image,      '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)

        # ── Timer: revisión de suscriptores ───────────────────────────────────
        # Activa o desactiva el procesamiento según si alguien está escuchando.
        self._processing_active = False
        self.create_timer(TIMER_CHECK_SUBS, self._check_subscribers)

        # ── Hilo de inferencia ────────────────────────────────────────────────
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()

        self.get_logger().info('Nodo iniciado — esperando suscriptores …')

    # ── Revisión de suscriptores ──────────────────────────────────────────────

    def _check_subscribers(self) -> None:
        """
        Activa el procesamiento si algún tópico de salida tiene suscriptores;
        lo desactiva (y resetea contadores) cuando no hay ninguno.
        Refleja la lógica de _check_subscribers de VisionNode.
        """
        has_subs = (
            self.pub.get_subscription_count()       > 0 or
            self.image_pub.get_subscription_count() > 0
        )

        if has_subs and not self._processing_active:
            self._processing_active = True
            self.get_logger().info('Suscriptor detectado — procesamiento ACTIVO')

        elif not has_subs and self._processing_active:
            self._processing_active = False
            # Resetear contadores para no arrastrar detecciones antiguas
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

        # ── Solo publicar en detection/name si hay suscriptores ───────────────
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

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        with self.lock:
            self.latest_frame  = frame
            self.latest_header = msg.header
            annotated          = self.last_annotated

        # ── Solo publicar el frame si hay suscriptores en detection/annotated ─
        if self.image_pub.get_subscription_count() > 0:
            annotated_msg        = self.bridge.cv2_to_imgmsg(
                annotated if annotated is not None else frame, encoding='bgr8')
            annotated_msg.header = msg.header
            self.image_pub.publish(annotated_msg)

    # ── Hilo de inferencia ────────────────────────────────────────────────────

    def inference_loop(self):
        while rclpy.ok():
            # Saltar inferencia si nadie escucha — ahorra CPU/GPU
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

            small     = cv2.resize(frame, (640, 360))
            clean_small = small.copy() 
            results   = self.model(small, conf=self.confidence_threshold, imgsz=320, verbose=False)[0]
            annotated = results.plot()

            detected_this_frame = set()

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

            annotated, qr_value, qr_coords = self._process_qr(annotated, scale_x, scale_y)

            if qr_value:
                z3d = qr_coords[2] if qr_coords else None
                self._publish_detection('ar_code', qr_value, x=0, y=0, z=z3d)
                self.get_logger().info(f'QR detectado: {qr_value}')

            with self.lock:
                self.last_annotated = annotated


# ── Punto de entrada ──────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.csv_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()