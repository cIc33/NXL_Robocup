import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
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


HAZMAT_CLASSES = {
    '1.1 Explosives', '1.5 Blasting Agents', '2 Flamable gas',
    '2 Non-Flamable gas', '2 Oxygen', '3 Fuel Oil',
    '4 Dangerous when wet', '4 Flammable solid', '4 Spontaneously Combustible',
    '5.1 Oxidizer', '5.2 Organic Peroxide', '6 Infectious Substance',
    '6 Inhalation hazard', '6 Poison', '7 Radioactive', '8 Corrosive',
}

REAL_OBJECT_CLASSES = {
    'Backpack', 'fire_extinguisher', 'gas tank', 'helmet',
}

YEAR                  = '2026'
TEAM_NAME             = 'NIXITO'
COUNTRY               = 'Mexico'
ROBOT                 = 'Nixito'
MODE                  = 'T'
QR_HOLD_SECS          = 2.0
DETECTION_COOLDOWN    = 20.0

PKG_DIR      = Path('/home/nixito/nixito_robot/src/detector')
MISSION_FILE = PKG_DIR / 'mission.txt'
CSV_DIR      = PKG_DIR / 'csv'


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')

        self.declare_parameter('model_path', '/home/nixito/nixito_robot/src/realsense/realsense/best.pt')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('confidence_threshold', 0.75)
        self.declare_parameter('min_confirmations', 5)

        model_path = self.get_parameter('model_path').value
        image_topic = self.get_parameter('image_topic').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.min_confirmations = self.get_parameter('min_confirmations').value

        self.model = YOLO(model_path)
        self.get_logger().info(f'Modelo cargado: {model_path}')

        self.bridge = CvBridge()
        self.detection_counts = {}
        self.detection_counter = 0
        self._last_published_time = {}  # nombre -> timestamp última publicación

        self.latest_frame = None
        self.latest_header = None
        self.last_annotated = None
        self.lock = threading.Lock()

        # QR
        self.qr_detector    = cv2.QRCodeDetector()
        self._clahe         = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._gamma_lut     = self._build_gamma_lut(gamma=1.5)
        self._qr_last_value = ''
        self._qr_last_pts   = None
        self._qr_last_seen  = 0.0
        self._qr_published  = set()

        # CSV
        self.csv_file, self.csv_writer = self._init_csv()

        self.pub = self.create_publisher(String, 'detection/name', 10)
        self.image_pub = self.create_publisher(Image, 'detection/annotated', 10)
        self.create_subscription(Image, image_topic, self.image_callback, 10)

        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()

        self.get_logger().info('Nodo iniciado')

    # ── CSV ──────────────────────────────────────────────────────────────────

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

        filename = f'RoboCup{YEAR}-{TEAM_NAME}-Prelim{mission_num}-{start_date}-{start_time_file}-pois.csv'
        filepath = CSV_DIR / filename

        f = open(filepath, 'w', newline='')
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

    def write_csv_row(self, detection_type: str, name: str):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.csv_writer.writerow([
            self.detection_counter,
            timestamp,
            detection_type,
            name,
            '', '', '',
            ROBOT,
            MODE,
        ])
        self.csv_file.flush()

    # ── Detección ────────────────────────────────────────────────────────────

    def get_detection_type(self, name: str) -> str:
        if name in HAZMAT_CLASSES:
            return 'hazmat_sign'
        elif name in REAL_OBJECT_CLASSES:
            return 'real_object'
        return 'unknown'

    def _publish_detection(self, detection_type: str, name: str, confidence: float = 0.0):
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
        self.write_csv_row(detection_type, name)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        out = String()
        out.data = (
            f'{self.detection_counter},'
            f'{timestamp},'
            f'{detection_type},'
            f'"{name}",'
            f' , , ,'
            f'{ROBOT},'
            f'{MODE}'
        )
        self.pub.publish(out)
        self.get_logger().info(f'Publicado: {out.data}')

    # ── Pipeline QR ──────────────────────────────────────────────────────────

    def _build_gamma_lut(self, gamma: float = 1.5) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ], dtype=np.uint8)
        return table

    def _preprocess_for_qr(self, frame: np.ndarray):
        h, w  = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2))
        scale = (w / (w // 2), h / (h // 2))

        gray      = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray      = self._clahe.apply(gray)
        gray      = cv2.bilateralFilter(gray, 5, 75, 75)
        blur      = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpened = cv2.addWeighted(gray, 2.5, blur, -1.5, 0)

        thresh = cv2.adaptiveThreshold(
            sharpened, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            11, 2,
        )

        gamma = cv2.LUT(gray, self._gamma_lut) if np.mean(gray) < 80 else sharpened
        return [frame, thresh, cv2.bitwise_not(thresh), gamma], scale

    def _validate_qr(self, value: str, pts, frame_shape) -> bool:
        if not value or not value.strip():
            return False
        pts_i       = np.int32(pts).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(pts_i)
        area        = w * h
        fh, fw      = frame_shape[:2]
        if area < 400 or area / (fw * fh) > 0.5:
            return False
        if h == 0 or abs(w / h - 1.0) > 0.3:
            return False
        if value.count('\x00') / len(value) > 0.1:
            return False
        return True

    def _process_qr(self, frame: np.ndarray) -> tuple:
        t0 = time.time()
        variants, (sx, sy) = self._preprocess_for_qr(frame)
        value, pts = '', None

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
                value, pts = v, p * ([sx, sy] if i > 0 else [1, 1])
                break

        new_detection = None

        if pts is not None and self._validate_qr(value, pts, frame.shape):
            self._qr_last_value = value
            self._qr_last_pts   = pts
            self._qr_last_seen  = time.time()

            if value not in self._qr_published:
                self._qr_published.add(value)
                new_detection = value

        age = time.time() - self._qr_last_seen
        if age < QR_HOLD_SECS and self._qr_last_pts is not None:
            pts_i         = np.int32(self._qr_last_pts).reshape(-1, 2)
            x, y, bw, bh  = cv2.boundingRect(pts_i)

            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            label = self._qr_last_value[:40] + ('…' if len(self._qr_last_value) > 40 else '')
            cv2.putText(frame, label, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.65, (255, 255, 255), 2)

        status = 'QR Active' if age < QR_HOLD_SECS else 'Scanning'
        cv2.putText(frame, f'{status} | {(time.time() - t0) * 1000:.1f} ms',
                    (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 0, 255), 1)

        return frame, new_detection

    # ── Callbacks ────────────────────────────────────────────────────────────

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        with self.lock:
            self.latest_frame  = frame
            self.latest_header = msg.header
            annotated          = self.last_annotated

        if annotated is not None:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        else:
            annotated_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

        annotated_msg.header = msg.header
        self.image_pub.publish(annotated_msg)

    def inference_loop(self):
        while rclpy.ok():
            with self.lock:
                frame = self.latest_frame

            if frame is None:
                continue

            small   = cv2.resize(frame, (640, 360))
            results = self.model(small, conf=self.confidence_threshold, imgsz=320, verbose=False)[0]
            annotated = results.plot()

            detected_this_frame = set()

            for box in results.boxes:
                class_id   = int(box.cls.item())
                confidence = float(box.conf.item())
                name       = results.names[class_id]
                detected_this_frame.add(name)

                self.detection_counts[name] = self.detection_counts.get(name, 0) + 1
                count = self.detection_counts[name]

                if count >= self.min_confirmations:
                    detection_type = self.get_detection_type(name)
                    if detection_type != 'unknown':
                        self._publish_detection(detection_type, name, confidence)
                    else:
                        self.get_logger().warn(f'Clase sin mapear: {name}')
                    self.detection_counts[name] = 0

            for name in list(self.detection_counts.keys()):
                if name not in detected_this_frame:
                    self.detection_counts[name] = 0

            annotated, qr_value = self._process_qr(annotated)

            if qr_value:
                self._publish_detection('ar_code', qr_value)
                self.get_logger().info(f'QR detectado: {qr_value}')

            with self.lock:
                self.last_annotated = annotated

            cv2.imshow('Detector', annotated)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.csv_file.close()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()