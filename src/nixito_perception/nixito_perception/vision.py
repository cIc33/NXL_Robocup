import time
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.lifecycle import LifecycleNode, Publisher, State, TransitionCallbackReturn
from sensor_msgs.msg import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YOLO_MODEL_PATH = "/home/sahid/nixito_robot/src/nixito_perception/NixitoS.pt"
YOLO_MIN_PERIOD  = 0.05        # seconds – max throughput ~20 fps
YOLO_CONF        = 0.5
YOLO_IMGSZ       = 320

THERMAL_ZOOM     = 1.5
THERMAL_T_MIN    = 5.0         # °C  – lower bound for normalisation
THERMAL_T_MAX    = 50.0        # °C  – upper bound for normalisation

QR_HOLD_SECS = 0.8   # segundos que persiste el resultado sin redetección

MOV_HISTORY      = 500
MOV_VAR_THRESHOLD = 8
MOV_LEARNING_RATE = 0.002
MOV_DIFF_THRESHOLD = 6         # pixel-diff threshold for temporal mask
MOV_MIN_AREA     = 500         # px² – discard tiny contours
MOV_GROUP_EPS    = 0.3         # groupRectangles overlap tolerance
MOV_WARMUP_LEN   = 15          # frames needed before detection starts

FRAME_BUFFER_MAXLEN = 15

# Topics
TOPIC_INPUT_RGB     = "/brazo/image_raw"
TOPIC_INPUT_THERMAL = "/thermal/image"
TOPIC_OUTPUTS = {
    "yolo":     ("vision/yolo",     10),
    "qr":       ("vision/qr",       10),
    "thermal":  ("vision/thermal",  10),
    "movement": ("vision/movement", 10),
}

# Timer periods (seconds)
TIMER_CHECK_SUBS = 2.0
TIMER_LOG_STATS  = 10.0


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class VisionNode(LifecycleNode):

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(self):
        super().__init__("vision_node")

        self.bridge = CvBridge()

        # Publishers – created in on_configure, stored by mode name.
        self.pubs: dict[str, Publisher] = {}

        # Runtime state
        self.active_mode: str | None = None

        # Lazy-loaded heavy resources
        self.model        = None
        self.model_loaded = False
        self.qr_detector  = None
        self.mog2         = None

        # Pre-allocated lightweight resources (cheap to keep around)
        self.last_thermal  = None
        self.frame_buffer  = deque(maxlen=FRAME_BUFFER_MAXLEN)
        self.kernel        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._clahe        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._gamma_lut    = np.array(
            [int(((i / 255.0) ** 2.0) * 255) for i in range(256)], dtype=np.uint8
        )
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=40, detectShadows=False
        )

        # YOLO rate-limiter
        self.last_yolo_time = 0.0

        # QR confirmation state
        self._qr_last_seen   = 0.0
        self._qr_last_value  = ""
        self._qr_last_pts    = None



        # Timer handles – created in on_configure
        self._timer_check_subs = None
        self._timer_stats      = None

        self.get_logger().info("VisionNode created (UNCONFIGURED)")

    # ------------------------------------------------------------------ #
    # Lifecycle callbacks                                                  #
    # ------------------------------------------------------------------ #

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Allocate lightweight resources: publishers, subscriptions, timers."""
        self.get_logger().info("Configuring node …")

        # Lifecycle publishers (activated/deactivated automatically with the node)
        for mode, (topic, qos) in TOPIC_OUTPUTS.items():
            self.pubs[mode] = self.create_lifecycle_publisher(Image, topic, qos)
            self.get_logger().info(f"  Publisher ready: {topic}")

        # Subscriptions
        self.sub_rgb = self.create_subscription(
            Image, TOPIC_INPUT_RGB, self._main_loop, 1
        )
        self.sub_thermal = self.create_subscription(
            Image, TOPIC_INPUT_THERMAL, self._thermal_callback, 10
        )

        # Monitoring timers
        self._timer_check_subs = self.create_timer(TIMER_CHECK_SUBS, self._check_subscribers)
        self._timer_stats      = self.create_timer(TIMER_LOG_STATS,  self._log_stats)

        self.get_logger().info("Node configured – ready to activate")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Node ACTIVE – waiting for subscribers …")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Node INACTIVE – releasing heavy resources …")
        self._unload_models()
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """Destroy subscriptions and timers, return to UNCONFIGURED."""
        self.get_logger().info("Cleaning up resources …")
        self.destroy_subscription(self.sub_rgb)
        self.destroy_subscription(self.sub_thermal)
        self.destroy_timer(self._timer_check_subs)
        self.destroy_timer(self._timer_stats)
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down …")
        self._unload_models()
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------ #
    # Model management                                                     #
    # ------------------------------------------------------------------ #

    def _load_models_for_mode(self, mode: str) -> None:
        """Load the heavy model required by *mode* (no-op if already loaded)."""
        if mode == "yolo" and not self.model_loaded:
            self.get_logger().info("Loading YOLO model …")
            t0 = time.time()
            self.model = YOLO(YOLO_MODEL_PATH)
            self.model.to('cuda')
            self.model_loaded = True
            self.get_logger().info(f"YOLO loaded in {time.time() - t0:.1f} s")

        elif mode == "qr" and self.qr_detector is None:
            self.get_logger().info("Initialising QR detector …")
            self.qr_detector = cv2.QRCodeDetector()

        elif mode == "movement" and self.mog2 is None:
            self.get_logger().info("Initialising MOG2 background subtractor …")
            self.mog2 = cv2.createBackgroundSubtractorMOG2(
                history=MOV_HISTORY,
                varThreshold=MOV_VAR_THRESHOLD,
                detectShadows=False,
            )

    def _unload_models(self) -> None:
        """Free memory held by heavy models."""
        if self.model_loaded:
            self.get_logger().info("Unloading YOLO model …")
            self.model = None
            self.model_loaded = False

        self.qr_detector = None
        self.mog2 = None

    # ------------------------------------------------------------------ #
    # Subscriber monitoring                                                #
    # ------------------------------------------------------------------ #

    def _check_subscribers(self) -> None:
        """Switch active mode to match the first topic that has a subscriber."""
        if not self._is_active():
            return

        for mode, pub in self.pubs.items():
            if pub.get_subscription_count() > 0:
                if self.active_mode != mode:
                    self.get_logger().info(f"Subscriber detected for mode: {mode}")
                    self._load_models_for_mode(mode)
                    self.active_mode = mode
                return  # only the first matching topic wins

        # No subscribers on any topic
        if self.active_mode is not None:
            self.get_logger().info(f"No subscribers – deactivating mode: {self.active_mode}")
            self.active_mode = None

    def _log_stats(self) -> None:
        """Periodic heartbeat log."""
        if self.active_mode:
            self.get_logger().info(f"Active mode: {self.active_mode}")
        else:
            self.get_logger().info("Idle – no active processing")

    # ------------------------------------------------------------------ #
    # Main processing loop                                                 #
    # ------------------------------------------------------------------ #

    def _main_loop(self, msg: Image) -> None:
        """Decode an incoming RGB frame and route it to the active pipeline."""
        if not self._is_active() or self.active_mode is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        processors = {
            "yolo":     (self.model_loaded,          self._process_yolo),
            "qr":       (self.qr_detector is not None, self._process_qr),
            "thermal":  (True,                        self._process_thermal),
            "movement": (self.mog2 is not None,       self._process_movement),
        }

        ready, process_fn = processors.get(self.active_mode, (False, None))
        if not ready or process_fn is None:
            return

        result = process_fn(frame)
        out_msg = self.bridge.cv2_to_imgmsg(result, "bgr8")
        self.pubs[self.active_mode].publish(out_msg)

    # ------------------------------------------------------------------ #
    # Processing pipelines                                                 #
    # ------------------------------------------------------------------ #

    # ── YOLO ─────────────────────────────────────────────────────────── #

    def _process_yolo(self, frame: np.ndarray) -> np.ndarray:
        """Run YOLOv10 segmentation; throttled to YOLO_MIN_PERIOD."""
        t0      = time.time()
        results = self.model(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False)[0]
        frame   = results.plot()
        latency = (time.time() - t0) * 1000

        cv2.putText(frame, f"YOLO Active | Latency: {latency:.1f} ms",
                    (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 255, 0), 1)
        return frame

    # ── QR ───────────────────────────────────────────────────────────── #
    def _preprocess_for_qr(self, frame: np.ndarray):
        h, w  = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2))
        scale = (w / (w // 2), h / (h // 2))

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)

        blur      = cv2.GaussianBlur(gray, (3, 3), 0)
        sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

        _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return [frame, thresh], scale
    
    def _validate_qr(self, value, pts, frame_shape):
        if not value or not value.strip():
            return False
        pts_i = np.int32(pts).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(pts_i)
        area = w * h
        fh, fw = frame_shape[:2]
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

        # ← solo esto cambia
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

        if pts is not None and self._validate_qr(value, pts, frame.shape):
            self._qr_last_value = value
            self._qr_last_pts   = pts
            self._qr_last_seen  = time.time()

        # Dibujar si hay detección vigente (reciente o actual)
        age = time.time() - self._qr_last_seen
        if age < QR_HOLD_SECS and self._qr_last_pts is not None:
            pts_i        = np.int32(self._qr_last_pts).reshape(-1, 2)
            x, y, bw, bh = cv2.boundingRect(pts_i)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            label = self._qr_last_value[:40] + ("…" if len(self._qr_last_value) > 40 else "")
            cv2.putText(frame, label, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.3, (255, 255, 255), 1)

        status = "QR Active" if age < QR_HOLD_SECS else "Scanning"
        cv2.putText(frame, f"{status} | {(time.time()-t0)*1000:.1f} ms",
                    (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (255, 0, 255), 1)
        return frame

    # ── Thermal ──────────────────────────────────────────────────────── #

    def _thermal_callback(self, msg: Image) -> None:
        """Cache the latest thermal frame (64-bit float, degrees Celsius)."""
        try:
            self.last_thermal = self.bridge.imgmsg_to_cv2(msg, desired_encoding="64FC1")
            self.last_thermal = np.roll(self.last_thermal, shift=0, axis=0)  # baja 4 filas
            self.last_thermal = np.roll(self.last_thermal, shift=-2, axis=1) 
        except Exception as exc:
            self.get_logger().warn(f"Thermal decode error: {exc}")

    def _process_thermal(self, frame: np.ndarray) -> np.ndarray:
        """Blend a false-colour thermal overlay onto the RGB frame."""
        t0 = time.time()

        if self.last_thermal is not None:
            thermal = self._normalize_thermal(self.last_thermal)
            thermal = self._virtual_zoom(thermal, THERMAL_ZOOM)
            thermal = cv2.GaussianBlur(thermal, (3, 3), 0)
            thermal = cv2.applyColorMap(thermal, cv2.COLORMAP_HSV)
            thermal = cv2.flip(thermal, 1)
            thermal = cv2.resize(thermal, (frame.shape[1], frame.shape[0]))
            frame   = cv2.addWeighted(frame, 0.7, thermal, 0.7, 0)

        latency = (time.time() - t0) * 1000
        cv2.putText(frame, f"Thermal Active | Latency: {latency:.1f} ms",
                    (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (255, 255, 255), 1)
        return frame

    # ── Movement ─────────────────────────────────────────────────────── #

    def _process_movement(self, frame: np.ndarray) -> np.ndarray:
        """
        Two-channel motion detector:
          1. Temporal diff  – accumulates abs-differences across multiple frame
             pairs (fast, medium and slow separation) to capture a wide range
             of object speeds.
          2. MOG2           – statistical background model with a slow learning
             rate so gradual movers are not absorbed into the background.

        Both masks are OR-combined, morphologically cleaned, and then grouped
        into bounding boxes.
        """
        t0 = time.time()

        # Preprocess and buffer
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        self.frame_buffer.append(gray)

        # Wait for the buffer to fill before making detections
        if len(self.frame_buffer) < self.frame_buffer.maxlen:
            cv2.putText(frame,
                        f"Warming up … {len(self.frame_buffer)}/{self.frame_buffer.maxlen}",
                        (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (255, 255, 255), 1)
            return frame

        # ── Mask 1: accumulated temporal differences ──────────────────
        # Comparing three frame pairs at different temporal distances lets us
        # detect both fast-moving (close pair) and slow-moving (distant pair)
        # objects simultaneously.
        frames = list(self.frame_buffer)
        mid    = len(frames) // 2
        pairs  = [(0, -1), (0, mid), (mid, -1)]

        accumulated = np.zeros_like(gray, dtype=np.float32)
        for i, j in pairs:
            accumulated += cv2.absdiff(frames[i], frames[j]).astype(np.float32)

        accumulated  = np.clip(accumulated / len(pairs), 0, 255).astype(np.uint8)
        _, mask_diff = cv2.threshold(accumulated, MOV_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        # ── Mask 2: MOG2 background subtractor ───────────────────────
        # Low learning rate prevents slow objects from being absorbed by
        # the background model.
        mask_mog2 = self.bg_subtractor.apply(gray, learningRate=MOV_LEARNING_RATE)

        # ── Combine, clean and find contours ─────────────────────────
        combined = cv2.bitwise_or(mask_diff, mask_mog2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  self.kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        combined = cv2.dilate(combined, self.kernel, iterations=2)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [
            cv2.boundingRect(c)
            for c in contours
            if cv2.contourArea(c) > MOV_MIN_AREA
        ]

        # Group overlapping / nearby rectangles to avoid fragmented detections
        if rects:
            rects_grouped, _ = cv2.groupRectangles(
                [list(r) for r in rects * 2],  # groupRectangles requires duplicated list
                groupThreshold=1,
                eps=MOV_GROUP_EPS,
            )
        else:
            rects_grouped = []

        for (x, y, w, h) in rects_grouped:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        latency        = (time.time() - t0) * 1000
        motion_detected = len(rects_grouped) > 0
        status         = "MOTION DETECTED" if motion_detected else "Monitoring …"
        color          = (0, 100, 255)     if motion_detected else (255, 255, 255)

        cv2.putText(frame, f"{status} | Latency: {latency:.1f} ms",
                    (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.3, color, 1)
        return frame

    # ------------------------------------------------------------------ #
    # Helper / utility methods                                             #
    # ------------------------------------------------------------------ #

    def _is_active(self) -> bool:
        """Return True when the lifecycle state machine is in the *active* state."""
        return self._state_machine.current_state[1] == "active"

    def _normalize_thermal(self, data: np.ndarray) -> np.ndarray:
        """Map thermal data (°C) to an 8-bit image using fixed temperature bounds."""
        normalized = np.clip((data - THERMAL_T_MIN) / (THERMAL_T_MAX - THERMAL_T_MIN), 0, 1) * 255
        return normalized.astype(np.uint8)

    def _virtual_zoom(self, frame: np.ndarray, factor: float) -> np.ndarray:
        """Centre-crop and upscale to simulate optical zoom."""
        if factor <= 1:
            return frame
        h, w    = frame.shape[:2]
        nw, nh  = int(w / factor), int(h / factor)
        x1, y1  = (w - nw) // 2, (h - nh) // 2
        return cv2.resize(frame[y1:y1 + nh, x1:x1 + nw], (w, h))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()