"""
vision_app.py
=============
Aplicación de escritorio (ventana emergente) para probar los distintos
modos de visión (YOLO, QR, Detección de movimiento) usando una cámara
Intel RealSense a través de pyrealsense2.

Se elimina toda dependencia de ROS2 / rclpy: la app corre de forma
standalone con una interfaz Tkinter que muestra el video en vivo y
tiene un botón por cada modo de visión.

Dependencias (instalar con pip):
    pip install pyrealsense2 opencv-python pillow numpy ultralytics
"""

import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image, ImageTk
from ultralytics import YOLO


# ===========================================================================
# Constantes de configuración
# ===========================================================================

# ── YOLO ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH = "/home/angel/NXL_Robocup/src/nixito_perception/modelos/Robocup_NXL_V2.pt"
YOLO_CONF       = 0.5
YOLO_IMGSZ      = 480

# ── QR (WeChatQRCode) ────────────────────────────────────────────────────────
QR_MODEL_DIR    = "/home/angel/NXL_Robocup/src/nixito_perception/drivers/qr_models"
QR_HOLD_SECS    = 0.1
QR_MIN_AREA     = 100
QR_MAX_RATIO    = 0.50
QR_MIN_PERIOD   = 0.05     # s → ~20 fps máximo para detección activa
QR_DETECT_WIDTH = 640      # El modelo WeChat trabaja bien a esta resolución

# ── Detección de movimiento ──────────────────────────────────────────────────
MOV_MIN_AREA        = 200
FRAME_BUFFER_MAXLEN = 15

# ── Cámara RealSense ─────────────────────────────────────────────────────────
CAM_WIDTH  = 640
CAM_HEIGHT = 480
CAM_FPS    = 30

# ── GUI ───────────────────────────────────────────────────────────────────────
GUI_REFRESH_MS = 15   # periodo del bucle de actualización de la ventana


# ===========================================================================
# Captura de cámara (pyrealsense2) en un hilo aparte
# ===========================================================================

class RealSenseCamera:
    """Hilo dedicado a leer frames de color de una cámara RealSense."""

    def __init__(self, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS):
        self._pipeline = rs.pipeline()
        self._config   = rs.config()
        self._config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self._latest_frame = None
        self._lock          = threading.Lock()
        self._running       = False
        self._thread         = None

    def start(self) -> None:
        self._pipeline.start(self._config)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                frames       = self._pipeline.wait_for_frames(timeout_ms=1000)
                color_frame  = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
                with self._lock:
                    self._latest_frame = frame
            except RuntimeError:
                # timeout esperando frames — seguimos intentando
                continue

    def get_latest_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self._pipeline.stop()
        except RuntimeError:
            pass


# ===========================================================================
# Aplicación principal
# ===========================================================================

class VisionApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Nixito — Visión")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Estado de modelos / modo activo ─────────────────────────────────
        self.active_mode: str | None = None   # "yolo" | "qr" | "movement" | None

        self.model        = None
        self.model_loaded = False
        self.qr_detector  = None   # cv2.wechat_qrcode_WeChatQRCode

        from collections import deque
        self.frame_buffer = deque(maxlen=FRAME_BUFFER_MAXLEN)
        self.kernel        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.tamano_roi     = 150

        self._last_qr_time  = 0.0
        self._qr_last_seen  = 0.0
        self._qr_last_value = ""
        self._qr_last_pts   = None

        # ── Cámara ───────────────────────────────────────────────────────────
        self.camera = RealSenseCamera()
        self.camera.start()

        # ── Construcción de la interfaz ────────────────────────────────────
        self._build_ui()

        # ── Arranca el bucle de refresco ────────────────────────────────────
        self.root.after(GUI_REFRESH_MS, self._update_frame)

    # -----------------------------------------------------------------------
    # Interfaz gráfica
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        video_frame = ttk.Frame(self.root)
        video_frame.pack(padx=8, pady=8)

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()

        controls = ttk.Frame(self.root)
        controls.pack(pady=(0, 8))

        ttk.Button(controls, text="YOLO", command=lambda: self._set_mode("yolo")).grid(
            row=0, column=0, padx=4)
        ttk.Button(controls, text="QR", command=lambda: self._set_mode("qr")).grid(
            row=0, column=1, padx=4)
        ttk.Button(controls, text="Movimiento", command=lambda: self._set_mode("movement")).grid(
            row=0, column=2, padx=4)
        ttk.Button(controls, text="Detener", command=lambda: self._set_mode(None)).grid(
            row=0, column=3, padx=4)

        self.status_var = tk.StringVar(value="Modo: ninguno (idle)")
        ttk.Label(self.root, textvariable=self.status_var).pack(pady=(0, 8))

    # -----------------------------------------------------------------------
    # Gestión de modos / modelos
    # -----------------------------------------------------------------------

    def _set_mode(self, mode: str | None) -> None:
        self.active_mode = mode
        if mode is not None:
            self._load_models_for_mode(mode)
        self.status_var.set(f"Modo: {mode if mode else 'ninguno (idle)'}")

    def _load_models_for_mode(self, mode: str) -> None:
        if mode == "yolo" and not self.model_loaded:
            print("Cargando modelo YOLO …")
            t0 = time.time()
            self.model = YOLO(YOLO_MODEL_PATH)
            self.model_loaded = True
            print(f"YOLO cargado en {time.time() - t0:.1f} s")

        elif mode == "qr" and self.qr_detector is None:
            print("Inicializando WeChatQRCode …")
            t0 = time.time()
            try:
                d  = f"{QR_MODEL_DIR}/detect.prototxt"
                dc = f"{QR_MODEL_DIR}/detect.caffemodel"
                s  = f"{QR_MODEL_DIR}/sr.prototxt"
                sc = f"{QR_MODEL_DIR}/sr.caffemodel"
                self.qr_detector = cv2.wechat_qrcode_WeChatQRCode(d, dc, s, sc)
                print(f"WeChatQRCode cargado con modelos CNN en {time.time() - t0:.1f} s")
            except Exception as e:
                print(
                    f"No se pudieron cargar modelos WeChat ({e}). "
                    "Usando modo sin modelos — descarga los .prototxt/.caffemodel "
                    f"en {QR_MODEL_DIR} para mejor detección."
                )
                self.qr_detector = cv2.wechat_qrcode_WeChatQRCode()

        elif mode == "movement":
            # No requiere carga de modelo pesado, solo reinicia el buffer
            self.frame_buffer.clear()

    def _unload_models(self) -> None:
        if self.model_loaded:
            print("Descargando modelo YOLO …")
            self.model = None
            self.model_loaded = False
        self.qr_detector = None

    # -----------------------------------------------------------------------
    # Bucle principal de actualización (llamado por root.after)
    # -----------------------------------------------------------------------

    def _update_frame(self) -> None:
        frame = self.camera.get_latest_frame()
        if frame is not None:
            if self.active_mode == "yolo" and self.model_loaded:
                frame = self._process_yolo(frame)
            elif self.active_mode == "qr" and self.qr_detector is not None:
                frame = self._process_qr(frame)
            elif self.active_mode == "movement":
                frame = self._process_movement(frame)

            self._display_frame(frame)

        self.root.after(GUI_REFRESH_MS, self._update_frame)

    def _display_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk   # evita que el GC borre la imagen
        self.video_label.configure(image=imgtk)

    # -----------------------------------------------------------------------
    # Pipeline YOLO
    # -----------------------------------------------------------------------

    def _process_yolo(self, frame: np.ndarray) -> np.ndarray:
        t0      = time.time()
        results = self.model(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False)[0]
        frame   = results.plot()
        cv2.putText(frame, f"YOLO Active | Latency: {(time.time() - t0) * 1000:.1f} ms",
                    (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
        return frame

    # -----------------------------------------------------------------------
    # Pipeline QR — WeChatQRCode
    # -----------------------------------------------------------------------

    def _validate_qr(self, value: str, pts: np.ndarray, frame_shape: tuple) -> bool:
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
        cv2.putText(frame, f"{status} | {(time.time() - t0) * 1000:.1f} ms",
                    (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 0, 255), 1)

    def _process_qr(self, frame: np.ndarray) -> np.ndarray:
        t0  = time.time()
        now = t0
        age = now - self._qr_last_seen

        if age < QR_HOLD_SECS and (now - self._last_qr_time) < QR_MIN_PERIOD:
            self._draw_qr_result(frame, age, t0)
            return frame

        self._last_qr_time = now

        h, w  = frame.shape[:2]
        scale = QR_DETECT_WIDTH / w
        if scale < 1.0:
            small = cv2.resize(frame, (QR_DETECT_WIDTH, int(h * scale)),
                                interpolation=cv2.INTER_AREA)
        else:
            small = frame

        try:
            texts, points = self.qr_detector.detectAndDecode(small)
        except cv2.error as e:
            print(f"WeChat detectAndDecode error: {e}")
            texts, points = [], []

        value, pts = "", None
        for text, pt in zip(texts, points):
            if not text:
                continue
            pts_full = pt / scale
            if self._validate_qr(text, pts_full, frame.shape):
                value = text
                pts   = pts_full
                break

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
            if cv2.contourArea(contour) > MOV_MIN_AREA:
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
    # Cierre
    # -----------------------------------------------------------------------

    def _on_close(self) -> None:
        self._unload_models()
        self.camera.stop()
        self.root.destroy()


# ===========================================================================
# Punto de entrada
# ===========================================================================

def main() -> None:
    root = tk.Tk()
    VisionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()