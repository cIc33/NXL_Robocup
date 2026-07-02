

import glob
import os
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

# ── Cámara térmica TC001 ──────────────────────────────────────────────────────
TC001_VENDOR_ID  = '0bda'
TC001_PRODUCT_ID = '5830'
TC001_WIDTH      = 256
TC001_HEIGHT     = 192
TC001_SCALE      = 3
TC001_THRESHOLD  = 2   # °C sobre/bajo el promedio para marcar puntos caliente/frío

# ── GUI ───────────────────────────────────────────────────────────────────────
GUI_REFRESH_MS = 15   # periodo del bucle de actualización de la ventana


# ===========================================================================
# Utilidades TC001
# ===========================================================================

def find_tc001_device():
    """
    Busca el dispositivo /dev/videoX de la cámara Topdon TC001
    comparando Vendor ID (0x0bda) y Product ID (0x5830) en sysfs.
    Retorna la ruta del dispositivo o None si no se encuentra.
    """
    for video_path in sorted(glob.glob('/sys/class/video4linux/video*')):
        try:
            real_path = os.path.realpath(video_path)
            parts = real_path.split('/')
            for i in range(len(parts), 0, -1):
                parent = '/'.join(parts[:i])
                vendor_file = os.path.join(parent, 'idVendor')
                product_file = os.path.join(parent, 'idProduct')
                if os.path.exists(vendor_file) and os.path.exists(product_file):
                    with open(vendor_file) as vf, open(product_file) as pf:
                        vendor = vf.read().strip()
                        product = pf.read().strip()
                    if vendor == TC001_VENDOR_ID and product == TC001_PRODUCT_ID:
                        dev_name = os.path.basename(video_path)
                        return f'/dev/{dev_name}'
                    break
        except (OSError, PermissionError):
            continue

    return None


# ===========================================================================
# Captura de cámara RealSense (pyrealsense2) en un hilo aparte
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
# Captura + procesamiento de cámara térmica TC001 en un hilo aparte
# ===========================================================================

class ThermalCamera:
    """
    Hilo dedicado a leer frames crudos de la Topdon TC001 y convertirlos
    directamente en el mapa de calor (heatmap) ya anotado con
    temperatura central, máxima y mínima — listo para mostrar en la GUI.
    """

    def __init__(self, device_path=None, width=TC001_WIDTH, height=TC001_HEIGHT,
                 scale=TC001_SCALE):
        self.width      = width
        self.height     = height
        self.scale      = scale
        self.new_width  = width * scale
        self.new_height = height * scale

        self._device_path = device_path or find_tc001_device()
        if self._device_path is None:
            raise RuntimeError(
                "No se encontró la cámara TC001. Verifica que esté conectada."
            )

        self._cap = cv2.VideoCapture(self._device_path, cv2.CAP_V4L)
        self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
        if not self._cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara TC001 en {self._device_path}")

        self._latest_frame = None
        self._lock          = threading.Lock()
        self._running       = False
        self._thread         = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue
            heatmap = self._process(frame)
            with self._lock:
                self._latest_frame = heatmap

    def _process(self, frame: np.ndarray) -> np.ndarray:
        # Separar imagen visible y datos de temperatura
        imdata, thdata = np.array_split(frame, 2)

        # --- Temperatura del centro ---
        hi = thdata[96][128][0]
        lo = thdata[96][128][1]
        rawtemp = hi + (lo * 256)
        center_temp = round((rawtemp / 64) - 273.15, 2)

        # --- Temperatura máxima ---
        lomax = thdata[..., 1].max()
        posmax = thdata[..., 1].argmax()
        mcol, mrow = divmod(posmax, self.width)
        himax = thdata[mcol][mrow][0]
        maxtemp = round(((himax + lomax * 256) / 64) - 273.15, 2)

        # --- Temperatura mínima ---
        lomin = thdata[..., 1].min()
        posmin = thdata[..., 1].argmin()
        lcol, lrow = divmod(posmin, self.width)
        himin = thdata[lcol][lrow][0]
        mintemp = round(((himin + lomin * 256) / 64) - 273.15, 2)

        # --- Temperatura promedio ---
        loavg = thdata[..., 1].mean()
        hiavg = thdata[..., 0].mean()
        avgtemp = round(((loavg * 256 + hiavg) / 64) - 273.15, 2)

        # --- Procesar imagen visual ---
        bgr = cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
        bgr = cv2.resize(bgr, (self.new_width, self.new_height), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_INFERNO)
        heatmap = cv2.rotate(heatmap, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cx = heatmap.shape[1] // 2   # ← usar shape en vez de new_width/new_height
        cy = heatmap.shape[0] // 2

        # --- Cruz central ---
        cv2.line(heatmap, (cx, cy + 20), (cx, cy - 20), (255, 255, 255), 2)
        cv2.line(heatmap, (cx + 20, cy), (cx - 20, cy), (255, 255, 255), 2)
        cv2.line(heatmap, (cx, cy + 20), (cx, cy - 20), (0, 0, 0), 1)
        cv2.line(heatmap, (cx + 20, cy), (cx - 20, cy), (0, 0, 0), 1)

        # Temperatura centro
        cv2.putText(heatmap, f'{center_temp} C', (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(heatmap, f'{center_temp} C', (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # --- Punto más caliente ---
        if maxtemp > avgtemp + TC001_THRESHOLD:
            px, py = mrow * self.scale, mcol * self.scale
            cv2.circle(heatmap, (px, py), 5, (0, 0, 0), 2)
            cv2.circle(heatmap, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(heatmap, f'{maxtemp} C', (px + 10, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(heatmap, f'{maxtemp} C', (px + 10, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # --- Punto más frío ---
        if mintemp < avgtemp - TC001_THRESHOLD:
            px, py = lrow * self.scale, lcol * self.scale
            cv2.circle(heatmap, (px, py), 5, (0, 0, 0), 2)
            cv2.circle(heatmap, (px, py), 5, (255, 0, 0), -1)
            cv2.putText(heatmap, f'{mintemp} C', (px + 10, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(heatmap, f'{mintemp} C', (px + 10, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        return heatmap

    def get_latest_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self._cap.release()
        except Exception:
            pass


# ===========================================================================
# Aplicación principal
# ===========================================================================

class VisionApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Nixito — Visión")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Estado de modelos / modo activo (solo aplica a RealSense) ───────
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

        # ── Cámara RealSense ─────────────────────────────────────────────────
        self.camera = RealSenseCamera()
        self.camera.start()

        # ── Cámara térmica TC001 (opcional: si no está conectada, seguimos) ─
        self.thermal_camera = None
        self._thermal_error = ""
        try:
            self.thermal_camera = ThermalCamera()
            self.thermal_camera.start()
        except RuntimeError as e:
            self._thermal_error = str(e)
            print(f"[TC001] {e} — el panel térmico quedará vacío.")

        # ── Construcción de la interfaz ────────────────────────────────────
        self._build_ui()

        # ── Arranca el bucle de refresco ────────────────────────────────────
        self.root.after(GUI_REFRESH_MS, self._update_frame)

    # -----------------------------------------------------------------------
    # Interfaz gráfica
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Contenedor con ambos videos, uno al lado del otro ───────────────
        videos_frame = ttk.Frame(self.root)
        videos_frame.pack(padx=8, pady=8)

        rs_frame = ttk.LabelFrame(videos_frame, text="RealSense")
        rs_frame.grid(row=0, column=0, padx=(0, 8))
        self.video_label = ttk.Label(rs_frame)
        self.video_label.pack()

        thermal_frame = ttk.LabelFrame(videos_frame, text="TC001 (térmica)")
        thermal_frame.grid(row=0, column=1)
        self.thermal_label = ttk.Label(thermal_frame)
        self.thermal_label.pack()

        if self.thermal_camera is None:
            # Placeholder visible mientras no haya cámara térmica conectada
            self.thermal_label.configure(
                text=f"TC001 no disponible\n{self._thermal_error}",
                anchor="center",
                justify="center",
                width=40,
            )

        # ── Controles (solo afectan al modo de la RealSense) ────────────────
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
    # Gestión de modos / modelos (RealSense)
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
        # ── RealSense (con el modo de visión seleccionado) ──────────────────
        frame = self.camera.get_latest_frame()
        if frame is not None:
            if self.active_mode == "yolo" and self.model_loaded:
                frame = self._process_yolo(frame)
            elif self.active_mode == "qr" and self.qr_detector is not None:
                frame = self._process_qr(frame)
            elif self.active_mode == "movement":
                frame = self._process_movement(frame)

            self._display_frame(self.video_label, frame)

        # ── TC001 (heatmap ya viene procesado desde el hilo de captura) ─────
        if self.thermal_camera is not None:
            thermal_frame = self.thermal_camera.get_latest_frame()
            if thermal_frame is not None:
                self._display_frame(self.thermal_label, thermal_frame)

        self.root.after(GUI_REFRESH_MS, self._update_frame)

    def _display_frame(self, label: ttk.Label, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk   # evita que el GC borre la imagen
        label.configure(image=imgtk)

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
        if self.thermal_camera is not None:
            self.thermal_camera.stop()
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