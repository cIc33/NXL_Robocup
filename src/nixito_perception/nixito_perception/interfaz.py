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
GUI_REFRESH_MS   = 15    # periodo del bucle de actualización de la ventana
OUTER_PAD        = 16    # margen exterior de la ventana
INNER_PAD        = 16    # separación entre el panel RealSense y el térmico
CONTROLS_HEIGHT  = 110   # alto reservado para la barra de botones
STATUS_HEIGHT    = 36    # alto reservado para la etiqueta de estado
LABELFRAME_CHROME = 40   # alto aproximado que añade el título del LabelFrame


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
        self.root.configure(bg="#1e1e1e")

        # ── Ventana redimensionable con el ratón (arrastrar bordes/esquinas) ─
        self.root.resizable(True, True)
        self.root.minsize(640, 480)
        # (a diferencia de -fullscreen, esto conserva los botones de
        # minimizar/maximizar/cerrar de la ventana)
        try:
            self.root.state("zoomed")          # Windows / algunos WMs de Linux
        except tk.TclError:
            try:
                self.root.attributes("-zoomed", True)   # X11 (Linux)
            except tk.TclError:
                # Último recurso: ajustar geometría al tamaño de pantalla
                self.root.geometry(
                    f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0"
                )

        # F11 sigue disponible por si en algún momento se quiere fullscreen
        # real sin bordes; Escape lo desactiva.
        self._is_fullscreen = False
        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))

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

        # ── Calcular tamaños de los paneles según la resolución de pantalla ──
        self._compute_panel_sizes()

        # ── Construcción de la interfaz ────────────────────────────────────
        self._build_ui()

        # ── Reaccionar cuando el usuario cambia el tamaño de la ventana
        #    (p. ej. al restaurarla tras minimizarla, o arrastrando el borde)
        self._resize_after_id = None
        self.root.bind("<Configure>", self._on_root_configure)

        # ── Arranca el bucle de refresco ────────────────────────────────────
        self.root.after(GUI_REFRESH_MS, self._update_frame)

    # -----------------------------------------------------------------------
    # Redimensionado de ventana (arrastrar borde, restaurar tras minimizar…)
    # -----------------------------------------------------------------------

    def _on_root_configure(self, event) -> None:
        # <Configure> se dispara también por cada widget hijo; nos interesa
        # solo cuando cambia el tamaño de la ventana principal.
        if event.widget is not self.root:
            return

        # "Debounce": esperamos a que el usuario deje de arrastrar el borde
        # antes de recalcular, para no recargar la CPU en cada píxel movido.
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(200, self._apply_new_panel_sizes)

    def _apply_new_panel_sizes(self) -> None:
        self._resize_after_id = None
        old_rs, old_th = self.rs_display_size, self.thermal_display_size
        self._compute_panel_sizes()
        if self.rs_display_size == old_rs and self.thermal_display_size == old_th:
            return   # nada cambió realmente, evitamos redibujar sin motivo

        if self.thermal_camera is None:
            thermal_w, _ = self.thermal_display_size
            self.thermal_label.configure(width=thermal_w)

    # -----------------------------------------------------------------------
    # Fullscreen opcional (F11) — la ventana arranca maximizada, no fullscreen
    # -----------------------------------------------------------------------

    def _toggle_fullscreen(self, event=None) -> None:
        self._is_fullscreen = not self._is_fullscreen
        self.root.attributes("-fullscreen", self._is_fullscreen)

    # -----------------------------------------------------------------------
    # Cálculo de tamaños (RealSense grande + térmica con el ancho restante)
    # -----------------------------------------------------------------------

    def _compute_panel_sizes(self) -> None:
        self.root.update_idletasks()
        # Usamos el tamaño real de la ventana ya maximizada (más fiable que
        # winfo_screenwidth/height, que ignora la barra de título / taskbar).
        screen_w = self.root.winfo_width()
        screen_h = self.root.winfo_height()
        if screen_w <= 1 or screen_h <= 1:
            # La ventana aún no se ha dibujado del todo: usamos el tamaño
            # de pantalla como aproximación razonable.
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()

        # Alto disponible para los paneles de vídeo, restando la barra de
        # controles, la etiqueta de estado y los márgenes/título de cada panel.
        video_area_h = (
            screen_h - CONTROLS_HEIGHT - STATUS_HEIGHT
            - 2 * OUTER_PAD - LABELFRAME_CHROME
        )
        video_area_h = max(video_area_h, 240)

        # RealSense: usa toda la altura disponible, manteniendo su relación
        # de aspecto real (4:3) → queda con un tamaño considerable.
        rs_h = video_area_h
        rs_w = int(rs_h * (CAM_WIDTH / CAM_HEIGHT))

        # Térmica: se queda con el ancho que sobra tras el panel RealSense.
        usable_w   = screen_w - 2 * OUTER_PAD - INNER_PAD
        thermal_w  = max(usable_w - rs_w, 320)
        thermal_h  = video_area_h

        self.rs_display_size      = (rs_w, rs_h)
        self.thermal_display_size = (thermal_w, thermal_h)

    # -----------------------------------------------------------------------
    # Interfaz gráfica
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 13), padding=(18, 10))
        style.configure("Exit.TButton", font=("Helvetica", 13), padding=(18, 10))
        style.configure("TLabelframe.Label", font=("Helvetica", 12, "bold"))
        style.configure("Status.TLabel", font=("Helvetica", 12))

        # ── Contenedor con ambos videos, uno al lado del otro ───────────────
        videos_frame = ttk.Frame(self.root)
        videos_frame.pack(padx=OUTER_PAD, pady=(OUTER_PAD, 8))

        rs_w, rs_h = self.rs_display_size
        thermal_w, thermal_h = self.thermal_display_size

        rs_frame = ttk.LabelFrame(videos_frame, text="RealSense")
        rs_frame.grid(row=0, column=0, padx=(0, INNER_PAD))
        self.video_label = ttk.Label(rs_frame)
        self.video_label.configure(anchor="center")
        self.video_label.pack()
        # Placeholder negro del tamaño final para que el panel no "salte"
        self._display_frame(self.video_label, np.zeros((rs_h, rs_w, 3), dtype=np.uint8),
                             rs_w, rs_h)

        thermal_frame = ttk.LabelFrame(videos_frame, text="TC001 (térmica)")
        thermal_frame.grid(row=0, column=1)
        self.thermal_label = ttk.Label(thermal_frame)
        self.thermal_label.configure(anchor="center")
        self.thermal_label.pack()

        if self.thermal_camera is None:
            # Placeholder visible mientras no haya cámara térmica conectada
            self.thermal_label.configure(
                text=f"TC001 no disponible\n{self._thermal_error}",
                anchor="center",
                justify="center",
                font=("Helvetica", 12),
                background="#1e1e1e",
                foreground="white",
                width=thermal_w,
            )
        else:
            self._display_frame(self.thermal_label, np.zeros((thermal_h, thermal_w, 3), dtype=np.uint8),
                                 thermal_w, thermal_h)

        # ── Controles (solo afectan al modo de la RealSense) ────────────────
        controls = ttk.Frame(self.root)
        controls.pack(pady=(4, 4))

        ttk.Button(controls, text="YOLO", command=lambda: self._set_mode("yolo")).grid(
            row=0, column=0, padx=8)
        ttk.Button(controls, text="QR", command=lambda: self._set_mode("qr")).grid(
            row=0, column=1, padx=8)
        ttk.Button(controls, text="Movimiento", command=lambda: self._set_mode("movement")).grid(
            row=0, column=2, padx=8)
        ttk.Button(controls, text="Detener", command=lambda: self._set_mode(None)).grid(
            row=0, column=3, padx=8)
        ttk.Button(controls, text="Salir", style="Exit.TButton",
                   command=self._on_close).grid(row=0, column=4, padx=(24, 0))

        self.status_var = tk.StringVar(value="Modo: ninguno (idle)")
        ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel").pack(
            pady=(0, 8))

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

            self._display_frame(self.video_label, frame, *self.rs_display_size)

        # ── TC001 (heatmap ya viene procesado desde el hilo de captura) ─────
        if self.thermal_camera is not None:
            thermal_frame = self.thermal_camera.get_latest_frame()
            if thermal_frame is not None:
                self._display_frame(self.thermal_label, thermal_frame, *self.thermal_display_size)

        self.root.after(GUI_REFRESH_MS, self._update_frame)

    @staticmethod
    def _letterbox(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Escala `frame` para que quepa en target_w x target_h sin deformarlo,
        rellenando con negro el sobrante (letterbox) en vez de estirar."""
        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        scale = min(target_w / w, target_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_off = (target_w - new_w) // 2
        y_off = (target_h - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    def _display_frame(self, label: ttk.Label, frame: np.ndarray,
                        target_w: int, target_h: int) -> None:
        frame = self._letterbox(frame, target_w, target_h)
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