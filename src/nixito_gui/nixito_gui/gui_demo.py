import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Bool, Int32
from cv_bridge import CvBridge

# Importaciones para Lifecycle (Nodos con estado)
from lifecycle_msgs.srv import GetState, ChangeState
from lifecycle_msgs.msg import Transition
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy

import threading
import time
import cv2
import numpy as np
import os
import subprocess



# PyQt6 imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, 
                             QProgressBar)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QUrl
from PyQt6.QtGui import QImage, QPixmap, QPainter, QBrush, QPen, QColor
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtGui import QWindow
import sys

# Colores usados en la UI
COLOR_STOP = "#D32F2F"
COLOR_ACCENT = "#00ADB5"
COLOR_BG = "#1e1e1e"
COLOR_BG_DARK = "#000000"
COLOR_PANEL = "#2b2b2b"

# Presets de posiciones para el brazo
PRESETS = {
    "HOME":        [0.0, 40.0, -140.0, 60.0],
    "CALIBRATION": [0.0, 0.0, 0.0, 0.0],
    "ATACK":       [0.0, 45.0, -45.0, 0.0]
}

# Variable global para el nodo (se asigna en MainWindow._init_)
ros_node = None

def ir_a_preset(nombre):
    """Envía un preset predefinido al brazo."""
    global ros_node
    if ros_node:
        if nombre in PRESETS:
            print(f"Enviando preset {nombre}: {PRESETS[nombre]}")
            ros_node.send_target_joints(PRESETS[nombre])
        else:
            print(f"Preset {nombre} no encontrado. Disponibles: {list(PRESETS.keys())}")
    else:
        print("Nodo ROS no inicializado aún")
        

class DisplayNode(Node):
    def __init__(self, gui):
        super().__init__("gui3_node")

        self.gui = gui
        self.bridge = CvBridge()
        # Control de modo visual activo
        self.current_mode = "raw"  # raw | yolo | qr | thermal

        # Control de FPS independiente
        self.last_raw_time = 0.0
        self.last_filter_time = 0.0
        self.services_ready = False
        
        # publicador posiciones rapidas para el brazo
        self.pub_joints = self.create_publisher(Float32MultiArray, "/brazo/set_joint_angles", 10)
        self.pub_stop = self.create_publisher(Bool, '/brazo/emergency_stop', 10)
        
        # Estado interno del nodo / sensores
        self.sensors = [0.0, 0.0, 0.0, 0.0]
        self.paro_activo = False
        self.routine_running = False
        self.mq_value = 0

        # Suscripción a los ángulos de sensores del brazo
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Float32MultiArray, '/brazo/angulos_sensores', self.cb_sensors, qos)

        # Suscripciones a tópicos de imagen
        self.create_subscription(Image, '/brazo/image_raw', self.cb_raw, 10)
        self.create_subscription(Image, '/principal/image_raw', self.cb_usb, 10)
        self.create_subscription(Image, 'vision/segmented', self.cb_filter, 10)
        # NUEVA USB CAM (SIEMPRE VISIBLE ARRIBA)
        self.create_subscription(Image, '/usb_cam/image_raw', self.cb_usb, 10)

        # suscripciones a topicos de sensores
        self.create_subscription(Int32, '/ADC', self.cb_mq, 10)

        # Clientes para controlar el ciclo de vida del nodo de visión
        self.get_state_client = self.create_client(GetState, "vision_node/get_state")
        self.change_state_client = self.create_client(ChangeState, "vision_node/change_state")
        self.set_param_client = self.create_client(SetParameters, "vision_node/set_parameters")

        # Se lanza un hilo para no bloquear la GUI mientras se espera a los servicios
        threading.Thread(target=self.wait_services, daemon=True).start()

    def wait_services(self):
        """Espera a que los servicios del nodo de visión estén disponibles."""
        while not (
            self.get_state_client.wait_for_service(timeout_sec=1.0) and
            self.change_state_client.wait_for_service(timeout_sec=1.0) and
            self.set_param_client.wait_for_service(timeout_sec=1.0)
        ):
            self.get_logger().info("Esperando servicios de Lifecycle...")
            time.sleep(1.0)

        self.get_logger().info("Servicios conectados.")
        self.services_ready = True
        # Usa señal para interactuar con la GUI desde otro hilo de forma segura
        self.gui.signals.services_ready.emit()

    def cb_raw(self, msg):
        # Solo mostrar imagen cruda si estamos en modo raw
        if self.current_mode != "raw":
            return

        # Limitador de FPS independiente
        if time.time() - self.last_raw_time < 0.03:  # ~30 FPS
            return

        self.last_raw_time = time.time()
        self.process(msg)

    def cb_filter(self, msg):
        # Solo mostrar imagen procesada si estamos en modo de visión
        if self.current_mode not in ["yolo", "qr", "thermal"]:
            return

        if time.time() - self.last_filter_time < 0.03:
            return

        self.last_filter_time = time.time()
        self.process(msg)

    def cb_mq(self, msg):
        self.mq_value = msg.data
        self.gui.signals.update_mq.emit(self.mq_value)
    
    def cb_usb(self, msg):
        # No depende del modo, siempre se muestra arriba
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.gui.signals.update_top_image.emit(frame)

    def process(self, msg):
        """Convierte mensaje de ROS a formato OpenCV."""
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.gui.signals.update_image.emit(frame)

    def cb_sensors(self, msg):
        """Callback para actualizar lecturas de encoders y notificar la GUI."""
        try:
            self.sensors = list(msg.data)
        except Exception:
            return
        if hasattr(self.gui.signals, 'update_sensors'):
            # Emitir señal con los datos de sensores
            self.gui.signals.update_sensors.emit(self.sensors)
            
    # paro de emergencia brazo
    def toggle_paro(self):
        """Alterna el paro de emergencia y publica el estado al tópico."""
        self.paro_activo = not self.paro_activo
        msg = Bool()
        msg.data = self.paro_activo
        self.pub_stop.publish(msg)
        self.gui.signals.emergency_state.emit(self.paro_activo)
            
    # posiciones rapidas para el brazo
    def send_target_joints(self, joints):
        """Envía los ángulos de articulación al tópico /brazo/set_joint_angles."""
        if self.paro_activo:
            self.get_logger().warning("Sistema en paro de emergencia. No se puede enviar comandos.")
            return
        msg = Float32MultiArray()
        msg.data = [float(x) for x in joints]
        self.pub_joints.publish(msg)
        self.get_logger().info(f"Posición enviada: {msg.data}")

    # Funciones para manejar el estado del nodo remoto (Lifecycle)
    def get_state(self):
        if not self.services_ready: 
            return "services_not_ready"
        req = GetState.Request()
        future = self.get_state_client.call_async(req)
        while rclpy.ok() and not future.done():
            time.sleep(0.01)
        return future.result().current_state.label

    def change_state(self, transition):
        if not self.services_ready: 
            return False
        req = ChangeState.Request()
        req.transition.id = transition
        future = self.change_state_client.call_async(req)
        while rclpy.ok() and not future.done():
            time.sleep(0.01)
        return future.result().success

    def set_parameter(self, name, value):
        if not self.services_ready: 
            return False
        req = SetParameters.Request()
        param = Parameter(name=name, value=value)
        req.parameters = [param.to_parameter_msg()]
        future = self.set_param_client.call_async(req)
        while rclpy.ok() and not future.done():
            time.sleep(0.01)
        return future.result().results[0].successful

    def set_mode(self, mode):
        """Cambia el modo de visión (YOLO/QR) manejando la transición de estados."""
        if not self.services_ready:
            self.get_logger().warning("Servicios no listos.")
            return "desactivado"

        state = self.get_state()
        if state == "inactive":
            self.set_parameter("vision_mode", mode)
            self.change_state(Transition.TRANSITION_ACTIVATE)
        elif state == "active":
            self.set_parameter("vision_mode", mode)
        return self.get_state()

    def deactivate(self):
        """Pasa el nodo de visión a estado inactivo."""
        if not self.services_ready: 
            return "desactivado"
        state = self.get_state()
        if state == "active":
            self.change_state(Transition.TRANSITION_DEACTIVATE)
        return self.get_state()

class CommunicationSignals(QObject):
    """Señales para comunicación entre hilos ROS y GUI"""
    update_image = pyqtSignal(object)
    update_top_image = pyqtSignal(object)
    update_mq = pyqtSignal(int)
    update_sensors = pyqtSignal(list)
    services_ready = pyqtSignal()
    emergency_state = pyqtSignal(bool)
    mode_changed = pyqtSignal(str, str)

class LedIndicator(QWidget):
    """Indicador LED personalizado"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.color = QColor("red")
        
    def set_color(self, color_name):
        self.color = QColor(color_name)
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(self.color))
        painter.setPen(QPen(Qt.GlobalColor.black, 1))
        painter.drawEllipse(2, 2, 16, 16)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Señales para comunicación con ROS
        self.signals = CommunicationSignals()
        
        # Inicializar diccionario de sensores ANTES de usarlo
        self.ui_sensores = {}
        
        # Variables para imágenes
        self.current_top_frame = None
        self.current_bottom_frame = None
        
        # Dimensiones
        self.camera_width = 960
        self.camera_height = 915
        self.single_camera_height = self.camera_height // 2
        self.extra_width = 890
        self.extra_height = 449
        
        # Configurar ventana
        self.setup_window()
        
        # Configurar señales
        self.setup_signals()
        
        # Crear UI
        self.setup_ui()
        
        # Inicializar ROS
        self.init_ros()
        
        # Timer para actualizar imágenes
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_images)
        self.timer.start(30)  # ~30 FPS

    def setup_signals(self):
        """Conectar señales con slots"""
        self.signals.update_image.connect(self.on_image_received)
        self.signals.update_top_image.connect(self.on_top_image_received)
        self.signals.update_mq.connect(self.on_mq_updated)
        self.signals.update_sensors.connect(self.on_sensors_updated)
        self.signals.services_ready.connect(self.on_services_ready)
        self.signals.emergency_state.connect(self.on_emergency_state_changed)

    def setup_window(self):
        """Configurar la ventana principal"""
        self.setWindowTitle("Vision Controller PRO")
        self.showFullScreen()
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLOR_BG_DARK};
            }}
            QLabel {{
                color: white;
            }}
            QPushButton {{
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
                background-color: #444;
                color: white;
                border: none;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
            QPushButton:disabled {{
                background-color: #333;
                color: #666;
            }}
        """)

    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 0, 20, 10)
        main_layout.setSpacing(0)
        
        # Título
        title = QLabel("UNIDAD DE CONTROL")
        title.setStyleSheet("font-size: 30px; font-weight: bold; color: white; margin: 5px 0; font-family: Segoe UI;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # Top container (cámara + paneles derecho)
        top_container = QWidget()
        top_layout = QHBoxLayout(top_container)
        top_layout.setContentsMargins(0, 20, 0, 0)
        top_layout.setSpacing(20)
        
        # Panel de cámaras
        self.setup_camera_panel()
        top_layout.addWidget(self.camera_panel, 0, Qt.AlignmentFlag.AlignTop)  # Alinear arriba
        
        # Paneles derecho
        right_container = self.setup_right_panels()
        top_layout.addWidget(right_container, 0, Qt.AlignmentFlag.AlignTop)  # Alinear arriba
        
        main_layout.addWidget(top_container)
        
        # Bottom container (controles, sensores, etc.)
        bottom_container = self.setup_bottom_panels()
        main_layout.addWidget(bottom_container)

    def setup_camera_panel(self):
        """Configurar el panel de cámaras - CON AJUSTE AUTOMÁTICO"""
        self.camera_panel = QFrame()
        # Eliminar setFixedSize y usar tamaño mínimo en su lugar
        self.camera_panel.setMinimumSize(self.camera_width, self.camera_height)
        self.camera_panel.setStyleSheet("border: none; background-color: transparent;")
        
        # Layout con márgenes CERO
        layout = QVBoxLayout(self.camera_panel)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Calcular altura exacta para cada cámara
        border_size = 4
        available_height = self.camera_height - layout.spacing() - border_size
        self.single_camera_height = available_height // 2
        
        # Contenedor para la cámara superior
        top_container = QFrame()
        # Usar tamaño mínimo en lugar de fijo
        top_container.setMinimumSize(self.camera_width, self.single_camera_height)
        top_container.setStyleSheet("border: none; background-color: transparent;")
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Cámara superior - CON ESCALADO AUTOMÁTICO
        self.top_camera_label = QLabel()
        # Usar tamaño mínimo
        self.top_camera_label.setMinimumSize(self.camera_width, self.single_camera_height)
        self.top_camera_label.setStyleSheet("border: 2px solid #333; background-color: black;")
        self.top_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # ACTIVAR ESCALADO AUTOMÁTICO
        self.top_camera_label.setScaledContents(True)
        top_layout.addWidget(self.top_camera_label)
        
        # Contenedor para la cámara inferior
        bottom_container = QFrame()
        bottom_container.setMinimumSize(self.camera_width, self.single_camera_height)
        bottom_container.setStyleSheet("border: none; background-color: transparent;")
        bottom_layout = QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        # Cámara inferior - CON ESCALADO AUTOMÁTICO
        self.bottom_camera_label = QLabel()
        self.bottom_camera_label.setMinimumSize(self.camera_width, self.single_camera_height)
        self.bottom_camera_label.setStyleSheet("border: 2px solid #333; background-color: black;")
        self.bottom_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # ACTIVAR ESCALADO AUTOMÁTICO
        self.bottom_camera_label.setScaledContents(True)
        bottom_layout.addWidget(self.bottom_camera_label)
        
        # Agregar los contenedores al layout principal
        layout.addWidget(top_container)
        layout.addWidget(bottom_container)
    
    def launch_embedded_rviz(self, parent_layout):
        """Lanza rviz2 y embebe su ventana X11 en un layout Qt."""
        rviz_config = "/home/axelcg_7905/nixito_ws/src/nixito_gui/config/nixito_embedded.rviz"

        if not os.path.isfile(rviz_config):
            print(f"[RVIZ] No existe config: {rviz_config}")
            placeholder = QLabel("No se encontró nixito_embedded.rviz")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: white; background-color: black;")
            parent_layout.addWidget(placeholder)
            return

        try:
            self.rviz_process = subprocess.Popen(
                ["rviz2", "-d", rviz_config],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"[RVIZ] Error al lanzar rviz2: {e}")
            placeholder = QLabel("No se pudo lanzar rviz2")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: white; background-color: black;")
            parent_layout.addWidget(placeholder)
            return

        # Esperar un poco a que aparezca la ventana
        QTimer.singleShot(2500, lambda: self.embed_rviz_window(parent_layout))
        
    def embed_rviz_window(self, parent_layout):
        """Busca la ventana de rviz2 y la mete dentro del panel Qt."""
        try:
            result = subprocess.check_output(
                ["xdotool", "search", "--name", "RViz"],
                text=True
            ).strip().splitlines()

            if not result:
                raise RuntimeError("No se encontró ventana RViz")

            win_id = int(result[-1])

            rviz_window = QWindow.fromWinId(win_id)
            rviz_widget = QWidget.createWindowContainer(rviz_window)
            rviz_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            rviz_widget.setStyleSheet("background-color: black;")

            while parent_layout.count():
                item = parent_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

            parent_layout.addWidget(rviz_widget)
            self.rviz_container = rviz_widget
            self.rviz_window = rviz_window

            print("[RVIZ] Embebido correctamente")

        except Exception as e:
            print(f"[RVIZ] Error embebiendo ventana: {e}")
            placeholder = QLabel("RViz no pudo embeberse")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: white; background-color: black;")
            parent_layout.addWidget(placeholder)
        
    def setup_right_panels(self):
        right_column = QWidget()
        right_column.setFixedWidth(self.extra_width)
        layout = QVBoxLayout(right_column)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)

        top_right = QFrame()
        top_right.setFixedSize(self.extra_width, self.extra_height)
        top_right.setStyleSheet("border: 2px solid #333; background-color: black;")

        self.rviz_layout = QVBoxLayout(top_right)
        self.rviz_layout.setContentsMargins(0, 0, 0, 0)

        loading = QLabel("Iniciando RViz2...")
        loading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading.setStyleSheet("color: white; background-color: black; font-size: 20px;")
        self.rviz_layout.addWidget(loading)

        layout.addWidget(top_right)

        self.launch_embedded_rviz(self.rviz_layout)

        bottom_right = QWidget()
        bottom_right.setFixedSize(self.extra_width, self.extra_height)
        bottom_layout = QHBoxLayout(bottom_right)
        bottom_layout.setSpacing(10)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        left_panel = QFrame()
        left_panel.setFixedSize(435, self.extra_height)
        left_panel.setStyleSheet(f"border: 2px solid #333; background-color: {COLOR_BG_DARK};")
        bottom_layout.addWidget(left_panel)

        right_panel = QFrame()
        right_panel.setFixedSize(435, self.extra_height)
        right_panel.setStyleSheet(f"""
            QFrame {{
                border: 2px solid #333;
                background-color: {COLOR_BG_DARK};
            }}
            QFrame > QWidget, QFrame > QLabel {{
                border: none;
            }}
        """)
        bottom_layout.addWidget(right_panel)

        self.setup_encoders_panel(right_panel)

        layout.addWidget(bottom_right)
        return right_column
    
    def launch_local_foxglove(self):
        """Lanza la aplicación Foxglove Studio conectada al bridge local"""
        try:
            subprocess.Popen(["foxglove-studio", "--url", "ws://localhost:8765"])
            self.status_label.setText("Estado: Visualizador 3D Activo")
        except Exception as e:
            self.status_label.setText(f"Error: No se encontró Foxglove App")
            print(f"Error al lanzar: {e}")

    def setup_encoders_panel(self, parent):
        """Configurar el panel de encoders - VERSIÓN GRANDE"""
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(20, 20, 20, 20)  # Márgenes generosos
        layout.setSpacing(25)  # Espaciado amplio entre elementos
        
        # Título MUY grande
        title = QLabel("Encoders Brazo")
        title.setStyleSheet("font-size: 25px; font-weight: bold; color: white; border: none; padding: 15px; font-family: Segoe UI;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Configuración de los joints
        joints_config = [
            ("Base (E1)", "E1"),
            ("Hombro (E2)", "E2"),
            ("Codo (E3)", "E3"),
            ("Muñeca (E4)", "E4")
        ]
        
        for name, key in joints_config:
            # Frame para cada joint - altura grande
            joint_frame = QWidget()
            joint_frame.setStyleSheet("border: none;")
            joint_frame.setFixedHeight(80)  # Altura fija grande para cada fila
            joint_layout = QHBoxLayout(joint_frame)
            joint_layout.setContentsMargins(15, 5, 15, 5)  # Márgenes amplios
            joint_layout.setSpacing(20)  # Espaciado grande
            
            # Label del nombre - GRANDE
            name_label = QLabel(name)
            name_label.setStyleSheet("font-size: 24px; color: #E0E0E0; border: none; font-weight: bold;")
            name_label.setFixedWidth(200)  # Ancho fijo grande
            name_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            joint_layout.addWidget(name_label)
            
            # Label del valor - GRANDE
            value_label = QLabel("0.0°")
            value_label.setStyleSheet("font-size: 24px; color: #00ADB5; font-weight: bold; border: none;")
            value_label.setFixedWidth(50)  # Ancho fijo grande
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            joint_layout.addWidget(value_label)
            
            # Barra de progreso - GRANDE
            progress = QProgressBar()
            progress.setRange(-90, 90)
            progress.setValue(0)
            progress.setStyleSheet("""
                QProgressBar {
                    border: 3px solid #444;
                    border-radius: 10px;
                    text-align: center;
                    background-color: #2C2C2C;
                    font-size: 18px;
                    color: white;
                    min-height: 20px;
                    max-height: 20px;
                }
                QProgressBar::chunk {
                    background-color: #00ADB5;
                    border-radius: 8px;
                }
            """)
            progress.setFixedHeight(25)  # Altura grande
            progress.setMinimumWidth(150)  # Ancho mínimo grande
            progress.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            joint_layout.addWidget(progress)
            
            # Guardar referencias
            self.ui_sensores[key] = {
                'progress': progress,
                'label': value_label
            }
            
            layout.addWidget(joint_frame)
        
        # Espacio al final para que todo quede arriba
        layout.addStretch(1)

    def setup_bottom_panels(self):
        """Configurar los paneles inferiores"""
        bottom_container = QWidget()
        bottom_container.setFixedHeight(80)
        bottom_layout = QHBoxLayout(bottom_container)
        bottom_layout.setSpacing(5)
        bottom_layout.setContentsMargins(0, 20, 0, 0)
        
        # Panel de controles
        self.setup_controls_panel()
        bottom_layout.addWidget(self.controls_frame)
        
        # Panel MQ
        self.setup_mq_panel()
        bottom_layout.addWidget(self.mq_frame)
        
        # Panel Magnetómetro
        self.setup_magnetometro_panel()
        bottom_layout.addWidget(self.magnetometro_frame)
        
        # Panel Presets
        self.setup_presets_panel()
        bottom_layout.addWidget(self.presets_frame)
        
        # Panel Emergencia
        self.setup_emergency_panel()
        bottom_layout.addWidget(self.emergency_frame)
        
        return bottom_container

    def setup_controls_panel(self):
        """Configurar panel de controles"""
        self.controls_frame = QFrame()
        self.controls_frame.setFixedSize(863, 55)
        self.controls_frame.setStyleSheet(f"background-color: {COLOR_PANEL}; border: none; border-radius: 5px;")
        
        layout = QHBoxLayout(self.controls_frame)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(10)
        
        # LED indicador
        self.led = LedIndicator()
        layout.addWidget(self.led)
        
        # Estado
        self.status_label = QLabel("Estado: Conectando...")
        self.status_label.setStyleSheet("color: white; font-size: 12px; font-family: Segoe UI;")
        layout.addWidget(self.status_label)
        
        # Botones de modo
        self.btn_yolo = QPushButton("Modo YOLO")
        self.btn_yolo.clicked.connect(lambda: self.activate_mode("yolo"))
        layout.addWidget(self.btn_yolo)
        
        self.btn_qr = QPushButton("Modo QR")
        self.btn_qr.clicked.connect(lambda: self.activate_mode("qr"))
        layout.addWidget(self.btn_qr)
        
        self.btn_termico = QPushButton("Modo Térmico")
        self.btn_termico.clicked.connect(lambda: self.activate_mode("thermal"))
        layout.addWidget(self.btn_termico)
        
        self.btn_detect = QPushButton("Movimiento")
        self.btn_detect.clicked.connect(lambda: self.activate_mode("detect"))
        layout.addWidget(self.btn_detect)
        
        self.btn_deactivate = QPushButton("Desactivar")
        self.btn_deactivate.clicked.connect(self.deactivate_node)
        layout.addWidget(self.btn_deactivate)
        
        self.disable_buttons()

    def setup_mq_panel(self):
        """Configurar panel del sensor MQ"""
        self.mq_frame = QFrame()
        self.mq_frame.setFixedSize(160, 55)
        self.mq_frame.setStyleSheet(f"background-color: {COLOR_PANEL}; border: none; border-radius: 5px;")
        
        layout = QVBoxLayout(self.mq_frame)
        layout.setSpacing(2)
        
        title = QLabel("Sensor MQ")
        title.setStyleSheet("color: white; font-weight: bold; font-size: 15px; font-family: Segoe UI;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        self.mq_value_label = QLabel("Valor: 0 ppm")
        self.mq_value_label.setStyleSheet("color: #00ff88; font-weight: bold; font-size: 15px;")
        self.mq_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.mq_value_label)

    def setup_magnetometro_panel(self):
        """Configurar panel del magnetómetro"""
        self.magnetometro_frame = QFrame()
        self.magnetometro_frame.setFixedSize(160, 55)
        self.magnetometro_frame.setStyleSheet(f"background-color: {COLOR_PANEL}; border: none; border-radius: 5px;")
        
        layout = QVBoxLayout(self.magnetometro_frame)
        layout.setSpacing(2)
        
        title = QLabel("Magnetometro")
        title.setStyleSheet("color: white; font-weight: bold; font-size: 15px; font-family: Segoe UI;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        self.magnetometro_value_label = QLabel("Valor: 0.0")
        self.magnetometro_value_label.setStyleSheet("color: #00ff88; font-weight: bold; font-size: 15px;")
        self.magnetometro_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.magnetometro_value_label)

    def setup_presets_panel(self):
        """Configurar panel de presets del brazo"""
        self.presets_frame = QFrame()
        self.presets_frame.setFixedSize(350, 55)
        self.presets_frame.setStyleSheet(f"background-color: {COLOR_PANEL}; border: none; border-radius: 5px;")
        
        layout = QVBoxLayout(self.presets_frame)
        layout.setSpacing(2)
        
        title = QLabel("Presets brazo")
        title.setStyleSheet("color: white; font-weight: bold; font-size: 15px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(5)
        
        self.btn_home = QPushButton("HOME")
        self.btn_home.setStyleSheet("background-color: #555; color: white; font-size: 12px; padding: 5px;")
        self.btn_home.clicked.connect(lambda: ir_a_preset("HOME"))
        buttons_layout.addWidget(self.btn_home)
        
        self.btn_calib = QPushButton("CALIB")
        self.btn_calib.setStyleSheet("background-color: #0099CC; color: white; font-size: 12px; padding: 5px;")
        self.btn_calib.clicked.connect(lambda: ir_a_preset("CALIBRATION"))
        buttons_layout.addWidget(self.btn_calib)
        
        self.btn_atack = QPushButton("ATACK")
        self.btn_atack.setStyleSheet("background-color: #FF8800; color: white; font-size: 12px; padding: 5px;")
        self.btn_atack.clicked.connect(lambda: ir_a_preset("ATACK"))
        buttons_layout.addWidget(self.btn_atack)
        
        layout.addLayout(buttons_layout)

    def setup_emergency_panel(self):
        """Configurar panel de emergencia"""
        self.emergency_frame = QFrame()
        self.emergency_frame.setFixedSize(320, 55)
        self.emergency_frame.setStyleSheet(f"background-color: {COLOR_PANEL}; border: none; border-radius: 5px;")
        
        layout = QHBoxLayout(self.emergency_frame)
        
        self.emergency_btn = QPushButton("PARO DE EMERGENCIA")
        self.emergency_btn.setStyleSheet(f"""
            background-color: {COLOR_STOP};
            color: white;
            font-weight: bold;
            font-size: 15px;
            padding: 8px 15px;
            border: none;
            border-radius: 4px;
        """)
        self.emergency_btn.clicked.connect(self.toggle_paro)
        layout.addWidget(self.emergency_btn)

    def init_ros(self):
        """Inicializar nodo ROS"""
        global ros_node
        
        rclpy.init()
        
        self.node = DisplayNode(self)
        ros_node = self.node
        
        self.ros_thread = threading.Thread(
            target=rclpy.spin,
            args=(self.node,),
            daemon=True
        )
        self.ros_thread.start()

    def on_image_received(self, frame):
        """Slot para imagen procesada"""
        self.current_bottom_frame = frame

    def on_top_image_received(self, frame):
        """Slot para imagen USB"""
        self.current_top_frame = frame

    def on_mq_updated(self, value):
        """Slot para actualizar valor MQ"""
        self.mq_value_label.setText(f"Valor: {value} ppm")

    def on_sensors_updated(self, sensors):
        """Slot para actualizar sensores"""
        if len(sensors) >= 4:
            joints_map = {0: 'E1', 1: 'E2', 2: 'E3', 3: 'E4'}
            for i, value in enumerate(sensors[:4]):
                key = joints_map[i]
                if key in self.ui_sensores:
                    self.ui_sensores[key]['label'].setText(f"{value:.1f}°")
                    self.ui_sensores[key]['progress'].setValue(int(value))

    def on_services_ready(self):
        """Slot cuando los servicios están listos"""
        self.enable_buttons()
        self.status_label.setText("Servicios conectados")
        self.led.set_color("green")

    def on_emergency_state_changed(self, active):
        """Slot para cambio en estado de emergencia"""
        if active:
            self.emergency_btn.setStyleSheet(f"""
                background-color: {COLOR_STOP};
                color: white;
                font-weight: bold;
                font-size: 12px;
                border: 2px solid white;
                padding: 8px 15px;
                border-radius: 4px;
            """)
        else:
            self.emergency_btn.setStyleSheet(f"""
                background-color: {COLOR_STOP};
                color: white;
                font-weight: bold;
                font-size: 15px;
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
            """)

    def update_images(self):
        """Actualizar imágenes en la GUI"""
        if self.current_top_frame is not None:
            rgb_image = cv2.cvtColor(self.current_top_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.camera_width, self.single_camera_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.top_camera_label.setPixmap(scaled_pixmap)
        
        if self.current_bottom_frame is not None:
            rgb_image = cv2.cvtColor(self.current_bottom_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.camera_width, self.single_camera_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.bottom_camera_label.setPixmap(scaled_pixmap)

    def activate_mode(self, mode):
        """Activar modo de visión"""
        threading.Thread(
            target=self._activate_mode_thread,
            args=(mode,),
            daemon=True
        ).start()

    def _activate_mode_thread(self, mode):
        """Hilo para activar modo"""
        new_state = self.node.set_mode(mode)
        
        if new_state == "active":
            self.node.current_mode = mode
        else:
            self.node.current_mode = "raw"
        
        # Actualizar UI en el hilo principal
        self.signals.mode_changed.emit(mode, new_state)

    def deactivate_node(self):
        """Desactivar nodo de visión"""
        threading.Thread(
            target=self._deactivate_thread,
            daemon=True
        ).start()

    def _deactivate_thread(self):
        """Hilo para desactivar nodo"""
        new_state = self.node.deactivate()
        self.node.current_mode = "raw"
        
        # Actualizar UI en el hilo principal
        QTimer.singleShot(0, lambda: self._update_deactivate_ui(new_state))

    def _update_deactivate_ui(self, new_state):
        """Actualizar UI después de desactivar"""
        self.led.set_color("red")
        self.status_label.setText(f"Estado: {new_state}")

    def toggle_paro(self):
        """Alternar paro de emergencia"""
        if hasattr(self, "node"):
            self.node.toggle_paro()

    def disable_buttons(self):
        """Deshabilitar botones de modo"""
        self.btn_yolo.setEnabled(False)
        self.btn_qr.setEnabled(False)
        self.btn_termico.setEnabled(False)
        self.btn_detect.setEnabled(False)
        self.btn_deactivate.setEnabled(False)

    def enable_buttons(self):
        """Habilitar botones de modo"""
        self.btn_yolo.setEnabled(True)
        self.btn_qr.setEnabled(True)
        self.btn_termico.setEnabled(True)
        self.btn_detect.setEnabled(True)
        self.btn_deactivate.setEnabled(True)

    def closeEvent(self, event):
        """Manejar cierre de la aplicación"""
        try:
            if hasattr(self, "rviz_process") and self.rviz_process is not None:
                self.rviz_process.terminate()
        except Exception:
            pass
        event.accept()
        if hasattr(self, "node"):
            self.node.destroy_node()
        rclpy.shutdown()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()