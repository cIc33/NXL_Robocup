#!/usr/bin/env python3
import tkinter as tk
from tkinter import font as tkfont
from tkinter import filedialog, messagebox
import json
import math
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8MultiArray, Bool, Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy



try:
    from orion_arm.orion_kinematics import OrionIK
except ImportError:
    print("ADVERTENCIA: IK Simulada.")
    class OrionIK:
        def calcular_ik(self, x, y, z, pitch): return [0,0,0,0]

# --- CONSTANTES VISUALES ---
COLOR_BG = "#121214"
COLOR_PANEL = "#1E1E1E"
COLOR_TEXTO = "#E0E0E0"
COLOR_SUBTEXTO = "#888888"
COLOR_ACCENT = "#00ADB5"
COLOR_DANGER = "#CF6679"
COLOR_STOP = "#D32F2F"
COLOR_SUCCESS = "#00C851"
COLOR_LIST = "#252526"
COLOR_BAR_BG = "#2C2C2C"

VEL_MANUAL = 10 
TOLERANCIA_LLEGADA = 4.0

# --- DIMENSIONES FÍSICAS ---
L_BASE = 135.5
L_HUMERO = 155.0
L_ANTEBRAZO = 131.0
L_MANO = 190.0

# --- CALIBRACIÓN DE PITCH ---
OFFSET_PITCH = -90.0
SIGN_PITCH = -1 

# --- CONSTANTES GRIPPER ---
G1_ABIERTO = -40.0 
G1_CERRADO = 0.0

# --- PRESETS (VALORES POR DEFECTO) ---
PRESETS = {
    "HOME":        [0.0, -30.0, -140.0, -20.0],
    "CALIBRATION": [0.0, 0.0, 0.0, 0.0],
    "ATACK":       [0.0, 45.0, -45.0, 0.0]
}

class OrionHybridGUI(Node):
    def __init__(self):
        super().__init__('gui_brazo')
        self.ik_solver = OrionIK()

        # Publicadores
        self.pub_raw = self.create_publisher(Int8MultiArray, '/brazo/raw_cmd', 10)
        self.pub_joints = self.create_publisher(Float32MultiArray, '/brazo/set_joint_angles', 10)
        self.pub_servos = self.create_publisher(Float32MultiArray, '/brazo/servos_pos_deg', 10)
        self.pub_raw_servos = self.create_publisher(Int8MultiArray, '/brazo/servos_cmd', 10)
        self.pub_stop = self.create_publisher(Bool, '/brazo/emergency_stop', 10)

        # Suscriptores
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Float32MultiArray, '/brazo/angulos_sensores', self.cb_sensors, qos)

        # Estado Brazo
        self.current_cmd = [0,0,0,0]
        self.sensors = [0.0, 0.0, 0.0, 0.0]
        self.paro_activo = False
        self.manual_active = False 
        self.last_ik_goal = None 
        self.routine_running = False
        self.tool_mode = False 
        
        # Estado Grippers (Memoria para rutinas)
        self.val_g1 = G1_ABIERTO 
        self.val_g2 = 0.0        
        self.servo_cmd = [0, 0] # [G1_cmd, G2_cmd] estado de pulsación
        
        self.create_timer(0.05, self.loop_raw)

    def cb_sensors(self, msg):
        data = list(msg.data)

        if len(data) < 4:
            return

        self.sensors = data[:4]
        keys = ["E1", "E2", "E3", "E4"]

        for i, k in enumerate(keys):
            if k in ui_sensores:
                actualizar_grafico_sensor(k, self.sensors[i])

    def set_manual_cmd(self, idx, val):
        if self.paro_activo or self.routine_running: val = 0
        self.current_cmd[idx] = int(val)

    def loop_raw(self):
        if self.routine_running:
            return

        is_moving_manual = any(v != 0 for v in self.current_cmd)

        if is_moving_manual:
            msg = Int8MultiArray()
            msg.data = self.current_cmd
            self.pub_raw.publish(msg)
            self.manual_active = True
            self.last_ik_goal = None

        elif self.manual_active:
            msg = Int8MultiArray()
            msg.data = [0, 0, 0, 0]
            self.pub_raw.publish(msg)
            self.manual_active = False

    def send_target_joints(self, joints):
        if self.paro_activo:
            return
        self.current_cmd = [0, 0, 0, 0]
        self.manual_active = False
        self.last_ik_goal = list(joints)

        msg = Float32MultiArray()
        msg.data = [float(x) for x in joints]
        self.pub_joints.publish(msg)

    # --- NUEVA FUNCIÓN DE DIRECCIÓN CONTINUA PARA SERVOS ---
    def set_servo_dir(self, idx, val):
        if self.paro_activo: return
        self.servo_cmd[idx] = int(val)
        
        msg = Int8MultiArray()
        # === INTERCAMBIO DE MOTORES (SWAP) ===
        # Enviamos [Giro, Pinza] -> [G2, G1]
        msg.data = [self.servo_cmd[1], self.servo_cmd[0]] 
        self.pub_raw_servos.publish(msg)

    # Función original mantenida intacta para no quebrar las rutinas/waypoints
    def update_servos(self, g1, g2):
        if self.paro_activo: return
        self.val_g1 = float(g1)
        self.val_g2 = float(g2)
        msg = Int8MultiArray()
        msg.data = [int(self.val_g2), int(self.val_g1)] 
        self.pub_raw_servos.publish(msg)
        
    def control_servos_raw(self, g1_cmd, g2_cmd):
        if self.paro_activo: return
        msg = Int8MultiArray()
        msg.data = [int(g2_cmd), int(g1_cmd)] 
        self.pub_raw_servos.publish(msg)
        
    def toggle_paro(self):
        self.paro_activo = not self.paro_activo
        msg = Bool(); msg.data = self.paro_activo
        self.pub_stop.publish(msg)
        if self.paro_activo:
            self.routine_running = False
            self.current_cmd = [0,0,0,0]
            # Detener también los servos
            self.set_servo_dir(0, 0); self.set_servo_dir(1, 0) 
            actualizar_barra("!!! PARO DE EMERGENCIA ACTIVADO !!!", COLOR_STOP)
            btn_paro.config(text="REANUDAR SISTEMA", bg=COLOR_SUCCESS)
        else:
            actualizar_barra("Sistema Reanudado", COLOR_ACCENT)
            btn_paro.config(text="!!! PARO DE EMERGENCIA !!!", bg=COLOR_STOP)

# ================== UI GLOBALS ==================
ros_node = None
ventana = None
ui_sensores = {}
entry_x = entry_y = entry_z = entry_pitch = None
barra_estado = None
btn_paro = None
WAYPOINTS = [] 
wp_listbox = None
chk_tool_var = None
lbl_g2_val = None

# ================== LÓGICA DE CONTROL ==================
def btn_manual_press(idx, direction):
    if ros_node.paro_activo: return
    ros_node.set_manual_cmd(idx, direction * VEL_MANUAL)

def btn_manual_release(idx):
    ros_node.set_manual_cmd(idx, 0)

# --- CONTROL GRIPPER (G1 y G2) CONTINUO ---
def btn_servo_press(idx, val):
    if ros_node: ros_node.set_servo_dir(idx, val)

def btn_servo_release(idx):
    if ros_node: ros_node.set_servo_dir(idx, 0)

def ir_a_preset(nombre):
    if ros_node.routine_running or ros_node.paro_activo: return
    if nombre in PRESETS:
        target = PRESETS[nombre]
        ros_node.send_target_joints(target)
        actualizar_barra(f"Moviendo a posición: {nombre}", COLOR_ACCENT)

def enviar_ik():
    if ros_node.routine_running or ros_node.paro_activo: return
    try:
        cx, cy = float(entry_x.get()), float(entry_y.get())
        cz, cp = float(entry_z.get()), float(entry_pitch.get())
        res = ros_node.ik_solver.calcular_ik(cx, cy, cz, cp)
        if res:
            actualizar_barra(f"IK: {cx},{cy},{cz}", COLOR_SUCCESS)
            ros_node.send_target_joints(res)
        else:
            actualizar_barra("Punto inalcanzable", COLOR_DANGER)
    except ValueError: pass

# --- JOG CARTESIANO ---
jog_c_active = False; jog_c_axis = ''; jog_c_dir = 0
def start_cart_jog(axis, d):
    if ros_node.routine_running or ros_node.paro_activo: return
    global jog_c_active, jog_c_axis, jog_c_dir
    jog_c_active = True; jog_c_axis = axis; jog_c_dir = d
    ros_node.last_ik_goal = None 
    loop_cart_jog()

def stop_cart_jog(e=None):
    global jog_c_active; jog_c_active = False

def loop_cart_jog():
    if not jog_c_active: return
    
    if ros_node.last_ik_goal is not None:
        max_error = max([abs(t - c) for t, c in zip(ros_node.last_ik_goal, ros_node.sensors)])
        if max_error > TOLERANCIA_LLEGADA:
            ventana.after(50, loop_cart_jog)
            return
            
    try:
        cx, cy = float(entry_x.get()), float(entry_y.get())
        cz, cp = float(entry_z.get()), float(entry_pitch.get())
        
        step = 2.0 * jog_c_dir 

        if not ros_node.tool_mode:
            # === MODO MUNDO ===
            if jog_c_axis == 'x': cx += step
            elif jog_c_axis == 'y': cy += step
            elif jog_c_axis == 'z': cz += step
            
        else: 
            # === MODO TOOL ===
            raw_sum = ros_node.sensors[1] - ros_node.sensors[2] + ros_node.sensors[3]
            pitch_real = (raw_sum + OFFSET_PITCH) * SIGN_PITCH
            
            rad_base = math.radians(ros_node.sensors[0]) 
            rad_pitch = math.radians(pitch_real) 
            
            cp = pitch_real 
            
            dx = dy = dz = 0
            
            if jog_c_axis == 'x': 
                step_tool = step 
                h = step_tool * math.cos(rad_pitch)
                dx = h * math.cos(rad_base)
                dy = h * math.sin(rad_base)
                dz = step_tool * math.sin(rad_pitch)
                
            elif jog_c_axis == 'y':
                dx = -step * math.sin(rad_base)
                dy = step * math.cos(rad_base)
                dz = 0
                
            elif jog_c_axis == 'z':
                h = -step * math.sin(rad_pitch)
                dx = h * math.cos(rad_base)
                dy = h * math.sin(rad_base)
                dz = step * math.cos(rad_pitch)

            cx += dx
            cy += dy
            cz += dz
            
            entry_pitch.delete(0, tk.END)
            entry_pitch.insert(0, f"{pitch_real:.1f}")

        res = ros_node.ik_solver.calcular_ik(cx, cy, cz, cp)
        if res:
            actualizar_ui_pos(cx, cy, cz, cp)
            ros_node.send_target_joints(res)

    except ValueError: pass
    ventana.after(50, loop_cart_jog)

def actualizar_ui_pos(x, y, z, p):
    entry_x.delete(0, tk.END); entry_x.insert(0, f"{x:.1f}")
    entry_y.delete(0, tk.END); entry_y.insert(0, f"{y:.1f}")
    entry_z.delete(0, tk.END); entry_z.insert(0, f"{z:.1f}")

# ================== SINCRONIZACIÓN ==================
def sincronizar_gui_con_robot():
    if not ros_node: return
    try:
        e1 = ros_node.sensors[0]
        e2 = ros_node.sensors[1]
        e3 = ros_node.sensors[2]
        
        raw_sum = ros_node.sensors[1] - ros_node.sensors[2] + ros_node.sensors[3]
        pitch_real = (raw_sum + OFFSET_PITCH) * SIGN_PITCH

        th1 = math.radians(e1)
        th2 = math.radians(e2) 
        th3 = math.radians(-e3) 
        th_pitch = math.radians(pitch_real)

        r2 = L_HUMERO * math.cos(th2)
        z2 = L_HUMERO * math.sin(th2)
        
        angle_arm = th2 + th3
        r3 = L_ANTEBRAZO * math.cos(angle_arm)
        z3 = L_ANTEBRAZO * math.sin(angle_arm)
        
        r4 = L_MANO * math.cos(th_pitch)
        z4 = L_MANO * math.sin(th_pitch)

        r_total = r2 + r3 + r4
        z_total = L_BASE + z2 + z3 + z4

        x_calc = r_total * math.cos(th1)
        y_calc = r_total * math.sin(th1)

        actualizar_ui_pos(x_calc, y_calc, z_total, pitch_real)
        entry_pitch.delete(0, tk.END)
        entry_pitch.insert(0, f"{pitch_real:.1f}")
        
        print(f"--- SYNC: P={pitch_real:.1f} ---")

    except Exception as e:
        print(f"Error en Sync: {e}")

def toggle_tool_mode_cb():
    if ros_node:
        ros_node.tool_mode = chk_tool_var.get()
        if ros_node.tool_mode:
            actualizar_barra("Modo: HERRAMIENTA -> Sincronizado", "cyan")
            sincronizar_gui_con_robot()
        else:
            actualizar_barra("Modo: MUNDO", "white")

# ================== RUTINAS ==================
def wp_agregar():
    if not ros_node: return
    joints = ros_node.sensors[:] 
    
    g1_state = ros_node.val_g1
    g2_state = ros_node.val_g2
    
    nombre = f"Punto {len(WAYPOINTS)+1}"
    dato = {
        "n": nombre, 
        "j": joints,
        "g1": g1_state,
        "g2": g2_state
    }
    WAYPOINTS.append(dato)
    wp_refrescar_lista()
    
    estado_texto = "ABIERTO" if g1_state == G1_ABIERTO else "CERRADO"
    actualizar_barra(f"Guardado: {nombre} ({estado_texto}, Giro {g2_state}°)", COLOR_ACCENT)

def wp_ejecutar_uno():
    if ros_node.routine_running or ros_node.paro_activo: return
    sel = wp_listbox.curselection()
    if not sel: return
    idx = sel[0]
    
    pt = WAYPOINTS[idx]
    
    # 1. Brazo
    ros_node.send_target_joints(pt['j'])
    
    # 2. Grippers (Si existen)
    if 'g1' in pt and 'g2' in pt:
        ros_node.update_servos(pt['g1'], pt['g2'])
    
    actualizar_barra(f"Yendo a: {pt['n']}", COLOR_SUCCESS)

def wp_reproducir_rutina():
    if ros_node.paro_activo: return
    if len(WAYPOINTS) == 0: return
    ros_node.routine_running = True
    actualizar_barra("Iniciando Rutina...", COLOR_ACCENT)
    step_rutina(0)

def step_rutina(idx):
    if not ros_node.routine_running or ros_node.paro_activo: return
    if idx >= len(WAYPOINTS):
        ros_node.routine_running = False
        actualizar_barra("Rutina Finalizada", COLOR_SUCCESS)
        return
    
    pt = WAYPOINTS[idx]
    target = pt['j']
    
    # Enviar comandos
    ros_node.send_target_joints(target)
    if 'g1' in pt and 'g2' in pt:
        ros_node.update_servos(pt['g1'], pt['g2'])
    
    wp_listbox.selection_clear(0, tk.END)
    wp_listbox.selection_set(idx)
    wp_listbox.see(idx)
    
    wait_arrival(target, idx)

def wait_arrival(target, idx):
    if not ros_node.routine_running: return
    current = ros_node.sensors
    errores = [abs(t - c) for t, c in zip(target, current)]
    if max(errores) < TOLERANCIA_LLEGADA:
        ventana.after(500, lambda: step_rutina(idx + 1))
    else:
        ventana.after(100, lambda: wait_arrival(target, idx))

def wp_detener_rutina():
    ros_node.routine_running = False
    actualizar_barra("Rutina Detenida", "orange")

def wp_eliminar():
    sel = wp_listbox.curselection()
    if not sel: return
    idx = sel[0]
    WAYPOINTS.pop(idx)
    wp_refrescar_lista()

def wp_guardar_json():
    f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
    if f:
        with open(f, 'w') as file: json.dump(WAYPOINTS, file)

def wp_cargar_json():
    global WAYPOINTS
    f = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
    if f:
        with open(f, 'r') as file: WAYPOINTS = json.load(file)
        wp_refrescar_lista()

def wp_refrescar_lista():
    wp_listbox.delete(0, tk.END)
    for pt in WAYPOINTS:
        j = pt['j']
        g_txt = ""
        if 'g1' in pt:
            state = "ABIERTO" if pt['g1'] == G1_ABIERTO else "CERRADO"
            rot = pt.get('g2', 0)
            g_txt = f"[{state} | {rot}°]"
            
        txt = f"{pt['n']} | {j[0]:.0f},{j[1]:.0f},{j[2]:.0f},{j[3]:.0f} {g_txt}"
        wp_listbox.insert(tk.END, txt)

def crear_boton_hold(parent, txt, press, release):
    b = tk.Button(parent, text=txt, font=("Segoe UI", 12, "bold"),
                  bg=COLOR_PANEL, fg="white", activebackground=COLOR_ACCENT, bd=0, width=4)
    b.bind("<ButtonPress-1>", press)
    b.bind("<ButtonRelease-1>", release)
    b.bind("<Leave>", release)
    return b

def crear_fila_manual(parent, txt, idx, key):
    f = tk.Frame(parent, bg=COLOR_BG, pady=2); f.pack(fill="x", padx=10)
    fc = tk.Frame(f, bg=COLOR_BG); fc.pack(side="left")
    tk.Label(fc, text=txt, font=("bold"), bg=COLOR_BG, fg=COLOR_SUBTEXTO, width=12, anchor="w").pack(side="left")
    crear_boton_hold(fc, "-", lambda e: btn_manual_press(idx, -1), lambda e: btn_manual_release(idx)).pack(side="left", padx=2)
    crear_boton_hold(fc, "+", lambda e: btn_manual_press(idx, 1), lambda e: btn_manual_release(idx)).pack(side="left", padx=2)
    fs = tk.Frame(f, bg=COLOR_BG); fs.pack(side="right", padx=5)
    l = tk.Label(fs, text="0.0°", font=("Consolas", 10), bg=COLOR_BG, fg=COLOR_ACCENT, width=5); l.pack(side="left")
    c = tk.Canvas(fs, width=100, height=8, bg=COLOR_BAR_BG, bd=0, highlightthickness=0); c.pack(side="left")
    r = c.create_rectangle(0,0,0,8, fill=COLOR_ACCENT, width=0)
    ui_sensores[key] = {'c':c, 'r':r, 'l':l}

def actualizar_grafico_sensor(key, val):
    ui = ui_sensores[key]
    ui['l'].config(text=f"{val:.1f}°")
    n = (float(val)+180)/360.0
    ui['c'].coords(ui['r'], 0, 0, n*100, 8)

def actualizar_barra(t, c):
    if barra_estado: barra_estado.config(text=t, fg=c)

# --- MAIN SETUP ---
def main(args=None):
    global ventana, ros_node, barra_estado, entry_x, entry_y, entry_z, entry_pitch, wp_listbox, btn_paro, chk_tool_var, lbl_g2_val
    rclpy.init(args=args)
    ros_node = OrionHybridGUI()
    
    ventana = tk.Tk()
    ventana.title("Orion Master V14 (Swapped & Inverted Grippers)")
    ventana.geometry("1150x850") 
    ventana.configure(bg=COLOR_BG)
    
    tk.Label(ventana, text="ORION CONTROL SYSTEM", font=("Segoe UI", 20, "bold"), bg=COLOR_BG, fg=COLOR_TEXTO).pack(pady=5, side="top")

    btn_paro = tk.Button(ventana, text="!!! PARO DE EMERGENCIA !!!", 
                         font=("Segoe UI", 16, "bold"), bg=COLOR_STOP, fg="white", 
                         command=ros_node.toggle_paro, pady=10)
    btn_paro.pack(side="bottom", fill="x")

    barra_estado = tk.Label(ventana, text="Sistema Listo", bg="black", fg="white", anchor="w", padx=10)
    barra_estado.pack(side="bottom", fill="x")

    main_container = tk.Frame(ventana, bg=COLOR_BG)
    main_container.pack(fill="both", expand=True, padx=20, pady=5)
    
    col_izq = tk.Frame(main_container, bg=COLOR_BG, width=400)
    col_izq.pack(side="left", fill="y", padx=10)
    
    f_presets = tk.Frame(col_izq, bg=COLOR_PANEL, pady=10)
    f_presets.pack(fill="x", pady=(0, 20))
    tk.Label(f_presets, text="POSICIONES RÁPIDAS", bg=COLOR_PANEL, fg="white", font=("bold")).pack(pady=5)
    fp_btn = tk.Frame(f_presets, bg=COLOR_PANEL); fp_btn.pack()
    def mk_pre(txt, key, col):
        tk.Button(fp_btn, text=txt, command=lambda: ir_a_preset(key), 
                  bg=col, fg="white", font=("bold"), width=10, bd=0, pady=5).pack(side="left", padx=5)
    mk_pre("HOME", "HOME", "#555")
    mk_pre("CALIB", "CALIBRATION", "#0099CC")
    mk_pre("ATACK", "ATACK", "#FF8800")

    lbl = tk.Label(col_izq, text="CONTROL ARTICULAR", font=("Segoe UI", 12, "bold"), bg=COLOR_BG, fg=COLOR_ACCENT)
    lbl.pack(anchor="w", pady=(0,10))
    crear_fila_manual(col_izq, "Base (E1)", 0, "E1")
    crear_fila_manual(col_izq, "Hombro (E2)", 1, "E2")
    crear_fila_manual(col_izq, "Codo (E3)", 2, "E3")
    crear_fila_manual(col_izq, "Muñeca (E4)", 3, "E4")
    
    # --- SECCIÓN GRIPPER (PULSACIÓN CONTINUA) ---
    tk.Frame(col_izq, height=20, bg=COLOR_BG).pack()
    tk.Label(col_izq, text="CONTROL DE GARRA (G1 & G2)", font=("Segoe UI", 12, "bold"), bg=COLOR_BG, fg=COLOR_ACCENT).pack(anchor="w")
    
    fg1 = tk.Frame(col_izq, bg=COLOR_BG); fg1.pack(pady=2, fill="x")
    tk.Label(fg1, text="G1 (Pinza):", font=("bold"), bg=COLOR_BG, fg=COLOR_SUBTEXTO, width=10, anchor="w").pack(side="left", padx=5)
    
    # Botón ABRIR (Envía -1 al pulsar, 0 al soltar)
    btn_g1_abrir = tk.Button(fg1, text="ABRIR", bg="#333", fg="white", width=8, font=("bold"))
    btn_g1_abrir.bind("<ButtonPress-1>", lambda e: btn_servo_press(0, 1))
    btn_g1_abrir.bind("<ButtonRelease-1>", lambda e: btn_servo_release(0))
    btn_g1_abrir.bind("<Leave>", lambda e: btn_servo_release(0))
    btn_g1_abrir.pack(side="left", padx=2)
    
    # Botón CERRAR (Envía 1 al pulsar, 0 al soltar)
    btn_g1_cerrar = tk.Button(fg1, text="CERRAR", bg="#333", fg="white", width=8, font=("bold"))
    btn_g1_cerrar.bind("<ButtonPress-1>", lambda e: btn_servo_press(0, -1))
    btn_g1_cerrar.bind("<ButtonRelease-1>", lambda e: btn_servo_release(0))
    btn_g1_cerrar.bind("<Leave>", lambda e: btn_servo_release(0))
    btn_g1_cerrar.pack(side="left", padx=2)
    
    fg2 = tk.Frame(col_izq, bg=COLOR_BG); fg2.pack(pady=2, fill="x")
    tk.Label(fg2, text="G2 (Giro):", font=("bold"), bg=COLOR_BG, fg=COLOR_SUBTEXTO, width=10, anchor="w").pack(side="left", padx=5)
    
    # Botón GIRO ↺ (Envía 1 al pulsar, 0 al soltar)
    btn_g2_izq = tk.Button(fg2, text="-", bg="#333", fg="white", width=4, font=("bold"))
    btn_g2_izq.bind("<ButtonPress-1>", lambda e: btn_servo_press(1, -1))
    btn_g2_izq.bind("<ButtonRelease-1>", lambda e: btn_servo_release(1))
    btn_g2_izq.bind("<Leave>", lambda e: btn_servo_release(1))
    btn_g2_izq.pack(side="left", padx=2)
    
    # Botón GIRO ↻ (Envía -1 al pulsar, 0 al soltar)
    btn_g2_der = tk.Button(fg2, text="-", bg="#333", fg="white", width=4, font=("bold"))
    btn_g2_der.bind("<ButtonPress-1>", lambda e: btn_servo_press(1, 1))
    btn_g2_der.bind("<ButtonRelease-1>", lambda e: btn_servo_release(1))
    btn_g2_der.bind("<Leave>", lambda e: btn_servo_release(1))
    btn_g2_der.pack(side="left", padx=2)
    
    lbl_g2_val = tk.Label(fg2, text="CMD", bg=COLOR_BG, fg="cyan", font=("Consolas", 10))
    lbl_g2_val.pack(side="left", padx=5)

    col_der = tk.Frame(main_container, bg=COLOR_PANEL, padx=10, pady=10)
    col_der.pack(side="right", fill="both", expand=True, padx=10)
    
    tk.Label(col_der, text="CINEMÁTICA INVERSA", bg=COLOR_PANEL, fg=COLOR_ACCENT, font=("bold")).pack(pady=5)
    fi = tk.Frame(col_der, bg=COLOR_PANEL); fi.pack(pady=5)
    def make_entry(l,v):
        tk.Label(fi, text=l, bg=COLOR_PANEL, fg="white", font=("bold")).pack(side="left", padx=(10,2))
        e=tk.Entry(fi, width=5, bg="#333", fg="white", insertbackground="white", justify="center"); e.insert(0,v); e.pack(side="left")
        return e
    entry_x=make_entry("X", 200); entry_y=make_entry("Y", 0)
    entry_z=make_entry("Z", 150); entry_pitch=make_entry("P", 0)
    tk.Button(fi, text="IR", command=enviar_ik, bg=COLOR_ACCENT, fg="black", font=("bold"), width=4).pack(side="left", padx=15)

    f_mode = tk.Frame(col_der, bg=COLOR_PANEL)
    f_mode.pack(pady=5)
    chk_tool_var = tk.BooleanVar() 
    
    tk.Checkbutton(f_mode, text="Modo Gripper (Tool)", variable=chk_tool_var, 
                   command=toggle_tool_mode_cb, 
                   bg=COLOR_PANEL, fg="white", selectcolor="#333", 
                   activebackground=COLOR_PANEL, activeforeground="white",
                   font=("Segoe UI", 10)).pack()

    fp = tk.Frame(col_der, bg=COLOR_PANEL); fp.pack(pady=10)
    def bc(t,a,d): return crear_boton_hold(fp, t, lambda e: start_cart_jog(a,d), stop_cart_jog)
    bc("Y+", 'y', 1).grid(row=0, column=1); bc("X-", 'x', -1).grid(row=1, column=0)
    bc("Y-", 'y', -1).grid(row=1, column=1); bc("X+", 'x', 1).grid(row=1, column=2)
    tk.Frame(fp, width=20, bg=COLOR_PANEL).grid(row=0, column=3)
    bc("Z+", 'z', 1).grid(row=0, column=4); bc("Z-", 'z', -1).grid(row=1, column=4)

    tk.Frame(col_der, height=2, bg=COLOR_ACCENT).pack(fill="x", pady=15)
    tk.Label(col_der, text="GESTOR DE RUTINAS", bg=COLOR_PANEL, fg="white", font=("bold")).pack()
    
    f_list = tk.Frame(col_der, bg=COLOR_PANEL); f_list.pack(fill="both", expand=True, pady=5)
    sb = tk.Scrollbar(f_list); sb.pack(side="right", fill="y")
    wp_listbox = tk.Listbox(f_list, bg=COLOR_LIST, fg=COLOR_TEXTO, height=8, font=("Consolas", 10), bd=0, yscrollcommand=sb.set)
    wp_listbox.pack(side="left", fill="both", expand=True)
    sb.config(command=wp_listbox.yview)

    f_wb = tk.Frame(col_der, bg=COLOR_PANEL); f_wb.pack(fill="x", pady=5)
    tk.Button(f_wb, text="+ Agregar P. Actual", command=wp_agregar, bg=COLOR_ACCENT, fg="black", font=("bold")).pack(side="left", padx=2)
    btn_style = {"bg": "#333", "fg": "white", "bd": 0, "padx": 5, "pady": 5}
    tk.Button(f_wb, text="Ir a Seleccionado", command=wp_ejecutar_uno, **btn_style).pack(side="left", padx=2)
    tk.Button(f_wb, text="Borrar", command=wp_eliminar, bg=COLOR_DANGER, fg="white").pack(side="left", padx=2)
    
    f_run = tk.Frame(col_der, bg=COLOR_PANEL); f_run.pack(fill="x", pady=5)
    tk.Button(f_run, text="REPRODUCIR RUTINA", command=wp_reproducir_rutina, bg=COLOR_SUCCESS, fg="white", font=("bold"), width=20).pack(side="left", padx=5)
    tk.Button(f_run, text="STOP RUTINA", command=wp_detener_rutina, bg=COLOR_DANGER, fg="white", font=("bold")).pack(side="left", padx=5)
    
    f_file = tk.Frame(col_der, bg=COLOR_PANEL); f_file.pack(fill="x", pady=5)
    tk.Button(f_file, text="Cargar JSON", command=wp_cargar_json, **btn_style).pack(side="right", padx=5)
    tk.Button(f_file, text="Guardar JSON", command=wp_guardar_json, **btn_style).pack(side="right", padx=5)

    def loop():
        if rclpy.ok(): rclpy.spin_once(ros_node, timeout_sec=0.001)
        ventana.after(10, loop)
    loop()
    
    try: ventana.mainloop()
    except KeyboardInterrupt: pass
    finally:
        if rclpy.ok(): ros_node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()