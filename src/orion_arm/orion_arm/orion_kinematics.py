import math

class OrionIK:
    def __init__(self):
        self.d_base = 135.5
        self.L2 = 155.0
        self.L3 = 131.0
        self.L4 = 190.0
        
        # Límites mecánicos (Software stops)
        self.LIMITES_EJE = {
              "E1": (-180.0, 180.0),
              "E2": (-150.0, 150.0),
              "E3": (-150.0, 150.0),
              "E4": (-140.0, 140.0),
        }

        # Zonas donde el robot choca consigo mismo
        self.ZONAS_PROHIBIDAS = {
            "E2": [(-180.0, -160.0), (160.0, 180.0)],
            "E3": [(-180.0, -160.0), (160.0, 180.0)],
            "E4": [(-180.0, -160.0), (160.0, 180.0)],
        }
    
    # --- ESTA ES LA FUNCIÓN QUE FALTABA ---
    def check_limits(self, t1, t2, t3, t4):
        """Valida si los ángulos están dentro de los límites permitidos"""
        angulos = {"E1": t1, "E2": t2, "E3": t3, "E4": t4}
        
        for eje, valor in angulos.items():
            # 1. Checar límites Min/Max
            min_lim, max_lim = self.LIMITES_EJE.get(eje, (-180, 180))
            if not (min_lim <= valor <= max_lim):
                return False
            
            # 2. Checar zonas prohibidas
            zonas = self.ZONAS_PROHIBIDAS.get(eje, [])
            for (z_min, z_max) in zonas:
                if z_min < valor < z_max:
                    return False
        return True

    def calcular_ik(self, x, y, z, pitch_deg=0):
        theta1 = math.atan2(y, x)
        
        # 2. Coordenadas relativas Muñeca
        r_ground = math.sqrt(x**2 + y**2)
        pitch_rad = math.radians(pitch_deg)
        
        r_wrist = r_ground - self.L4 * math.cos(pitch_rad)
        z_wrist = (z - self.d_base) - self.L4 * math.sin(pitch_rad)
        
        # 3. Hipotenusa
        D = math.sqrt(r_wrist**2 + z_wrist**2)
        
        # Verificación Alcance
        if D > (self.L2 + self.L3) or D < abs(self.L2 - self.L3):
            return None 

        # 4. Codo
        cos_q3 = (self.L2**2 + self.L3**2 - D**2) / (2 * self.L2 * self.L3)
        cos_q3 = max(-1.0, min(1.0, cos_q3)) 
        q3_rad = math.acos(cos_q3)
        theta3_geom = math.pi - q3_rad 

        # 5. Hombro
        alpha = math.atan2(z_wrist, r_wrist)
        cos_beta = (self.L2**2 + D**2 - self.L3**2) / (2 * self.L2 * D)
        cos_beta = max(-1.0, min(1.0, cos_beta))
        beta = math.acos(cos_beta)
        theta2_geom = alpha + beta 
        
        # --- CONVERSIÓN A MOTORES ---
        t1 = math.degrees(theta1)
        t2 =  90.0 - math.degrees(theta2_geom) 
        t3 = -math.degrees(theta3_geom)
        t4_geom_deg = math.degrees(pitch_rad - theta2_geom + theta3_geom)
        t4 = -t4_geom_deg
        
        # Validación final usando la función que acabamos de agregar
        if self.check_limits(t1, t2, t3, t4):
            return (t1, t2, t3, t4)
        else:
            return None