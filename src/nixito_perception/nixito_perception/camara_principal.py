"""
Ventana de camara con OpenCV como una ventana normal del sistema:
- Se abre como ventana normal (no pantalla completa), redimensionable.
- Se puede minimizar con el boton nativo de la ventana (el del sistema
  operativo), y al hacer clic en su icono en la barra de tareas/dock
  se expande de nuevo, igual que cualquier otra ventana.
- Tecla ESC o 'q' para salir.
"""

import cv2

WINDOW_NAME = "Camara"


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        return

    # WINDOW_NORMAL crea una ventana normal del sistema operativo,
    # con sus botones nativos de minimizar / maximizar / cerrar.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame de la camara.")
            break

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC o 'q' para salir
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()