  
import cv2

captura = cv2.VideoCapture(0)
ancho = 640
alto = 480
codigo = cv2.VideoWriter_fourcc(*'DIVX')
salida = cv2.VideoWriter('Videoisra.mp4', codigo, 20, (ancho, alto))

while True:
    ret, video = captura.read()
    video = cv2.flip(video, 1)
    salida.write(video)
    cv2.imshow('Frame', video)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
salida.release()
cv2.destroyAllWindows()
