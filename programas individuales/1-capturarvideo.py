import cv2


print("Escribe tu primer nombre")
name1 =  input()
captura = cv2.VideoCapture(0)
ancho = 640
alto = 480
codigo = cv2.VideoWriter_fourcc(*'DIVX')
salida = cv2.VideoWriter(name1.lower() + '.mp4', codigo, 20, (ancho, alto))

print("A continuacion haremos algunas tomas de tu cara, cuando consideres que han pasado aproximadanente 10 segundos presiona la tecla Q en tu teclado" "Presiona enter para abrir la camara")

nonsense1 = input()


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
