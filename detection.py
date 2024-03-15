import cv2
import numpy as np
import keras 
# Inicializando Variables
video_capture = cv2.VideoCapture(0)
eyeLeft = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
eyeRight = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
model = keras.models.load_model('keras_model.h5')
#model = keras.models.load_model('keras_model.h5') # Si estás seguro de que Keras está instalado y quieres usar keras.models.load_model

left_x, left_y, left_w, left_h = 0, 0, 0, 0
right_x, right_y, right_w, right_h = 0, 0, 0, 0
contador = 0

# Ejecutando Video
while True:
    ret, frame = video_capture.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Mostrando Contador
    cv2.rectangle(frame, (0, 0), (width, int(height*0.1)), (0, 0, 0), -1)
    cv2.putText(frame, 'Contador: ' + str(contador), (int(width*0.65), int(height*0.08)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    
    # Identificando el Ojo Derecho
    ojo_der = eyeRight.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=3,
        minSize=(30, 30)
    )
    for (x, y, w, h) in ojo_der:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        right_x, right_y, right_w, right_h = x, y, w, h
        break
    
    # Identificando el Ojo Izquierdo
    ojo_izq = eyeLeft.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=3,
        minSize=(30, 30)
    )
    for (x, y, w, h) in ojo_izq:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        left_x, left_y, left_w, left_h = x, y, w, h
        break
    
    # Identificando Coordenadas x,y iniciales y finales para extraer la foto de los ojos
    if left_x > right_x:
        start_x, end_x = right_x, (left_x+left_w)
    else:
        start_x, end_x = left_x, (right_x+right_w)

    if left_y > right_y:
        start_y, end_y = right_y, (left_y+left_h)
    else:
        start_y, end_y = left_y, (right_y+right_h)
    
    
    # Algoritmo de deteccion de sueño
    if (end_x-start_x) > 120 and (end_y-start_y) < 200:
        start_x, start_y, end_x, end_y = start_x-30, start_y-50, end_x+30, end_y+50
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        img = frame[start_y:end_y, start_x:end_x]
        imagen = cv2.resize(img, (224, 224))
        imagen_normalizada = (imagen.astype(np.float32) / 127.0) - 1
        data[0] = imagen_normalizada
        prediction = model.predict(data)
        
        if list(prediction)[0][1] >= 0.95:
            cv2.putText(frame, 'Durmiendo : ' + str(round(list(prediction)[0][1], 3)), (10, int(height*0.08)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            contador += 1
        
        if list(prediction)[0][0] >= 0.95:
            cv2.putText(frame, 'Despierto : ' + str(round(list(prediction)[0][0], 3)), (10, int(height*0.08)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            contador = max(0, contador - 1)
        
    # Reproduciendo sonido si el contador alcanza un cierto límite
    if contador >= 10:
        # Para reproducir el sonido, es posible que necesites instalar la biblioteca pygame y descomentar esta línea:
        # pygame.mixer.init()
        # pygame.mixer.music.load("wakeup.mp3")
        # pygame.mixer.music.play()
        contador = 0
    
    cv2.imshow('Detección de sueño', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberando recursos
video_capture.release()
cv2.destroyAllWindows()
