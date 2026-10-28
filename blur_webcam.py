import cv2

model_path = "face_detection_yunet_2023mar.onnx"
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    cap.release()
    raise RuntimeError("No se pudo leer un fotograma de la webcam.")
h, w = frame.shape[:2]
pixeles = 10  # más bajo = más anónimo (prueba 4-8)
margen = 0.35
ultimas_cajas = []     # coordenadas (fx, fy, fw, fh) de las últimas caras vistas
frames_sin_caras = 0   # cuántos frames seguidos llevamos sin detectar nada
MAX_SIN_CARAS = 200     # cuántos frames aguantamos tapando la última posición

##Aqui empezamos con el dectector
detector = cv2.FaceDetectorYN.create(
    model=model_path,
    config="",
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,  # CPU
    target_id=cv2.dnn.DNN_TARGET_CPU,  # CPU
    input_size=(w, h),
    score_threshold=0.5,  ##Confianza
    nms_threshold=0.3,  ##supresion
    top_k=10,  ##Cantidad de detecciones --probar con mas gente --
)

##Aqui tenemos la webcam
while True:
    ret, frame = cap.read()
    if not ret:  ##Si no detecta la camara error
        break

    detector.setInputSize((frame.shape[1], frame.shape[0]))  ##Encuadrar el cuadrado
    _, faces = detector.detect(frame)  ##Donde me pilla la cara

    ## PARTE 1: ACTUALIZAR LA MEMORIA
    if faces is not None:
        ultimas_cajas = []  # vaciamos para no acumular cajas viejas
        for face in faces:
            fx, fy, fw, fh = map(int, face[:4])
            ultimas_cajas.append((fx, fy, fw, fh))  # guardamos solo coordenadas
        frames_sin_caras = 0  # acabamos de ver una cara: contador a cero
    else:
        frames_sin_caras += 1  # += suma
        if frames_sin_caras > MAX_SIN_CARAS:
            ultimas_cajas = []  # llevamos demasiado sin verte: olvidamos

    ## PARTE 2: PIXELAR LO QUE HAYA EN LA MEMORIA (no lo que haya en faces)
    for (fx, fy, fw, fh) in ultimas_cajas:
        ## Margenes para que no se vea barbilla ni pelo
        mx = int(fw * margen)
        my = int(fh * margen)

        ## Coordenadas de inicio, sin salir por arriba o por la izquierda
        x1, y1 = max(0, fx - mx), max(0, fy - my)
        ## Coordenadas finales, sin salir por abajo o por la derecha
        x2 = min(frame.shape[1], fx + fw + mx)
        y2 = min(frame.shape[0], fy + fh + my)

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size:
            alto, ancho = face_roi.shape[:2]
            ## Encogemos a muy pocos pixeles (resize va en orden ancho, alto)
            pequeña = cv2.resize(
                face_roi, (pixeles, pixeles), interpolation=cv2.INTER_AREA
            )
            ## Estiramos al tamaño original sin suavizar
            pixelada = cv2.resize(
                pequeña, (ancho, alto), interpolation=cv2.INTER_NEAREST
            )
            frame[y1:y2, x1:x2] = pixelada  # la imagen se pega al frame, no se guarda

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

##Para limpiar todo
cap.release()
cv2.destroyAllWindows()