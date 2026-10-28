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
ultimacara= []


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
    if faces is not None:
        for face in faces:
            fx, fy, fw, fh = map(int, face[:4])
            ## Margenes para la cara no se vea barbilla ni pelo
            mx = int(fw * margen)
            my = int(fh * margen)
            ##Aqui ya lo añadimos para que aparezca

            ## Coordenadas de inicio de la cara, sin salir por arriba o por la izquierda
            x1, y1 = max(0, fx - mx), max(0, fy - my)
            ## Coordenadas finales de la cara, sin salir por abajo o por la derecha
            x2 = min(frame.shape[1], fx + fw + mx)
            y2 = min(frame.shape[0], fy + fh + my)
            ## Recortamos del frame solo la parte donde esta la cara (primero filas y luego columnas)
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size:
                alto, ancho = face_roi.shape[:2]
                ## Encogemos a muy pocos pixeles (resize va en orden ancho, alto)
                pequeña = cv2.resize(
                    face_roi, (pixeles, pixeles), interpolation=cv2.INTER_AREA
                )
                ## Estiramos al tamaño original sin suavizar, para que se vean los bloques
                pixelada = cv2.resize(
                    pequeña, (ancho, alto), interpolation=cv2.INTER_NEAREST
                )
                frame[y1:y2, x1:x2] = pixelada

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

##Para limpiar todo
cap.release()
cv2.destroyAllWindows()
