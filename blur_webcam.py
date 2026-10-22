import cv2

model_path = "face_detection_yunet_2023mar.onnx"
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    cap.release()
    raise RuntimeError("No se pudo leer un fotograma de la webcam.")
h, w = frame.shape[:2]
##Aqui empezamos con el dectector

detector = cv2.FaceDetectorYN.create(
    model=model_path,
    config="",
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,  # CPU
    target_id=cv2.dnn.DNN_TARGET_CPU,  # CPU
    input_size=(w, h),
    score_threshold=0.9,  ##Confianza
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
    if (
        faces is not None
    ):  ## YA aqui es me dectecta cara que me ponga el cuadrado o como inteto que me desenfoque
        for face in faces:
            x, y, w, h = map(int, face[:4])
            ## Coordenadas de inicio de la cara, sin salir por arriba o por la izquierda
            x1, y1 = max(0, x), max(0, y)
            ## Coordenadas finales de la cara, sin salir por abajo o por la derecha
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            ## Recortamos del frame solo la parte donde esta la cara (primero filas y luego columnas)
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size:
                ## Desenfocamos la parte recortada y la ponemos de nuevo en el frame
                blurred_face = cv2.GaussianBlur(face_roi, (15, 15), 87)
                frame[y1:y2, x1:x2] = blurred_face

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

##Para limpiar todo basicamente como haciamos en java
cap.release()
cv2.destroyAllWindows()
