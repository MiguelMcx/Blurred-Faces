# Blurred Faces

Programa en Python que abre la webcam y difumina las caras que detecta en tiempo real. Lo hice para aprender cómo funciona la detección facial con OpenCV.

## Cómo funciona

1. Lee un fotograma de la webcam.
2. Detecta las caras con el modelo YuNet (`face_detection_yunet_2023mar.onnx`), que ya está incluido en el repositorio.
3. Recorta la zona de cada cara y le aplica un desenfoque gaussiano.
4. Muestra el resultado en una ventana.

Funciona solo con CPU y detecta hasta 10 caras a la vez.

## Requisitos

- Python 3
- OpenCV

```bash
pip install opencv-python
```

## Uso

```bash
git clone https://github.com/MiguelMcx/Blurred-Faces.git
cd Blurred-Faces
python blur_webcam.py
```

Pulsa `q` para cerrar la ventana.

## Ajustes

En `blur_webcam.py` se pueden cambiar:

- `score_threshold` (0.9): confianza mínima para aceptar una cara. Si no detecta bien con poca luz, se puede bajar.
- `nms_threshold` (0.3): evita detecciones repetidas de la misma cara.
- `top_k` (10): máximo de caras por fotograma.
- `cv2.GaussianBlur(face_roi, (15, 15), 87)`: el primer valor es el tamaño del kernel; si lo subes, el desenfoque es más fuerte.

## Tecnologías

Python, OpenCV y el modelo YuNet del [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
