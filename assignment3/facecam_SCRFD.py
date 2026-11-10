
import cv2
import time
from insightface.app import FaceAnalysis

# Inicializar SCRFD (apenas deteção facial)
detector = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection"],
    providers=["CPUExecutionProvider"]
)
detector.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)

# Abrir webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Nao foi possivel abrir a webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.perf_counter()

    # Detetar faces
    faces = detector.get(frame)

    # Desenhar bounding boxes
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        confidence = face.det_score

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

    # Calcular FPS
    fps = 1.0 / max(time.perf_counter() - start, 1e-9)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.imshow("SCRFD - Face Detection", frame)

    # Premir Q para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
