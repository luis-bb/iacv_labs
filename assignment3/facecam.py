import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível aceder à câmara.")
    exit()

print("Câmara aberta com sucesso! Pressione 'q' para fechar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao receber o frame. A terminar...")

    faces = app.get(frame)
    for index, face in enumerate(faces):
        bbox = face.bbox.astype(int).tolist()
        kps = face.kps.astype(int).tolist()
        print(f"face {index}: bbox={bbox}")
        print(f"face {index}: keypoints={kps}")

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    cv2.imshow("detec", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
