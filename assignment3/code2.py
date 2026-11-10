import cv2
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)
img = cv2.imread(".jpg")
if img is None:
    raise FileNotFoundError("input image not found")
faces = app.get(img)
for index, face in enumerate(faces):
    bbox = face.bbox.astype(int).tolist()
    kps = face.kps.astype(int).tolist()
    print(f"face {index}: bbox={bbox}")
    print(f"face {index}: keypoints={kps}")