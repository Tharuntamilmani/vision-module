import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

# -----------------------------
# Initialize InsightFace
# -----------------------------
app = FaceAnalysis(
    name="buffalo_l",
    root=r"F:\insightface_models"
)

app.prepare(ctx_id=-1)

# -----------------------------
# Load Known Faces
# -----------------------------
known_embeddings = []

faces_dir = "known_faces"

print("Loading known faces...")

for file in os.listdir(faces_dir):

    if file.lower().endswith((".png", ".jpg", ".jpeg")):

        path = os.path.join(faces_dir, file)

        img = cv2.imread(path)

        faces = app.get(img)

        if len(faces) > 0:

            embedding = faces[0].embedding
            known_embeddings.append(embedding)

            print(f"Loaded: {file}")

        else:
            print(f"No face found: {file}")

if len(known_embeddings) == 0:
    raise Exception("No faces loaded!")

print(f"\nLoaded {len(known_embeddings)} faces.\n")

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

THRESHOLD = 0.55

while True:

    ret, frame = cap.read()

    if not ret:
        break

    faces = app.get(frame)

    for face in faces:

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        current_embedding = face.embedding

        best_score = -1

        for known_embedding in known_embeddings:

            similarity = np.dot(
                current_embedding,
                known_embedding
            ) / (
                np.linalg.norm(current_embedding)
                * np.linalg.norm(known_embedding)
            )

            best_score = max(
                best_score,
                similarity
            )

        if best_score > THRESHOLD:
            name = "Tharun"
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            f"{name} ({best_score:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow(
        "Jarvis Face Recognition",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()