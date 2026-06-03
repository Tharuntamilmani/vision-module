from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    root=r"F:\insightface_models"
)

app.prepare(ctx_id=-1)

print("InsightFace loaded successfully!")