import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import threading
from fastapi import FastAPI, UploadFile, File
import uvicorn
import tempfile

# -----------------------------
# Load YOLO Model
# -----------------------------
model = YOLO("best.pt")

# -----------------------------
# FastAPI Backend for n8n
# -----------------------------
api = FastAPI()

@api.post("/predict")
async def predict_api(image: UploadFile = File(...)):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes))

    # YOLO prediction
    results = model(img)

    predictions = []
    asymmetry_list = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1

            asym = abs(width - height) / max(width, height) * 100 if max(width, height) else 0
            asymmetry_list.append(asym)

            predictions.append({
                "class": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": [x1, y1, x2, y2],
                "asymmetry_percentage": round(asym, 2)
            })

    avg_asym = round(sum(asymmetry_list) / len(asymmetry_list), 2) if asymmetry_list else 0

    return {
        "predictions": predictions,
        "count": len(predictions),
        "average_asymmetry_percentage": avg_asym
    }


# -----------------------------
# Start FastAPI in Background
# -----------------------------
def start_fastapi():
    uvicorn.run(api, host="0.0.0.0", port=8000)

threading.Thread(target=start_fastapi, daemon=True).start()


# -----------------------------
# STREAMLIT FRONTEND
# -----------------------------
st.title("YOLO Asymmetry Detection")
uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        img.save(tmp.name)
        results = model(tmp.name)

    st.subheader("Predictions")

    for result in results:
        for box in result.boxes:
            st.write({
                "class": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })
