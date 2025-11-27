from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load the YOLO model
model = YOLO("best.pt")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes))

    results = model(img)

    predictions = []
    asymmetry_list = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1

            # Calculate asymmetry percentage
            asym = abs(width - height) / max(width, height) * 100 if max(width, height) > 0 else 0
            asymmetry_list.append(asym)

            predictions.append({
                "class": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "asymmetry_percentage": round(asym, 2)
            })

    avg_asymmetry = round(sum(asymmetry_list) / len(asymmetry_list), 2) if asymmetry_list else 0

    return {
        "predictions": predictions,
        "count": len(predictions),
        "average_asymmetry_percentage": avg_asymmetry
    }

@app.get("/")
def home():
    return {"status": "YOLO asymmetry API is running!"}
