from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load YOLO model
model = YOLO("best.pt")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes))

    results = model(img)

    asymmetry_list = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1

            # calculate asymmetry
            asym = abs(width - height) / max(width, height) * 100 if max(width, height) else 0
            asymmetry_list.append(asym)

    # Overall asymmetry
    avg_asymmetry = round(sum(asymmetry_list) / len(asymmetry_list), 2) if asymmetry_list else 0

    # Classification threshold
    THRESHOLD = 10  # adjust as needed

    if avg_asymmetry > THRESHOLD:
        return "DEFORMED"
    else:
        return "NORMAL"

@app.get("/")
def home():
    return "YOLO asymmetry API is running!"
