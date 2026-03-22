from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2

app = FastAPI()

model = None

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("grape_model.h5")
    return model

classes = ["black_rot", "esca", "leaf_blight", "healthy"]

SPRAY_MAP = {
    "black_rot": {"chemical": "Mancozeb", "dose": "2.5 g/L"},
    "esca": {"chemical": "Carbendazim", "dose": "1 g/L"},
    "leaf_blight": {"chemical": "Copper Oxychloride", "dose": "2.5 g/L"},
    "healthy": {"chemical": "No spray needed", "dose": "-"}
}

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    model = get_model()
    prediction = model.predict(image)

    class_id = np.argmax(prediction)
    confidence = float(np.max(prediction))

    disease = classes[class_id]

    return {
        "disease": disease,
        "confidence": round(confidence, 2),
        "recommendation": SPRAY_MAP[disease]
    }
