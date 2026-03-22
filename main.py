from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import tensorflow as tf

app = FastAPI()

# Lazy load interpreter
interpreter = None

def get_interpreter():
    global interpreter
    if interpreter is None:
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
    return interpreter


# Class labels (EDIT if needed)
class_names = ["black_rot", "esca", "healthy", "leaf_blight"]


@app.get("/")
def home():
    return {"message": "Grape Disease Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Invalid image"}

        # Resize (IMPORTANT: match training size)
        image = cv2.resize(image, (224, 224))

        # 🔥 Add smoothing (stability improvement)
        image = cv2.GaussianBlur(image, (5, 5), 0)

        # Normalize
        image = image / 255.0

        # Expand dims
        image = np.expand_dims(image, axis=0).astype(np.float32)

        # Load interpreter
        interpreter = get_interpreter()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Predict
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # 🔥 Stability logic
        probs = prediction[0]
        max_index = int(np.argmax(probs))
        confidence = float(probs[max_index])

        # Threshold check
        THRESHOLD = 0.6

        if confidence < THRESHOLD:
            return {
                "disease": "uncertain",
                "confidence": round(confidence, 3),
                "message": "Image unclear. Please capture a clear leaf image."
            }

        predicted_class = class_names[max_index]

        return {
            "disease": predicted_class,
            "confidence": round(confidence, 3)
        }

    except Exception as e:
        return {"error": str(e)}