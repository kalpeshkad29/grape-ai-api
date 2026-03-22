from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import tensorflow as tf

app = FastAPI()

# Load TFLite model (lazy loading)
interpreter = None

def get_interpreter():
    global interpreter
    if interpreter is None:
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
    return interpreter


# Class labels (change if needed)
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

        # Resize (IMPORTANT: match your training size)
        image = cv2.resize(image, (224, 224))

        # Normalize
        image = image / 255.0

        # Expand dimensions
        image = np.expand_dims(image, axis=0).astype(np.float32)

        # Load interpreter
        interpreter = get_interpreter()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], image)

        # Run prediction
        interpreter.invoke()

        # Get output
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # Get result
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return {
            "disease": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}