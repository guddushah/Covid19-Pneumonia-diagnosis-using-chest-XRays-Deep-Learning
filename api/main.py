from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["http://localhost",
           "http://localhost:3000",
            
            ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../saved_models/1")
# prod_model = tf.keras.models.load_model("./saved_models/1")
# beta_model = tf.keras.models.load_model("./saved_models/2")
CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(
        Image.open(BytesIO(data))
    )  # reads the bytes as pillow image and converts into numpy array
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())  # numpy array
    # Images are converted into batches
    img_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {"class": predicted_class, "confidence": float(confidence)}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
