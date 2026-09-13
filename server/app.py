"""FastAPI inference endpoint for the quantized sketch classifier."""
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent.parent / "models" / "sketch_oracle_quant.tflite"
CLASSES_PATH = Path(__file__).parent.parent / "models" / "classes.txt"

app = FastAPI(title="Sketch Oracle")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
_input_detail = interpreter.get_input_details()[0]
_output_detail = interpreter.get_output_details()[0]
_classes = [line.strip() for line in CLASSES_PATH.read_text().splitlines() if line.strip()]


class PredictRequest(BaseModel):
    pixels: list[int] = Field(..., min_length=784, max_length=784, description="28x28 grayscale, row-major, 0-255")


class Guess(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    predictions: list[Guess]


@app.get("/health")
def health():
    return {"status": "ok", "classes": len(_classes)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    bitmap = np.array(req.pixels, dtype=np.uint8).reshape(1, 28, 28, 1)

    interpreter.set_tensor(_input_detail["index"], bitmap)
    interpreter.invoke()
    probs = interpreter.get_tensor(_output_detail["index"])[0]

    top5_idx = np.argsort(probs)[-5:][::-1]
    predictions = [Guess(label=_classes[i], confidence=float(probs[i])) for i in top5_idx]
    return PredictResponse(predictions=predictions)
