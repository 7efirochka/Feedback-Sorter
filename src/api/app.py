from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import joblib
from pathlib import Path
import numpy as np
import sys

# import torch

# Важно для joblib: добавляем корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(
    title="Sentiment Analysis API",
    description="API для определения тональности текста (нейтральное/хорошее/плохое)",
    version="1.0.0",
)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, example="Dark night put down on the ground")


MODELS = {}
CLASS_NAMES = {0: "нейтральное", 1: "хорошее", 2: "плохое"}


def load_models():
    global MODELS

    models_path = Path("src/saved_models")

    if not models_path.exists():
        print(f"Папка {models_path} не найдена")
        return {}

    models = {}

    for model_file in models_path.glob("*.pkl"):
        model_name = model_file.stem
        model_name = model_name
        try:
            obj = joblib.load(model_file)
            # Проверяем, что загружен объект с методом predict (пайторч заработай умоляю)
            if not hasattr(obj, "predict"):
                continue

            if "pytorch" in model_name.lower():
                if not hasattr(obj, "vectorizer"):
                    continue
            models[model_name] = obj
        except Exception as e:
            print(f"Ошибка загрузки {model_name}: {e}")

    return models


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "models_loaded": len(MODELS),
        "models": list(MODELS.keys()),
    }


@app.on_event("startup")
async def startup_event():
    global MODELS
    MODELS = load_models()
    print(f"Загружено моделей: {len(MODELS)}")


@app.get("/")
def root():
    return {
        "status": "running",
        "models_loaded": len(MODELS),
        "available_models": list(MODELS.keys()),
        "classes": CLASS_NAMES,
    }


def format_output(probs):
    if isinstance(probs, np.ndarray):
        return [round(float(i), 4) for i in probs]
    return [round(i, 4) for i in probs]


@app.post("/predict")
def predict(request: TextRequest):
    if not MODELS:
        raise HTTPException(status_code=503, detail="No models loaded")

    results = {}

    for name, model in MODELS.items():

        try:
            pred_idx = int(model.predict([request.text])[0])
            proba_raw = model.predict_proba([request.text])
            proba_raw = np.squeeze(proba_raw)

            probas = format_output(proba_raw)
            confidence = float(max(probas)) if probas else None

            probabilities = (
                {CLASS_NAMES.get(i, f"class_{i}"): p for i, p in enumerate(probas)}
                if probas
                else {}
            )

            results[name] = {
                "sentiment": CLASS_NAMES.get(pred_idx, f"class_{pred_idx}"),
                "label": pred_idx,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return {"text": request.text, "prediction": results}
