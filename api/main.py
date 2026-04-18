from fastapi import FastAPI
from api.schemas import PredictRequest, PredictResponse
from src.indic_sentiment.inference import load_inference_model, predict_text

app = FastAPI(title="Indic Sentiment API", version="1.0.0")

model, tokenizer = load_inference_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    label, raw_output = predict_text(
        text=request.text,
        model=model,
        tokenizer=tokenizer,
    )
    return PredictResponse(
        label=label,
        raw_output=raw_output,
        model_name="annavivin/tinyllama-indic-sentiment-full",
    )