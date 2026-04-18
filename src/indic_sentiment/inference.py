import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.indic_sentiment.prompts import build_prompt

LABELS = ["Positive", "Negative", "Neutral"]

def load_inference_model():
    model_name = "annavivin/tinyllama-indic-sentiment-full"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    return model, tokenizer

def extract_label(text):
    for label in LABELS:
        if label.lower() in text.lower():
            return label
    return "Unknown"

def predict_text(text, model, tokenizer):
    prompt = build_prompt(text)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=3)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer = decoded.split("Answer:")[-1].strip()

    return extract_label(answer), decoded