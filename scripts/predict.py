import argparse
from src.indic_sentiment.inference import load_inference_model, predict_text

model, tokenizer = load_inference_model()

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, required=True)

args = parser.parse_args()

label, _ = predict_text(args.text, model, tokenizer)

print(label)