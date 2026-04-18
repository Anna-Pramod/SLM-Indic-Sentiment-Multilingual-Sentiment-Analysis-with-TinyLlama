import gradio as gr
from src.indic_sentiment.inference import load_inference_model, predict_text

# Load model once (important for performance)
model, tokenizer = load_inference_model()

def predict_sentiment(text):
    if not text.strip():
        return "Please enter some text.", ""

    label, raw_output = predict_text(text, model, tokenizer)
    return label, raw_output


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🌍 Multilingual Sentiment Analyzer")
    gr.Markdown("Enter text in English or Indic languages")

    with gr.Row():
        input_text = gr.Textbox(label="Input Text", lines=3)

    with gr.Row():
        output_label = gr.Textbox(label="Predicted Sentiment")
        raw_output = gr.Textbox(label="Raw Model Output")

    submit_btn = gr.Button("Analyze")

    submit_btn.click(
        predict_sentiment,
        inputs=input_text,
        outputs=[output_label, raw_output],
    )

# Launch app
if __name__ == "__main__":
    demo.launch()