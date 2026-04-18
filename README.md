# Multilingual Sentiment Analysis with TinyLlama

A compact, instruction-tuned sentiment classification project built around **TinyLlama**, **Unsloth**, and the **`dhruv0808/indic_sentiment_analyzer`** dataset.

This repository documents an end-to-end workflow for:
- preparing a multilingual sentiment dataset for instruction tuning,
- fine-tuning a Small Language Model under tight hardware limits,
- evaluating the model on held-out data,
- and exposing the model through an interactive **Gradio** demo.

## Project Summary

This project explores whether a **small language model (SLM)** can be adapted for multilingual sentiment classification across Indic languages while remaining practical on commodity hardware such as a **15 GB NVIDIA T4 GPU on Google Colab**.

The final system is based on:
- **Base model:** TinyLlama / TinyLlama-chat family
- **Optimization stack:** Unsloth + LoRA / QLoRA
- **Dataset:** `dhruv0808/indic_sentiment_analyzer`
- **Task:** classify text into exactly one of **Positive**, **Negative**, or **Neutral**
- **Deployment target:** Gradio interface hosted on **Hugging Face Spaces**

## Why this project matters

Multilingual sentiment analysis is often treated as a classification problem solved with encoder models. This project instead investigates a **generative SLM approach** that reframes sentiment as an instruction-following task. That makes the workflow especially useful for:
- rapid prototyping with instruction-tuned LLM pipelines,
- resource-constrained experimentation,
- multilingual use cases where deployment footprint matters,
- and educational demonstration of end-to-end SLM adaptation.

## Dataset

**Dataset:** `dhruv0808/indic_sentiment_analyzer`

The dataset is multilingual and contains sentiment labels across multiple Indian languages along with English. The target labels are normalized into the three canonical classes below:
- `Positive`
- `Negative`
- `Neutral`

### Dataset considerations

A few practical issues emerged during development:
- some rows contain missing labels,
- the dataset does not expose a native `language` column in the loaded schema,
- and sequence lengths cluster close to 128 tokens after prompt formatting.

Because of this, the preprocessing pipeline includes:
- filtering invalid rows,
- defensive label normalization,
- prompt-completion conversion,
- and token-length sanity checks before training.

## Model and training strategy

### Model choice

TinyLlama was selected as a **resource-aware baseline**. It is small enough to fine-tune in Colab while still being expressive enough to follow short, structured prompts for classification.

Two variants were considered during experimentation:
- a **base TinyLlama** model,
- and a **chat-tuned TinyLlama** model.

For this task, the chat-tuned variant is the better fit because sentiment classification is framed as an **instruction-following generation task**, not open-ended language modeling.

### Fine-tuning approach

The model is fine-tuned with:
- **4-bit loading** for memory efficiency,
- **LoRA / QLoRA adapters** to avoid full-model updates,
- **completion-only supervision** so the model learns only the label portion of the response,
- and **short sequence lengths** appropriate for sentiment prompts.

### Hardware constraint

A key design requirement for this project was that training should remain feasible on a **15 GB T4 GPU** in Google Colab. To make that work, the project uses:
- 4-bit quantization,
- small batch sizes,
- gradient accumulation,
- gradient checkpointing,
- paged 8-bit optimization,
- and tight sequence-length control.

## Prompt format

The dataset is converted into a prompt-completion format similar to:

```text
You are a sentiment classifier.
Classify the sentiment of the text as exactly one of: Positive, Negative, or Neutral.

Text: <input sentence>
Answer: <label>
```

This design is important because it aligns the task with the behavior of chat-tuned causal language models while keeping the supervised target extremely short.

## Evaluation approach

The evaluation pipeline focuses on:
- **overall accuracy**,
- **macro F1**,
- **weighted F1**,
- **invalid prediction rate**,
- and **manual multilingual sanity checks**.

Because the dataset does not provide an explicit language column, per-language analysis must be estimated through:
- inferred language identification on validation texts, or
- manually curated multilingual test examples.

## Interactive demo

A lightweight Gradio application is used to expose the model through a simple:

**Input text -> Predicted sentiment**

The demo is intended for:
- non-technical users,
- project showcasing,
- and quick qualitative testing across languages.

The deployment path is documented in [`docs/deployment.md`](docs/deployment.md).

## Hugging Face artifacts

### Fine-tuned model
- `annavivin/tinyllama-indic-sentiment-full`

### Intended app flow
- user enters text in English or an Indic language,
- the model generates one of the three allowed labels,
- the application extracts the final answer segment,
- and the UI returns a clean sentiment output.

## Repository structure

```text
.
├── README.md
├── app.py
├── requirements.txt
└── docs/
    ├── training_and_evaluation.md
    └── deployment.md
```

## Key lessons from the project

This project surfaced a few important engineering lessons:

1. **Base models are not enough for instruction-style classification.**  
   Chat-tuned checkpoints behave much better for structured outputs.

2. **Loss alone is not sufficient.**  
   A falling training loss can still hide instability or generation failure, so inference checks are essential.

3. **Prompt extraction matters.**  
   If the app reads the entire generated string rather than the final answer segment, it can misreport the label even when the model output is correct.

4. **Small models require disciplined optimization.**  
   Learning rate, packing, sequence length, and adapter rank all materially affect stability on low-memory hardware.

## Current scope

This repository currently documents:
- the project rationale,
- the fine-tuning setup,
- evaluation design,
- and the interactive deployment path.

It is intended to serve both as:
- a reproducible project record,
- and a clear portfolio-ready explanation of the system.

## Future improvements

Planned extensions include:
- stronger validation reporting with full experiment logs,
- per-language analysis with language identification,
- improved UI styling for the Hugging Face Space,
- confidence calibration or constrained decoding,
- and comparison against stronger multilingual SLMs.

## Acknowledgements

This project builds on the open-source ecosystems around:
- Hugging Face Transformers,
- Hugging Face Datasets,
- Gradio,
- Unsloth,
- and the maintainers of the Indic sentiment dataset.

## License

Add a license file if you intend to open-source the project more formally.
