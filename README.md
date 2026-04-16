# SLM-Indic-Sentiment-Multilingual-Sentiment-Analysis-with-TinyLlama
This project provides a production-ready framework for fine-tuning TinyLlama (1.1B) to perform sentiment analysis on product reviews across 11 Indian languages (plus English) . By leveraging Unsloth and LoRA, this repository enables high-performance fine-tuning on consumer-grade hardware with significantly reduced memory requirements
1. Project Purpose & Scope
The goal is to adapt a Small Language Model (SLM) to understand nuanced sentiments in low-resource and code-mixed Indian contexts (e.g., Hindi, Telugu, Tamil, Kannada, Bengali, and more)
. Using the indic_sentiment_analyzer dataset, the model is trained to classify reviews into Positive, Negative, or Neutral categories
.
Key advantages of this approach include:
Efficiency: Uses Unsloth's custom Triton kernels to train 2–5x faster while using 70–80% less VRAM
.
Privacy: The entire pipeline can run on local hardware or private cloud instances, ensuring data remains secure
.
Edge-Ready: TinyLlama's compact size (1.1B parameters) makes it ideal for offline deployment on mobile or edge devices
.
2. System Architecture (FTI Pipeline)
This repository follows the Feature/Training/Inference (FTI) design pattern to ensure modularity and scalability
.
Feature Pipeline: Extracts raw reviews from the indic_sentiment_analyzer and transforms them into a structured Alpaca instruction-following format
. These are stored as versioned ZenML artifacts
.
Training Pipeline: Loads the base TinyLlama model, applies LoRA (Low-Rank Adaptation) adapters to all linear layers (q_proj, v_proj, etc.), and executes Supervised Fine-Tuning (SFT)
.
Inference Pipeline: Serves the fine-tuned model via a FastAPI business microservice, potentially utilizing RAG (Retrieval-Augmented Generation) for added context
.
3. Setup Guide
Prerequisites
Python: 3.11.8 (managed via pyenv)
.
Hardware: Single GPU with at least 8GB-16GB VRAM (e.g., NVIDIA T4 or L4)
.
Dependency Management: Poetry
.
Installation
Clone the Repository:
Install Dependencies:
Configure Environment: Create a .env file based on .env.example and add your Hugging Face and Comet ML tokens
.
4. Usage Instructions
Data Preprocessing
The llm_engineering/application/preprocessing.py script maps the dataset to structured prompts
.
# Format: Analyze the sentiment of this review.
# Input: [Sentence in Indian Language]
# Response: [Positive/Negative/Neutral]
Training
Execute the training pipeline using the centralized YAML configuration
:
poetry run python -m tools.run --run-training --config configs/training.yaml
Hyperparameters: Default settings include a Learning Rate of 2e-4, Rank (r) of 32, and 1 Epoch for rapid iteration
.
Exporting
Export the fine-tuned LoRA adapters or convert the model to GGUF format for use in Ollama
:
python tools/export.py --format gguf
