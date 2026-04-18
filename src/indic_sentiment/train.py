from trl import SFTTrainer
from transformers import TrainingArguments
from src.indic_sentiment.model import load_model
from src.indic_sentiment.preprocessing import load_and_prepare

def train():
    model, tokenizer = load_model()
    dataset = load_and_prepare()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=128,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            max_steps=300,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=10,
            output_dir="outputs",
        ),
    )

    trainer.train()
    model.save_pretrained("outputs")