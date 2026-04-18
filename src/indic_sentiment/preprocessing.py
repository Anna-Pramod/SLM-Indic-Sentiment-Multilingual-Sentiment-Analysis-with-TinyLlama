from datasets import load_dataset

LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

def normalize_label(label):
    if label in LABEL_MAP:
        return LABEL_MAP[label]
    return None

def load_and_prepare():
    ds = load_dataset("dhruv0808/indic_sentiment_analyzer")["train"]

    def process(example):
        label = normalize_label(example.get("label"))
        text = example.get("text")

        if label is None or text is None:
            return None

        return {
            "text": f"You are a sentiment classifier.\n\nText: {text}\nAnswer: {label}"
        }

    ds = ds.map(process)
    ds = ds.filter(lambda x: x is not None)

    return ds