import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load saved model
tokenizer = AutoTokenizer.from_pretrained("./sentiment_model_v3")
model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model_v3")

model.to(device)
model.eval()

# Load test dataset
dataset = load_dataset("stanfordnlp/imdb")
test_data = dataset["test"].shuffle(seed=42).select(range(500))

# Tokenize test data
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128
    )

test_data = test_data.map(tokenize_function, batched=True)

test_data.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

test_loader = DataLoader(
    test_data,
    batch_size=4,
    collate_fn=lambda batch: {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0
        ),
        "label": torch.stack([item["label"] for item in batch])
    }
)

predictions = []
labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device)
        )

        preds = torch.argmax(outputs.logits, dim=1)

        predictions.extend(preds.cpu().tolist())
        labels.extend(batch["label"].tolist())

accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test F1-score: {f1:.4f}")