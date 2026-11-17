
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from torch.utils.data import DataLoader
from torch.optim import AdamW

random.seed(42)
torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load existing fine-tuned sentiment model
tokenizer = AutoTokenizer.from_pretrained("./saved_model")
model = AutoModelForSequenceClassification.from_pretrained(
    "./saved_model"
)
model.to(device)

# Load new dataset - training split only
dataset = load_dataset("surrey-nlp/BESSTIE-CW-26")
train_data = dataset["train"]

sarcastic = [
    row for row in train_data
    if int(row["Sarcasm"]) == 1
]

normal = [
    row for row in train_data
    if int(row["Sarcasm"]) == 0
]

random.shuffle(normal)

# Mix sarcastic and non-sarcastic texts
training_samples = sarcastic + normal[:len(sarcastic)]
random.shuffle(training_samples)

print("Training samples:", len(training_samples))
print("Device:", device)

def collate_fn(batch):
    inputs = tokenizer(
        [row["text"] for row in batch],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs["labels"] = torch.tensor(
        [int(row["Sentiment"]) for row in batch],
        dtype=torch.long
    )

    return inputs

train_loader = DataLoader(
    training_samples,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)

optimizer = AdamW(model.parameters(), lr=1e-5)

# Continue fine-tuning
model.train()
total_loss = 0

for step, batch in enumerate(train_loader):
    optimizer.zero_grad()

    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    outputs = model(**batch)
    loss = outputs.loss

    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    if (step + 1) % 20 == 0:
        print(
            f"Step {step + 1}/{len(train_loader)} "
            f"- Loss: {total_loss / (step + 1):.4f}",
            flush=True
        )

# Save separately - preserve original model
model.save_pretrained("./sentiment_model_v2")
tokenizer.save_pretrained("./sentiment_model_v2")

print("Fine-tuning completed!")
print("New model saved successfully!")
