
import json
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# Load V1 model
model_path = "./saved_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path
)

model.to(device)

# Load V3 training data
with open("train_data_v3.jsonl", "r", encoding="utf-8") as f:
    training_data = [json.loads(line) for line in f]

print("Training samples:", len(training_data))

# Tokenize and pad each batch
def collate_fn(batch):
    texts = [item["text"] for item in batch]
    labels = [int(item["label"]) for item in batch]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs["labels"] = torch.tensor(
        labels, dtype=torch.long
    )

    return inputs

train_loader = DataLoader(
    training_data,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=1e-5
)

# Train for one epoch
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

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0
    )

    optimizer.step()
    total_loss += loss.item()

    if (step + 1) % 25 == 0:
        print(
            f"Step {step + 1}/{len(train_loader)} "
            f"| Loss: {total_loss / (step + 1):.4f}",
            flush=True
        )

# Save V3 separately
model.save_pretrained("./sentiment_model_v3")
tokenizer.save_pretrained("./sentiment_model_v3")

print("Training V3 completed!")
print("Model saved successfully!")
