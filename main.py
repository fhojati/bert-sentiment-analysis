from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
# Load dataset
dataset = load_dataset("stanfordnlp/imdb")

# Select a small subset
train_data = dataset["train"].shuffle(seed=42).select(range(2000))
split = train_data.train_test_split(test_size=0.2, seed=42)
train_data = split["train"]
validation_data = split["test"]
test_data = dataset["test"].shuffle(seed=42).select(range(500))

# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize text
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_data = train_data.map(tokenize_function, batched=True)
validation_data = validation_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)
print("Training samples:", len(train_data))
print("Test samples:", len(test_data))
print(train_data[0]["input_ids"][:20])
# Load BERT model for sentiment classification
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Model loaded successfully!")
print("Device:", next(model.parameters()).device)
# Prepare datasets for PyTorch
columns = ["input_ids", "attention_mask", "label"]

train_data.set_format(type="torch", columns=columns)
validation_data.set_format(type="torch", columns=columns)

# Create DataLoaders
train_loader = DataLoader(
    train_data,
    batch_size=4,
    shuffle=True
)

validation_loader = DataLoader(
    validation_data,
    batch_size=4
)

print("Training batches:", len(train_loader))
print("Validation batches:", len(validation_loader))
from torch.optim import AdamW

# Training settings
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 1

# Training loop
model.train()

total_loss = 0

for step, batch in enumerate(train_loader):

    optimizer.zero_grad()

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

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

print("Training completed!")
# Save fine-tuned BERT model
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

print("Model saved successfully!")
# Evaluate the trained model on test data
model.eval()

test_data.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

test_loader = DataLoader(
    test_data,
    batch_size=4,
    shuffle=False
)

all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predictions = torch.argmax(outputs.logits, dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(batch["label"].tolist())

accuracy = accuracy_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1-score: {f1:.4f}")