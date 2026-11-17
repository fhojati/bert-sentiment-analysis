
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load our existing trained model
tokenizer = AutoTokenizer.from_pretrained("./sentiment_model_v3")
model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model_v3")
model.to(device)
model.eval()

# Load independent test data
dataset = load_dataset("surrey-nlp/BESSTIE-CW-26")
test_data = dataset["test"]

# Keep only sarcastic examples
sarcastic = [
    row for row in test_data
    if int(row["Sarcasm"]) == 1
]

print("Sarcastic test samples:", len(sarcastic))

predictions = []
labels = []

for start in range(0, len(sarcastic), 4):
    batch = sarcastic[start:start + 4]

    texts = [row["text"] for row in batch]
    labels.extend(int(row["Sentiment"]) for row in batch)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions.extend(
        outputs.logits.argmax(dim=1).cpu().tolist()
    )

print("Sarcasm Sentiment Accuracy:",
      round(accuracy_score(labels, predictions) * 100, 2), "%")

print("Sarcasm Macro F1:",
      round(f1_score(labels, predictions, average="macro"), 4))


from sklearn.metrics import classification_report, confusion_matrix

print("\nSentiment distribution:")
print("Actual Negative:", labels.count(0))
print("Actual Positive:", labels.count(1))

print("\nClassification Report:")
print(classification_report(
    labels,
    predictions,
    labels=[0, 1],
    target_names=["Negative", "Positive"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(labels, predictions, labels=[0, 1]))
