import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load your trained BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("./sentiment_model_v2")
model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model_v2")
model.to(device)
model.eval()

# Test sentences: (text, expected sentiment)
test_sentences = [
    ("Oh great, another movie that wasted two hours of my life!", "Negative"),
    ("I just love watching terrible movies. What a wonderful waste of time!", "Negative"),
    ("Thanks for ruining my evening. Amazing job!", "Negative"),
    ("Fantastic! The movie was so boring I almost fell asleep.", "Negative"),
    ("What a masterpiece! I've never been so disappointed.", "Negative"),
    ("I can't believe how good this movie was. I expected to hate it!", "Positive"),
    ("Well, that was surprisingly enjoyable!", "Positive"),
    ("I thought this would be awful, but it was actually brilliant.", "Positive"),
    ("Oh wonderful, another predictable ending. How original!", "Negative"),
    ("Great, two hours of my life I'll never get back.", "Negative"),
]

correct = 0

for text, expected in test_sentences:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_id = torch.argmax(outputs.logits, dim=1).item()

    prediction = "Positive" if predicted_id == 1 else "Negative"
    is_correct = prediction == expected
    correct += int(is_correct)

    print(f"\nText: {text}")
    print(f"Expected: {expected}")
    print(f"Predicted: {prediction}")
    print(f"Result: {'CORRECT' if is_correct else 'WRONG'}")

print(f"\nTest result: {correct}/{len(test_sentences)} correct")