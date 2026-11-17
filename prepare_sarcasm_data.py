
from datasets import load_dataset

dataset = load_dataset("surrey-nlp/BESSTIE-CW-26")

print(dataset)

print("\nExamples of sarcastic sentences:")

count = 0
for example in dataset["train"]:
    if example["Sarcasm"] == 1:
        print("\nText:", example["text"][:300])
        print(
            "Sentiment:",
            "Positive" if example["Sentiment"] == 1 else "Negative"
        )
        count += 1

    if count == 5:
        break

from collections import Counter

train_data = dataset["train"]

print("All training sentiments:")
print(Counter(train_data["Sentiment"]))

sarcastic = train_data.filter(
    lambda row: row["Sarcasm"] == 1
)

print("Sarcastic training sentiments:")
print(Counter(sarcastic["Sentiment"]))
