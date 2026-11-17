
import random
import json
from collections import Counter
from datasets import load_dataset

random.seed(42)

# Original IMDb training data
imdb = load_dataset("stanfordnlp/imdb")

imdb_subset = imdb["train"].shuffle(seed=42).select(range(2000))
imdb_train = imdb_subset.train_test_split(
    test_size=0.2, seed=42
)["train"]

samples = [
    {"text": row["text"], "label": int(row["label"])}
    for row in imdb_train
]

# Sarcasm dataset - training portion only
besstie = load_dataset("surrey-nlp/BESSTIE-CW-26")
train = besstie["train"]

sarcastic_negative = [
    row for row in train
    if int(row["Sarcasm"]) == 1
    and int(row["Sentiment"]) == 0
]

sarcastic_positive = [
    row for row in train
    if int(row["Sarcasm"]) == 1
    and int(row["Sentiment"]) == 1
]

normal_positive = [
    row for row in train
    if int(row["Sarcasm"]) == 0
    and int(row["Sentiment"]) == 1
]

# Add 200 negative and 200 positive examples
extra = (
    random.sample(sarcastic_negative, 200)
    + sarcastic_positive
    + random.sample(normal_positive, 200 - len(sarcastic_positive))
)

samples.extend(
    {"text": row["text"], "label": int(row["Sentiment"])}
    for row in extra
)

random.shuffle(samples)

with open("train_data_v3.jsonl", "w", encoding="utf-8") as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print("Total training samples:", len(samples))
print("Class distribution:", Counter(x["label"] for x in samples))
print("V3 dataset saved successfully!")
