import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model_v3")
tokenizer = AutoTokenizer.from_pretrained("./sentiment_model_v3")

model.to(device)
model.eval()


def predict_sentiment(text):
    if not text.strip():
        return {"Positive": 0.0, "Negative": 0.0}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    return {
        "Negative": float(probabilities[0]),
        "Positive": float(probabilities[1])
    }


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Enter a movie review...",
        label="Your Review"
    ),
    outputs=gr.Label(label="Sentiment Prediction"),
    title="AI Sentiment Analysis with BERT",
    description="Analyze the sentiment of movie reviews using a fine-tuned BERT model."
)

demo.launch()