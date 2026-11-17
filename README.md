
# BERT Sentiment Analysis with Sarcasm Evaluation

An NLP project using a fine-tuned BERT model to classify English text as positive or negative, with additional evaluation of sarcastic and context-dependent language.

## Project Overview

This project explores sentiment classification using BERT and PyTorch.

The model was initially fine-tuned on IMDb movie reviews and subsequently improved using additional sentiment-labelled data.

An interactive Gradio application allows users to enter text and view sentiment predictions.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- BERT (bert-base-uncased)
- Scikit-learn
- Hugging Face Datasets
- Gradio
- NVIDIA CUDA

## Model Performance

| Metric | Original Model | Final Model (V3) |
|--------|----------------|------------------|
| IMDb Accuracy | 82.60% | 86.00% |
| IMDb F1-score | 0.8228 | 0.8583 |

Results were obtained using 500 IMDb test reviews.

## Sarcasm Evaluation

The model was also evaluated on sarcastic expressions using the BESSTIE dataset.

Final model results:
- Sarcastic test samples: 305
- Accuracy: 88.20%
- Macro F1-score: 0.5591

The sarcasm test set was highly imbalanced, containing 297 negative and only 8 positive examples.

These results highlight limitations in interpreting sarcasm and implicit sentiment. Sarcasm accuracy should not be interpreted as evidence of reliable sarcasm understanding.

## Application

An interactive interface was developed with Gradio.

Users can:
- Enter English text
- Predict positive or negative sentiment
- View model prediction probabilities

## Project Files

- `main.py` - Initial BERT training
- `prepare_v3.py` - Prepare combined training data
- `train_v3.py` - Fine-tune the V3 model
- `evaluate.py` - Evaluate IMDb performance
- `evaluate_sarcasm.py` - Evaluate sarcastic sentiment
- `app.py` - Gradio interface

## Running the Application

Install the required Python dependencies and generate the model checkpoints using the training scripts.

Then run:

```bash
python app.py
```

The application runs locally using Gradio.

Note: Trained model checkpoints and local datasets are excluded from this repository due to file size.

## Limitations

The model performs well on many conventional sentiment classification examples but can misinterpret sarcasm, especially when positive wording conveys negative meaning.

Further research is needed to improve robustness and evaluate performance on more diverse, balanced datasets.
