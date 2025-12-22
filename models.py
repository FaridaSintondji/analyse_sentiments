# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 15:58:09 2025

@author: faris
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
import numpy as np

# Charger VADER
vader_analyzer = SentimentIntensityAnalyzer()

def analyze_vader(text):
    score = vader_analyzer.polarity_scores(text)['compound']
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

# Fonction générique pour Hugging Face models
class HFModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = torch.argmax(probs, dim=1).item()
        # Assumer label 0 = negative, 1 = neutral, 2 = positive (à adapter selon modèle)
        mapping = {0:"negative", 1:"neutral", 2:"positive"}
        return mapping.get(label, "neutral")

# Initialiser les modèles
bertweet_model = HFModel("vinai/bertweet-base-sentiment-analysis")
mbert_model = HFModel("nlptown/bert-base-multilingual-uncased-sentiment")
xlm_model = HFModel("cardiffnlp/twitter-xlm-roberta-base-sentiment")
