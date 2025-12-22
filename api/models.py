# -*- coding: utf-8 -*-
"""
Sentiment models loader
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from textblob_model import TextBlobModel
import pickle


# -------------------------------------------------
# 🔹 Classe générique pour Transformers
# -------------------------------------------------
class SentimentModel:
    """Classe générique pour les modèles de sentiment basés sur Transformers."""

    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Configuration explicite des labels pour forcer les 3 classes
        self.id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        self.label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # C'est ICI que la correction est appliquée :
        # on force num_labels=3 et on ignore les erreurs de taille pour écraser la config par défaut
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=3,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True 
        )
        
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # On s'assure que logits a bien la dimension pour 3 classes
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Labels correspondant exactement à l'ordre des probabilités (0, 1, 2)
        labels = ["negative", "neutral", "positive"]

        # Sécurité : Si le modèle sort moins de 3 probabilités (ne devrait plus arriver avec le fix), on gère
        neg = float(probs[0]) if len(probs) > 0 else 0.0
        neu = float(probs[1]) if len(probs) > 1 else 0.0
        pos = float(probs[2]) if len(probs) > 2 else 0.0

        return {
            "label": labels[probs.argmax()],
            "scores": {
                "negative": neg,
                "neutral": neu,
                "positive": pos
            }
        }


# -------------------------------------------------
# 🔹 Fonction de chargement des modèles
# -------------------------------------------------
def load_model(model_key: str):
    model_key = model_key.lower()

    # --- TextBlob ---
    if model_key == "textblob":
        return TextBlobModel()
    
    # --- mBERT  ---
    if model_key == "mbert":
        model_path = "../ml_models/mbert-finetuned"
        # Si le dossier fine-tuné n'existe pas, on fallback sur le modèle de base HuggingFace
        if not os.path.exists(model_path):
            print("⚠️ Dossier mBERT local introuvable, chargement depuis HuggingFace.")
            model_path = "bert-base-multilingual-cased"
        else:
            print("➡️ Chargement du modèle mBERT local")
        
        return SentimentModel(model_path)
    
    # --- mDistilBERT  ---
    if model_key == "mdistilbert":
        # Chemin local
        model_path = "../ml_models/mdistilbert-finetuned"
        abs_path = os.path.abspath(model_path)
        
        if os.path.exists(abs_path):
             print(f"➡️ Chargement de mDistilBERT local : {abs_path}")
             return SentimentModel(abs_path)
        else:
             # Si le dossier local n'existe pas, on charge la version de base HuggingFace
             # Le __init__ de la classe forcera quand même les 3 labels.
             print("⚠️ Dossier mDistilBERT introuvable, chargement de 'distilbert-base-multilingual-cased'")
             return SentimentModel("distilbert-base-multilingual-cased")

    # --- BERTweet ---
    if model_key == "bertweet":
        finetuned_path = "../ml_models/bertweet-base-finetuned"
        base_path = "vinai/bertweet-base"

        if os.path.exists(finetuned_path):
            print("➡️ Chargement du modèle BERTweet FINE-TUNÉ")
            return SentimentModel(finetuned_path)

        print("⚠️ Modèle fine-tuné non trouvé, chargement du modèle BERTweet base")
        return SentimentModel(base_path)

    raise ValueError(f"Modèle inconnu : {model_key}")


# -------------------------------------------------
# 🔹 Instances globales pour l’API
# -------------------------------------------------
#bertweet_model = load_model("bertweet")
#textblob_model = load_model("textblob")