# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 16:18:18 2025

@author: faris
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
import shutil
from google.colab import files

# !pip install transformers accelerate datasets scikit-learn (à lancer dans une cellule avant)

# --- CONFIGURATION CIBLE ---
# C'est ici que tout change : on vise DistilBERT
MODEL_NAME = "distilbert-base-multilingual-cased"
OUTPUT_DIR = "./mdistilbert_finetuned"

# --- 1. Chargement des données ---
try:
    df_train = pd.read_csv("dataset_train.csv")
    df_test = pd.read_csv("dataset_test.csv")
except FileNotFoundError:
    print("ERREUR : Uploadez dataset_train.csv et dataset_test.csv !")
    raise

# Renommage colonnes
if "review_text" in df_train.columns:
    df_train = df_train.rename(columns={"review_text": "text"})
    df_test = df_test.rename(columns={"review_text": "text"})

# Gestion Labels (si sentiment_label est en texte)
if "sentiment_label" in df_train.columns and df_train['sentiment_label'].dtype == 'O':
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df_train['label'] = df_train['sentiment_label'].map(label_map)
    df_test['label'] = df_test['sentiment_label'].map(label_map)
elif "sentiment_id" in df_train.columns:
    df_train = df_train.rename(columns={"sentiment_id": "label"})
    df_test = df_test.rename(columns={"sentiment_id": "label"})
else:
    # Fallback si déjà nommée 'label'
    pass

# --- 2. Dataset Class ---
class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row["text"])
        try:
            label = int(row["label"])
        except:
            label = 1 # Neutre par défaut si erreur

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

# --- 3. Initialisation ---
print(f"Téléchargement du modèle : {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# DistilBERT s'initialise exactement comme BERT
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label={0: "negative", 1: "neutral", 2: "positive"},
    label2id={"negative": 0, "neutral": 1, "positive": 2}
)

train_dataset = SentimentDataset(df_train, tokenizer)
eval_dataset = SentimentDataset(df_test, tokenizer)

# --- 4. Entraînement ---
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch", # Nouveau nom du paramètre
    save_strategy="no",
    fp16=True,             # GPU Boost
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

print("🔥 Démarrage Fine-Tuning mDistilBERT...")
trainer.train()

# --- 5. Sauvegarde & Téléchargement ---
print(f"Sauvegarde...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Zip pour le téléchargement
shutil.make_archive('mon_modele_mdistilbert', 'zip', OUTPUT_DIR)
files.download('mon_modele_mdistilbert.zip')