import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
import os

# ---------------- 1. Configuration des Chemins ----------------
# Le dossier où se trouve le modèle vide (téléchargé)
MODEL_PATH = "../ml_models/mbert"
# Le dossier où on sauvegarde le modèle entraîné (on écrase le précédent)
SAVE_PATH = "../ml_models/mbert"

print(f"Chargement du modèle depuis : {os.path.abspath(MODEL_PATH)}")

# ---------------- 2. Préparation des Datasets ----------------
# Dataset d'entraînement
df_train = pd.read_csv("../data/dataset_train.csv")
# On renomme pour que ce soit clair, et on s'assure d'avoir les bonnes colonnes
df_train = df_train.rename(columns={"review_text": "text"}) 
# Si ta colonne s'appelle 'sentiment_label' (str) ou 'sentiment_id' (int), adapte ici :
if "sentiment_id" in df_train.columns:
    df_train = df_train.rename(columns={"sentiment_id": "label"})

# Dataset de test
df_test = pd.read_csv("../data/dataset_test.csv")
df_test = df_test.rename(columns={"review_text": "text"})
if "sentiment_id" in df_test.columns:
    df_test = df_test.rename(columns={"sentiment_id": "label"})

# Classe Dataset (Celle que tu avais était très bien !)
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
        
        # Gestion sécurisée du label
        label = int(row["label"])
            
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

# ---------------- 3. Chargement Tokenizer & Modèle ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("Tokenizer chargé !")

# Création des objets Dataset (C'est ICI que tu avais une erreur)
train_dataset = SentimentDataset(df_train, tokenizer)
eval_dataset = SentimentDataset(df_test, tokenizer)

# Chargement du modèle
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=3,
    ignore_mismatched_sizes=True 
    # id2label et label2id sont déjà dans le config.json grâce au script de download
)

# ---------------- 4. Entraînement ----------------
training_args = TrainingArguments(
    output_dir="./results",          # Dossier temporaire pour les checkpoints
    num_train_epochs=3,              # 3 passages complets
    per_device_train_batch_size=8,   # Réduit à 8 pour éviter de saturer la mémoire
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_steps=10,
    evaluation_strategy="epoch",     # Evaluer à la fin de chaque époque
    save_strategy="no",              # On sauvegarde manuellement à la fin
    learning_rate=2e-5,              # Vitesse d'apprentissage standard pour BERT
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, # <--- CORRECTION : on passe l'objet Dataset, pas le DataFrame
    eval_dataset=eval_dataset,   # <--- CORRECTION : idem
    compute_metrics=compute_metrics
)

print("🚀 Démarrage de l'entraînement...")
trainer.train()

# ---------------- 5. Sauvegarde ----------------
print(f"💾 Sauvegarde du modèle entraîné dans : {os.path.abspath(SAVE_PATH)}")
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print("✅ Terminé ! Le modèle est prêt pour l'API.")