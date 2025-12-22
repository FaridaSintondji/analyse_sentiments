# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 16:17:44 2025

@author: faris
"""
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_and_configure_mdistilbert():
    # 1. Configuration
    model_name = "distilbert-base-multilingual-cased"
    save_directory = "../ml_models/mdistilbert"

    print(f"Début du téléchargement de : {model_name}")
    print(f"Destination : {os.path.abspath(save_directory)}")

    # 2. Création du dossier
    os.makedirs(save_directory, exist_ok=True)

    # 3. Téléchargement du Tokenizer
    print("⏳ Téléchargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)

    # 4. Configuration du Modèle (3 classes)
    print("⏳ Configuration du modèle pour 3 labels (Neg, Neu, Pos)...")
    
    # On définit explicitement les labels pour éviter les bugs d'index
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label2id = {"negative": 0, "neutral": 1, "positive": 2}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    # 5. Sauvegarde
    model.save_pretrained(save_directory)

    print("-" * 30)
    print(f"SUCCÈS ! mDistilBERT configuré et sauvegardé.")
    print(f"📍 Chemin : {os.path.abspath(save_directory)}")

if __name__ == "__main__":
    download_and_configure_mdistilbert()