# -*- coding: utf-8 -*-
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_and_configure_mbert():
    # 1. Configuration des chemins
    # On définit le nom officiel du modèle chez Hugging Face
    model_name = "bert-base-multilingual-cased"
    
    save_directory = "../ml_models/mbert"

    print(f"Début du téléchargement de : {model_name}")
    print(f"Destination : {os.path.abspath(save_directory)}")

    # 2. Création du dossier si nécessaire
    os.makedirs(save_directory, exist_ok=True)

    # 3. Téléchargement du Tokenizer (le "découpeur" de mots)
    print("Téléchargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)

    # 4. Téléchargement et Configuration du Modèle
    print("Téléchargement et configuration du modèle (3 classes)...")
    
    # On définit explicitement quels sont nos 3 labels.
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label2id = {"negative": 0, "neutral": 1, "positive": 2}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,       # On force 3 sorties
        id2label=id2label,  # On sauvegarde les noms des labels dans le modèle
        label2id=label2id
    )

    # 5. Sauvegarde définitive
    model.save_pretrained(save_directory)

    print("-" * 30)
    print(f"Sauvegardé dans : {os.path.abspath(save_directory)}")

if __name__ == "__main__":
    download_and_configure_mbert()