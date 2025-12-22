import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split

# ---------------------------
# Chemins
# ---------------------------

DATASET_PATH = "../data/dataset.csv"                       # ton fichier unique
MODEL_PATH = "../ml_models/bertweet-base"                 # modèle non fine-tuné
SAVE_PATH = "../ml_models/bertweet-base-finetuned"        # modèle entraîné

os.makedirs("../ml_models", exist_ok=True)

# ---------------------------
# Chargement du dataset CSV
# ---------------------------

for sep in [",", ";", "\t"]:
    for enc in ["utf-8", "windows-1252", "latin-1"]:
        try:
            df = pd.read_csv(DATASET_PATH, sep=sep, encoding=enc)
            print(f"Fichier chargé avec séparateur '{sep}' et encodage '{enc}'")
            break
        except:
            df = None
    if df is not None:
        break

if df is None:
    raise ValueError("Impossible de lire dataset.csv. Vérifie séparateur ou encodage.")

# ---------------------------
# Séparation train/test
# ---------------------------

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ---------------------------
# Conversion HuggingFace dataset
# ---------------------------

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df[['review_text','sentiment_id']]),
    "test":  Dataset.from_pandas(test_df[['review_text','sentiment_id']])
})

# Renommer les colonnes pour Hugging Face
dataset = dataset.rename_column("review_text", "text")
dataset = dataset.rename_column("sentiment_id", "label")

# ---------------------------
# Chargement modèle + tokenizer
# ---------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
num_labels = df['sentiment_id'].nunique()  # déduit automatiquement le nombre de classes
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

# ---------------------------
# Tokenization
# ---------------------------

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)

# ---------------------------
# Entraînement
# ---------------------------

training_args = TrainingArguments(
    output_dir="./bertweet",
    do_eval=True,
    save_total_limit=2,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()

# ---------------------------
# Sauvegarde modèle fine-tuné
# ---------------------------

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print(f"Modèle fine-tuné sauvegardé dans : {SAVE_PATH}")
