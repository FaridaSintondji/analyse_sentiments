
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

#BERTweet
# Token Hugging Face (facultatif si modèle public)
token = os.getenv('HF_TOKEN')

# Nom du modèle BERTweet sur Hugging Face
model_name = "vinai/bertweet-base"

# Chemin relatif vers le dossier de sauvegarde (au même niveau que models.py)
save_path = "ml_models/bertweet-base"

# Créer le dossier si nécessaire
os.makedirs(save_path, exist_ok=True)

# Télécharger le tokenizer et le modèle
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token)

# Sauvegarder localement
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"BERTweet téléchargé et sauvegardé dans : {save_path}")


