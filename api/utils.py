import re
import pandas as pd
from sklearn.metrics import accuracy_score

# ---------------------------
# Chargement des données CSV
# ---------------------------
def load_data(path: str) -> pd.DataFrame:
    """
    Charge un fichier CSV en essayant plusieurs séparateurs et encodages.
    """
    for sep in [",", ";", "\t"]:
        for enc in ["utf-8", "windows-1252", "latin-1"]:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc)
                print(f"Fichier '{path}' chargé avec séparateur '{sep}' et encodage '{enc}'")
                return df
            except Exception:
                continue
    raise ValueError(f"Impossible de lire {path}. Vérifie séparateur ou encodage.")

# ---------------------------
# Prétraitement du texte
# ---------------------------
def preprocess_text(text: str) -> str:
    """
    Nettoie le texte : met en minuscules, supprime URL, mentions, hashtags et caractères non alphanumériques.
    Garde les caractères accentués.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", '', text)
    text = re.sub(r"#\w+", '', text)
    text = re.sub(r"[^a-zA-Z0-9\sÀ-ÿ]", '', text)  # garder accents
    text = re.sub(r"\s+", ' ', text).strip()       # enlever les espaces multiples
    return text

# ---------------------------
# Filtrage par marque/entreprise
# ---------------------------
def filter_by_company(df: pd.DataFrame, company_name: str) -> pd.DataFrame:
    """
    Filtre le dataframe pour ne garder que les lignes correspondant à la marque spécifiée.
    """
    if 'brand' not in df.columns:
        raise ValueError("La colonne 'brand' est manquante dans le dataset.")
    return df[df['brand'].str.lower() == company_name.lower()]

# ---------------------------
# Sélection du modèle
# ---------------------------
def select_model(models_dict: dict, model_name: str):
    """
    Renvoie le modèle choisi parmi un dictionnaire de modèles disponibles.
    """
    model_name = model_name.lower()
    if model_name not in models_dict:
        raise ValueError(f"Modèle '{model_name}' non disponible. Choisis parmi {list(models_dict.keys())}.")
    return models_dict[model_name]

def evaluate_model(model, df: pd.DataFrame, text_col: str = "review_text", label_col: str = "sentiment") -> float:

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Colonnes '{text_col}' ou '{label_col}' manquantes dans le dataset.")

    y_true = df[label_col].tolist()
    y_pred = []

    for _, row in df.iterrows():
        text = preprocess_text(row[text_col])
        result = model.predict(text)

        # --- Correction unifiée ---
        if isinstance(result, list):
            result = {"label": result[0]}
        elif isinstance(result, str):
            result = {"label": result}

        pred_label = result.get("label")
        if pred_label is None:
            raise ValueError("Le modèle n'a pas retourné de label")

        y_pred.append(pred_label)

    return accuracy_score(y_true, y_pred)
