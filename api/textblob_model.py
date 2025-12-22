"""

Ce fichier définit un petit modèle de sentiment basé sur TextBlob.
Le but est de fournir un "modèle entraîné" simple, que l'API pourra charger.

Fonctionnement :
- predict_one(text)      → renvoie 0 (négatif), 1 (neutre), 2 (positif)
- predict(list_texts)    → applique la prédiction sur plusieurs textes
- score_brand_product()  → calcule une note /10 pour une marque + produit

Le fichier peut aussi générer un fichier .pkl (modèle sauvegardé).
Il suffit d'exécuter ce fichier pour créer : textblob_model.pkl
"""

from textblob import TextBlob
import pickle
import pandas as pd
import os


class TextBlobModel:
    """Petit modèle basé sur TextBlob (lexique)."""

    def __init__(self, pos_thresh=0.1, neg_thresh=-0.1):
        # Seuils utilisés pour décider du sentiment
        self.pos_thresh = pos_thresh   # au-dessus = positif
        self.neg_thresh = neg_thresh   # en-dessous = négatif

    def predict_one(self, text):
        """Retourne un label 0/1/2 pour un texte donné."""
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity  # valeur entre -1 et 1

        # Décision simple basée sur la polarité
        if polarity > self.pos_thresh:
            return 2   # positif
        elif polarity < self.neg_thresh:
            return 0   # négatif
        else:
            return 1   # neutre

    def predict(self, texts):
        """Applique predict_one à une liste de textes."""
        return [self.predict_one(t) for t in texts]

    def score_brand_product(self, df, brand, product, text_col="review_text"):
        """
        Calcule une note /10 pour une marque + produit à partir d'un DataFrame.
        - df : dataframe contenant les avis
        """
        # On filtre les avis concernés
        subset = df[(df["brand"] == brand) & (df["product"] == product)]

        if subset.empty:
            return {
                "brand": brand,
                "product": product,
                "score10": None,
                "n_total": 0,
                "n_pos": 0,
                "n_neu": 0,
                "n_neg": 0
            }

        y_pred = self.predict(subset[text_col])

        n = len(y_pred)
        n_pos = sum(1 for y in y_pred if y == 2)
        n_neu = sum(1 for y in y_pred if y == 1)
        n_neg = sum(1 for y in y_pred if y == 0)

        # Calcul de la note /10 (simple et lisible)
        score10 = 10.0 * (n_pos + 0.5 * n_neu) / n

        return {
            "brand": brand,
            "product": product,
            "score10": round(score10, 2),
            "n_total": n,
            "n_pos": n_pos,
            "n_neu": n_neu,
            "n_neg": n_neg
        }




# génèration du fichier textblob_model.pkl
if __name__ == "__main__":

    # Instanciation du modèle
    model = TextBlobModel(pos_thresh=0.1, neg_thresh=-0.1)

    # Chemin où le modèle sera sauvegardé
    save_path = "../ml_models/textblob/textblob_model.pkl"

    # Création des dossiers si ils n'existent pas
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Sauvegarde du modèle en .pkl
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Modèle sauvegardé dans : {save_path}")
