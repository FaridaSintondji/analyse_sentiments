# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:48:56 2025

@author: faris
"""

import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import utils
import models
import matplotlib.pyplot as plt
import io
from fastapi.responses import StreamingResponse

app = FastAPI(title="Sentiment Analysis API")

# -------------------------------------------------
# 📌 Charger les datasets train + test
# -------------------------------------------------
train_df = utils.load_data("../data/dataset_train.csv")
test_df  = utils.load_data("../data/dataset_test.csv")
tweets_df = pd.concat([train_df, test_df], ignore_index=True)
tweets_df = tweets_df.rename(columns={"review_text": "text"})

# -------------------------------------------------
# 📌 Modèles disponibles pour l’utilisateur
# -------------------------------------------------
AVAILABLE_MODELS = ["bertweet", "textblob", "mbert", "mdistilbert"]

# -------------------------------------------------
# 📌 Schéma d’entrée API
# -------------------------------------------------
class RequestBody(BaseModel):
    company: str
    model: str  # "bertweet" ou "textblob"

# -------------------------------------------------
# 📌 Endpoint d’évaluation
# -------------------------------------------------
@app.get("/evaluate_model/{model_name}")
def evaluate_model_endpoint(model_name: str):
    model_name = model_name.lower()
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Modèle non reconnu. Choisir parmi {AVAILABLE_MODELS}"
        )

    try:
        # 1️⃣ Charger le modèle
        if model_name == "textblob":
            from textblob_model import TextBlobModel
            model = TextBlobModel()
            label_col = "sentiment_id"         # Pour textblob
        else:
            model = models.load_model(model_name)
            label_col = "sentiment_label"      # Pour les autres modèles

        # 2️⃣ Evaluation sur test_df avec colonne adaptée
        accuracy = utils.evaluate_model(
            model,
            test_df,
            text_col="review_text",
            label_col=label_col
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur évaluation : {str(e)}")

    return {"model": model_name, "accuracy": f"{accuracy:.2%}"}

# -------------------------------------------------
# 📌 Endpoint principal
# -------------------------------------------------
@app.post("/analyze")
def analyze_sentiments(request: RequestBody):
    company = request.company
    model_name = request.model.lower()

    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Modèle non reconnu. Choisir parmi {AVAILABLE_MODELS}"
        )

    filtered = utils.filter_by_company(tweets_df, company)
    if filtered.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Aucun tweet trouvé pour l'entreprise '{company}'."
        )

    # Chargement du modèle
    try:
        if model_name == "textblob":
            from textblob_model import TextBlobModel
            model = TextBlobModel()
        else:
            model = models.load_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur modèle : {str(e)}")

    results = []
    for _, row in filtered.iterrows():
        text = utils.preprocess_text(row["text"])
        try:
            if model_name == "textblob":
                label_num = model.predict_one(text)  # 0/1/2
                label_map = {0: "negative", 1: "neutral", 2: "positive"}
                sentiment = {"label": label_map[label_num]}
            else:
                sentiment = model.predict(text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur prédiction : {str(e)}")

        results.append({
            "id": row["id"],
            "text": row["text"],
            "sentiment": sentiment
        })

    return {
        "company": company,
        "model": model_name,
        "results": results
    }

# -------------------------------------------------
# 📌 Endpoint graphique
# -------------------------------------------------
@app.get("/analyze_graph/{company}")
def analyze_graph(company: str, model_name: str = "bertweet"):
    filtered = utils.filter_by_company(tweets_df, company)
    if filtered.empty:
        raise HTTPException(status_code=404, detail=f"Aucun tweet trouvé pour l'entreprise '{company}'.")

    # Chargement du modèle
    try:
        if model_name.lower() == "textblob":
            from textblob_model import TextBlobModel
            model = TextBlobModel()
        else:
            model = models.load_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur modèle : {str(e)}")

    sentiments = []
    for _, row in filtered.iterrows():
        text = utils.preprocess_text(row["text"])
        if model_name.lower() == "textblob":
            label_num = model.predict_one(text)
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            sentiments.append(label_map[label_num])
        else:
            result = model.predict(text)
            sentiments.append(result["label"])

    sentiment_counts = pd.Series(sentiments).value_counts(normalize=True) * 100

    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', ax=ax, color=['red', 'gray', 'green'])
    ax.set_title(f"Répartition des sentiments pour {company}")
    ax.set_ylabel("Pourcentage (%)")
    ax.set_xlabel("Sentiment")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")