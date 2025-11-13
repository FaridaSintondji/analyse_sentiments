# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:48:56 2025

@author: faris
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import utils
import models

app = FastAPI(title="Sentiment Analysis API")

# Charger la base de données
tweets_df = utils.load_data()

# Schéma d'entrée
class RequestBody(BaseModel):
    company: str
    model: str  # "vader", "bertweet", "mbert", "xlm"

# Endpoint principal
@app.post("/analyze")
def analyze_sentiments(request: RequestBody):
    company = request.company
    model_name = request.model.lower()
    
    # Filtrer les tweets
    filtered = utils.filter_by_company(tweets_df, company)
    if filtered.empty:
        raise HTTPException(status_code=404, detail="Aucun tweet trouvé pour cette entreprise.")
    
    results = []
    for _, row in filtered.iterrows():
        text = utils.preprocess_text(row['text'])
        
        if model_name == "vader":
            sentiment = models.analyze_vader(text)
        elif model_name == "bertweet":
            sentiment = models.bertweet_model.predict(text)
        elif model_name == "mbert":
            sentiment = models.mbert_model.predict(text)
        elif model_name == "xlm":
            sentiment = models.xlm_model.predict(text)
        else:
            raise HTTPException(status_code=400, detail="Modèle non reconnu.")
        
        results.append({
            "id": row['id'],
            "text": row['text'],
            "sentiment": sentiment
        })
    
    return {"company": company, "model": model_name, "results": results}

