# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:50:23 2025

@author: faris
"""

import re
import pandas as pd

# Charger la base de données
def load_data(path="data/tweets.csv"):
    return pd.read_csv(path)

# Prétraitement simple du texte
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", '', text)
    text = re.sub(r"#\w+", '', text)
    text = re.sub(r"[^a-zA-Z0-9\s]", '', text)
    return text

# Filtrage par entreprise
def filter_by_company(df, company_name):
    return df[df['company'].str.lower() == company_name.lower()]
