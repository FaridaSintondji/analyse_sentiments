# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:16:37 2025

@author: faris
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from langdetect import detect, LangDetectException
import re
import time

def clean_text(text: str) -> str:
    """Nettoie le texte pour l'analyse NLP."""
    if not text: return ""
    # Retire les sauts de ligne et espaces superflus
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def scrape_reddit_search(marque: str, produit: str, max_results: int = 50) -> pd.DataFrame:
    """
    Scrape les résultats de recherche via old.reddit.com (HTML statique et stable).
    Cible précisément 'MARQUE + PRODUIT'.
    """
    reviews_list = []
    
    # 1. Construction de l'URL de recherche intelligente
    # Ex: "Xiaomi Redmi Note 12" -> "Xiaomi+Redmi+Note+12"
    query = f"{marque} {produit}".replace(" ", "+")
    
    # On cible old.reddit.com pour éviter les classes cryptées du nouveau site
    target_url = f"https://old.reddit.com/search?q={query}&sort=relevance&t=all"
    
    print(f"--- 🚀 Recherche sur Reddit pour : '{marque} {produit}' ---")
    print(f"URL cible : {target_url}")
    
    # 2. En-têtes OBLIGATOIRES pour ne pas être bloqué (Erreur 429)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.google.com/'
    }

    try:
        response = requests.get(target_url, headers=headers, timeout=10)
        
        # Gestion anti-bot basique
        if response.status_code == 429:
            print("⚠️ Trop de requêtes (Erreur 429). Reddit nous bloque temporairement. Réessayez dans 2 min.")
            return pd.DataFrame()
            
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 3. Le "Décryptage" : Sélecteurs stables de old.reddit
        # Chaque résultat est dans une div class="search-result-link"
        search_results = soup.find_all('div', class_='search-result-link')
        
        if not search_results:
            print("❌ Aucun résultat trouvé. Vérifie l'orthographe ou essaie un produit plus connu.")
            return pd.DataFrame()
            
        count = 0
        for result in search_results:
            if count >= max_results: break
            
            # Extraction du TITRE (C'est souvent là qu'est le sentiment principal sur Reddit)
            title_tag = result.find('a', class_='search-title')
            title_text = title_tag.get_text(strip=True) if title_tag else ""
            
            # Pour le POC, on utilise le titre comme commentaire. 
            # (C'est suffisant car les titres Reddit sont très descriptifs : "My Redmi Note 12 battery is dead!")
            clean_review = clean_text(title_text)
            
            if clean_review:
                # Détection de la langue (Important pour le multilingue)
                try:
                    lang = detect(clean_review)
                except:
                    lang = 'unknown'
                
                # Formatage IDENTIQUE à ton fichier dataset__test.csv
                record = {
                    'id': f"REDDIT_{count}", 
                    'brand': marque,
                    'product': produit,
                    'source': 'reddit',         
                    'language': lang,
                    'review_text': clean_review, 
                    'sentiment_label': None, # Vide : sera rempli par l'API de Farida
                    'sentiment_id': None,    # Vide
                    'sarcastic': None        # Vide
                }
                reviews_list.append(record)
                count += 1
        
        print(f"✅ SUCCÈS : {count} discussions récupérées sur Reddit.")

    except Exception as e:
        print(f"❌ Erreur technique : {e}")

    # Création du DataFrame final
    columns = ['id', 'brand', 'product', 'source', 'language', 'review_text', 'sentiment_label', 'sentiment_id', 'sarcastic']
    df = pd.DataFrame(reviews_list, columns=columns)
    
    return df

# --- ZONE DE TEST (À exécuter pour valider) ---
# Imagine que l'utilisateur tape ça dans l'interface :
# TEST_MARQUE = "Apple"
# TEST_PRODUIT = "iPhone 15"

TEST_MARQUE = "Samsung"
TEST_PRODUIT = "galaxy s24"

# On lance le scraping
df_resultat = scrape_reddit_search(TEST_MARQUE, TEST_PRODUIT)

if not df_resultat.empty:
    print("\n--- Aperçu des données (Prêtes pour Farida) ---")
    print(df_resultat[['product', 'review_text', 'language']].head())
    
    # On sauvegarde le fichier pour prouver que ça marche
    nom_fichier = "donnees_reddit_live.csv"
    df_resultat.to_csv(nom_fichier, index=False, encoding='utf-8')
    print(f"\n📁 Fichier '{nom_fichier}' généré avec succès !")