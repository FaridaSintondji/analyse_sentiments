# 🚀 Analyse de Sentiments - NLP API & Dashboard

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-green)](https://flask.palletsprojects.com/)
[![Hugging Face](https://img.shields.io/badge/Models-Hugging%20Face-yellow)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/ML-PyTorch-red)](https://pytorch.org/)

Ce projet est une application complète de **Machine Learning Engineering** permettant d'analyser les sentiments (Positif, Négatif, Neutre) de tweets et avis clients concernant de grandes entreprises (Apple, Samsung, Tesla, etc.).

L'application compare les performances de plusieurs modèles de **Traitement du Langage Naturel (NLP)**, allant des méthodes basées sur des règles aux architectures Transformers fine-tunées.

## ✨ Fonctionnalités

* **🔍 Analyse Multi-Modèles :** Comparaison entre :
    * **TextBlob** (Approche Lexicale)
    * **BERTweet** (Spécialisé pour les tweets en anglais)
    * **mBERT** (Multilingual BERT)
    * **mDistilBERT** (Version légère et rapide, fine-tunée)
* **📊 Visualisation de Données :** Génération dynamique de graphiques en barres pour visualiser la répartition des sentiments par entreprise.
* **⚙️ Gestion Hybride des Modèles :** Système intelligent de chargement de modèles capable d'utiliser des modèles **fine-tunés localement** ou de basculer automatiquement sur des modèles **Hugging Face publics** en cas d'absence de fichiers locaux.
* **📈 Évaluation de Performance :** Calcul de la précision (Accuracy) sur un jeu de données de test.
* **💻 Interface Web :** Dashboard interactif développé avec **Flask** et **Bootstrap**.

## 🛠️ Stack Technique

* **Langage :** Python
* **Backend & API :** Flask, FastAPI (pour la documentation Swagger)
* **Deep Learning :** PyTorch, Transformers (Hugging Face)
* **Data Processing :** Pandas, NumPy
* **Visualisation :** Matplotlib (Backend Agg), Base64 encoding
* **Frontend :** HTML5, Jinja2, Bootstrap 5

## 📂 Structure du Projet

```bash
analyse_sentiments/
│
├── api/
│   ├── app.py             # Point d'entrée de l'application Flask
│   ├── api_swagger.py     # API alternative (FastAPI)
│   ├── models.py          # Logique de chargement des modèles (Factory pattern)
│   ├── utils.py           # Fonctions utilitaires (nettoyage de texte, metrics)
│   ├── scraping.py        # Fonction pour le webscraping
│   └── textblob_model.py  # Wrapper pour TextBlob
│
├── data/
│   ├── dataset_train.csv  # Données d'entraînement
│   └── dataset_test.csv   # Données de test
│
├── ml_models/             # Dossier pour les modèles fine-tunés 
│   ├── mdistilbert-finetuned/
│   ├── mdistilbert/
│   ├── mbert/
│   ├── mbert-finetuned/
│   ├── bertweet-base/
│   ├── bertweet-base-finetuned/
│   └── textblob/
│    
├── scripts/             # Dossier pour le téléchargements des modèles
│   ├── mdistilbert_download
│   ├── mbert_download
│   ├── bertweet_download
│   └── textblob/ 
│ 
├── training/             # Dossier contenant les fonctions d'entraînement des modèles
│   ├── mdistilbert_model
│   ├── mbert_model
│   ├── finetune_bertweet
│   └── textblob/  
├── requirements.txt       # Dépendances Python
└── README.md              # Documentation

```

## 🚀 Installation et Démarrage

### 1. Cloner le dépôt

```bash
git clone [https://github.com/FaridaSintondji/analyse_sentiments.git](https://github.com/FaridaSintondji/analyse_sentiments.git)
cd analyse_sentiments

```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt

```

### 4. Lancer l'application

```bash
python api/app.py

```

L'application sera accessible sur : `http://127.0.0.1:5000`

## 🧠 Détails des Modèles

Le cœur du projet réside dans `models.py`, qui orchestre le choix du modèle :

1. **TextBlob :** Utilisé comme *baseline*. Rapide mais moins précis sur le contexte complexe.
2. **Transformers (BERT family) :**
* Le code vérifie d'abord la présence d'un modèle entraîné localement dans le dossier `ml_models/`.
* Si le modèle local est absent, le système télécharge automatiquement une version performante depuis le **Hugging Face Hub** (ex: `lxyuan/distilbert-base-multilingual-cased-sentiments-student`).
* Gestion forcée des **3 labels** (Negative, Neutral, Positive) pour assurer la cohérence des outputs.


## 👤 Auteur

**Farida Sintondji**

* [GitHub](https://github.com/FaridaSintondji)

---

```