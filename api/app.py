# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_file, render_template_string
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64  # <--- Ajouté pour encoder l'image
import utils
import models

app = Flask(__name__)

# ------------------------------------------------------------------
# 1. Chargement et Préparation des données (Au démarrage)
# ------------------------------------------------------------------
try:
    train_df = utils.load_data("../data/dataset_train.csv")
    test_df  = utils.load_data("../data/dataset_test.csv")
    tweets_df = pd.concat([train_df, test_df], ignore_index=True)
    tweets_df = tweets_df.rename(columns={"review_text": "text"})
    
    # Récupération automatique de la liste des entreprises
    if "company" in tweets_df.columns:
        COMPANIES = sorted(tweets_df['company'].unique().tolist())
    else:
        COMPANIES = ["Samsung", "Apple", "Xiaomi", "Tesla", "Netflix", "McDonalds", "Sony", "Google"] 
        
except Exception as e:
    print(f"Erreur chargement données: {e}")
    tweets_df = pd.DataFrame()
    COMPANIES = []

AVAILABLE_MODELS = ["bertweet", "textblob", "mbert", "mdistilbert"]

# ------------------------------------------------------------------
# 2. Templates HTML
# ------------------------------------------------------------------

HTML_HOME = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analyse de Sentiments AI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; }
        .card { margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; margin-bottom: 2rem; }
    </style>
</head>
<body>

    <div class="header text-center">
        <h1>🚀 Sentiment Analysis API</h1>
        <p>Analysez vos tweets avec l'Intelligence Artificielle</p>
    </div>

    <div class="container">
        <div class="row">
            
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header bg-primary text-white">📊 1. Performance</div>
                    <div class="card-body">
                        <p>Calculer la précision (accuracy) d'un modèle.</p>
                        <form action="/evaluate_model" method="get">
                            <label class="form-label">Choisir le modèle :</label>
                            <select name="model" class="form-select mb-3">
                                {% for m in models %}
                                    <option value="{{ m }}">{{ m|upper }}</option>
                                {% endfor %}
                            </select>
                            <button type="submit" class="btn btn-primary w-100">Évaluer</button>
                        </form>
                    </div>
                </div>
            </div>

            <div class="col-md-4">
                <div class="card">
                    <div class="card-header bg-success text-white">📝 2. Analyser les Tweets</div>
                    <div class="card-body">
                        <p>Voir les prédictions texte par texte.</p>
                        <form action="/analyze_form" method="get">
                            <label class="form-label">Entreprise :</label>
                            <select name="company" class="form-select mb-2">
                                {% for c in companies %}
                                    <option value="{{ c }}">{{ c }}</option>
                                {% endfor %}
                            </select>
                            
                            <label class="form-label">Modèle :</label>
                            <select name="model" class="form-select mb-3">
                                {% for m in models %}
                                    <option value="{{ m }}">{{ m|upper }}</option>
                                {% endfor %}
                            </select>
                            <button type="submit" class="btn btn-success w-100">Analyser</button>
                        </form>
                    </div>
                </div>
            </div>

            <div class="col-md-4">
                <div class="card">
                    <div class="card-header bg-danger text-white">📈 3. Visualisation</div>
                    <div class="card-body">
                        <p>Générer un graphique en barres.</p>
                        <form action="/analyze_graph_form" method="get">
                            <label class="form-label">Entreprise :</label>
                            <select name="company" class="form-select mb-2">
                                {% for c in companies %}
                                    <option value="{{ c }}">{{ c }}</option>
                                {% endfor %}
                            </select>
                            
                            <label class="form-label">Modèle :</label>
                            <select name="model" class="form-select mb-3">
                                {% for m in models %}
                                    <option value="{{ m }}">{{ m|upper }}</option>
                                {% endfor %}
                            </select>
                            <button type="submit" class="btn btn-danger w-100">Générer Graphique</button>
                        </form>
                    </div>
                </div>
            </div>

        </div>
    </div>
</body>
</html>
"""

HTML_RESULTS = """
<!DOCTYPE html>
<html>
<head>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="p-4">
    <div class="container">
        <h2>Résultats pour {{ company }} ({{ model }})</h2>
        <a href="/" class="btn btn-secondary mb-3">⬅️ Retour</a>
        <table class="table table-striped table-hover">
            <thead class="table-dark">
                <tr>
                    <th>ID</th>
                    <th>Tweet</th>
                    <th>Sentiment Prédit</th>
                </tr>
            </thead>
            <tbody>
                {% for row in results %}
                <tr>
                    <td>{{ row.id }}</td>
                    <td>{{ row.text }}</td>
                    <td>
                        {% if row.sentiment == 'positive' %}
                            <span class="badge bg-success">Positif</span>
                        {% elif row.sentiment == 'negative' %}
                            <span class="badge bg-danger">Négatif</span>
                        {% else %}
                            <span class="badge bg-secondary">Neutre</span>
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</body>
</html>
"""

HTML_GRAPH = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Visualisation - {{ company }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light text-center p-5">
    <div class="container">
        <div class="card shadow-lg">
            <div class="card-header bg-danger text-white">
                <h2>📊 Analyse Graphique : {{ company }}</h2>
                <p>Modèle : {{ model|upper }}</p>
            </div>
            <div class="card-body">
                <img src="data:image/png;base64,{{ img_data }}" class="img-fluid border rounded" alt="Graphique Sentiment">
                <br><br>
                <a href="/" class="btn btn-primary btn-lg">⬅️ Retour à l'accueil</a>
            </div>
        </div>
    </div>
</body>
</html>
"""

# ------------------------------------------------------------------
# 3. Routes de l'API
# ------------------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_HOME, companies=COMPANIES, models=AVAILABLE_MODELS)

@app.route("/evaluate_model", methods=["GET"])
def evaluate_model_form():
    model_name = request.args.get("model", "").lower()
    if model_name not in AVAILABLE_MODELS:
        return f"Modèle non reconnu.", 400

    try:
        if model_name == "textblob":
            from textblob_model import TextBlobModel
            model = TextBlobModel()
            label_col = "sentiment_id"
        else:
            model = models.load_model(model_name)
            label_col = "sentiment_label"

        accuracy = utils.evaluate_model(model, test_df, text_col="review_text", label_col=label_col)
    except Exception as e:
        return f"<h3>Erreur : {str(e)}</h3><br><a href='/'>Retour</a>", 500

    return f"""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <div class="container mt-5 text-center">
        <div class="alert alert-info">
            <h1>Résultat de l'évaluation</h1>
            <h3>Modèle : <b>{model_name.upper()}</b></h3>
            <h2 class="display-4">Précision : {accuracy:.2%}</h2>
            <br>
            <a href='/' class="btn btn-primary">⬅️ Retour à l'accueil</a>
        </div>
    </div>
    """

@app.route("/analyze_form", methods=["GET"])
def analyze_form():
    company = request.args.get("company")
    model_name = request.args.get("model", "").lower()

    filtered = utils.filter_by_company(tweets_df, company)
    if filtered.empty:
        return f"Aucun tweet trouvé.", 404

    try:
        if model_name == "textblob":
            from textblob_model import TextBlobModel
            model = TextBlobModel()
        else:
            model = models.load_model(model_name)
    except Exception as e:
        return str(e), 500

    results_data = []
    for _, row in filtered.head(50).iterrows():
        text = utils.preprocess_text(row["text"])
        
        if model_name == "textblob":
            label_num = model.predict_one(text)
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            sentiment = label_map[label_num]
        else:
            pred = model.predict(text)
            sentiment = pred["label"] if isinstance(pred, dict) else pred

        results_data.append({
            "id": row.get("id", "N/A"),
            "text": row["text"],
            "sentiment": sentiment
        })

    return render_template_string(HTML_RESULTS, company=company, model=model_name, results=results_data)

@app.route("/analyze_graph_form", methods=["GET"])
def analyze_graph_form():
    company = request.args.get("company")
    model_name = request.args.get("model", "bertweet").lower()

    filtered = utils.filter_by_company(tweets_df, company)
    if filtered.empty:
        return "Aucune donnée", 404

    try:
        if model_name == "textblob":
            from textblob_model import TextBlobModel
            model = TextBlobModel()
        else:
            model = models.load_model(model_name)
    except Exception as e:
        return str(e), 500

    sentiments = []
    for _, row in filtered.iterrows():
        text = utils.preprocess_text(row["text"])
        if model_name == "textblob":
            label_num = model.predict_one(text)
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            sentiments.append(label_map[label_num])
        else:
            pred = model.predict(text)
            sentiments.append(pred["label"] if isinstance(pred, dict) else pred)

    # 1. On compte les résultats bruts
    sentiment_counts = pd.Series(sentiments).value_counts(normalize=True) * 100
    
    # 2. CORRECTION : On force les 3 index (negative, neutral, positive)
    # Le 'fill_value=0' mettra 0% si le label n'est pas trouvé
    sentiment_counts = sentiment_counts.reindex(['positive', 'negative', 'neutral'], fill_value=0)
    
    # Création du graphique stylisé
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'negative': '#ff4d4d', 'neutral': '#d1d1d1', 'positive': '#2ecc71'}
    
    bar_colors = [colors.get(idx, 'blue') for idx in sentiment_counts.index]
    
    sentiment_counts.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='black')
    
    ax.set_title(f"Sentiments pour {company} ({model_name})", fontsize=15)
    ax.set_ylabel("Pourcentage (%)")
    ax.set_xlabel("Sentiment")
    plt.xticks(rotation=0)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    return render_template_string(HTML_GRAPH, company=company, model=model_name, img_data=data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)