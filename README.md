# Analyse des Avis Clients — Assurance NLP

> Projet NLP 2026 — Project 2 
> Analyse automatique de 34 435 avis clients dans le domaine de l'assurance

---

## Description

Ce projet applique des techniques de **traitement automatique du langage naturel (NLP)** sur un dataset réel d'avis clients d'assureurs.

##  Notebooks

### 01 — Exploration des données

### 02 — Nettoyage du texte

### 03 — Embeddings

### 04 — Modèles supervisés

### 05 — Interprétation des résultats

##  Application Streamlit
! IMPORTANT ! Le lien suivant permet d'avoir le dataset a placé dans le meme dossier que app.py
https://drive.google.com/file/d/1_pdQmzLrP4S5gf9yd8mop6qDzq23jjMr/view?usp=drive_link

L'application propose 5 fonctionnalités :

| Page | Description |
|---|---|
|  **Prédiction** | Saisir un avis → sentiment + note estimée + confiance en temps réel |
|  **Analyse Assureurs** | Classement, stats détaillées, résumé positif/négatif, recherche par mot-clé |
|  **Explication LIME** | Visualisation des mots influents — rend l'IA transparente |
|  **Recherche RAG** | Recherche sémantique via Word2Vec — avis les plus similaires à une requête |
|  **Questions & Réponses** | Questions en langage naturel sur un assureur → réponse synthétique |

### Lancer l'application

```bash
# 1. Cloner le repo
git clone https://github.com/ilanbbb/project2_nlp.git
cd project2_nlp

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer les notebooks dans l'ordre (génère les fichiers nécessaires)
# 01_exploration → 02_cleaning → 03_embeddings → 04_supervised_models

# 4. Lancer l'app
streamlit run app.py
```
### lien vidéo.
https://drive.google.com/file/d/1F_g-pFxgVUm1kZiuji0PL_OHT0VwuAYb/view?usp=drive_link


**Packages principaux :**

```
pandas, numpy, matplotlib, seaborn   # data & visualisation
nltk, gensim, pyspellchecker         # NLP
scikit-learn, joblib                 # ML classique
tensorflow, torch, transformers      # deep learning & BERT
lime, shap, pyLDAvis                 # interprétabilité
streamlit                            # application web
```

##  Auteurs

- Étudiant 1 — BOULMIER Ilan
- Étudiant 2 — COLIN DE VERDIERES Thomas

---
