"""
Application Streamlit — Analyse des avis clients d'assurance
Projet NLP 2026 — Project 2
"""

import streamlit as st

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Analyse Avis Assurance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Navigation ────────────────────────────────────────────────────────────────
st.sidebar.title("🛡️ Avis Assurance NLP")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Accueil",
        "🔮 Prédiction",
        "📊 Analyse Assureurs",
        "💡 Explication (LIME)",
        "🔍 Recherche (RAG)",
        "❓ Questions (QA)",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Projet NLP 2026\nProject 2 — Supervised Learning")

# ── Imports communs ───────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import re

# ── Chargement des données et modèles (mis en cache) ─────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("avis_train_clean.csv")
    df = df.dropna(subset=["avis_clean", "note"]).reset_index(drop=True)
    return df

@st.cache_resource
def load_models():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    clf   = joblib.load("lr_classifier.pkl")
    le    = joblib.load("label_encoder.pkl")
    return tfidf, clf, le

def clean_text(text):
    """Nettoyage rapide d'un texte utilisateur."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s\-àâäéèêëîïôùûüçœæ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil":
    st.title("🛡️ Analyse des Avis Clients — Assurance")
    st.markdown("### Bienvenue sur l'application NLP d'analyse des avis clients")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    try:
        df = load_data()
        col1.metric("📝 Avis analysés", f"{len(df):,}")
        col2.metric("🏢 Assureurs", f"{df['assureur'].nunique()}")
        col3.metric("⭐ Note moyenne", f"{df['note'].mean():.2f}/5")
        col4.metric("😊 % Positifs", f"{(df['sentiment_3']=='positif').mean()*100:.1f}%")
    except:
        col1.metric("📝 Avis analysés", "24 104")
        col2.metric("🏢 Assureurs", "56")
        col3.metric("⭐ Note moyenne", "3.2/5")
        col4.metric("😊 % Positifs", "~42%")

    st.markdown("---")
    st.markdown("## 🗺️ Fonctionnalités disponibles")

    col_a, col_b = st.columns(2)
    with col_a:
        st.success("**🔮 Prédiction** — Soumettez un avis et obtenez une prédiction de sentiment et de note")
        st.info("**📊 Analyse Assureurs** — Statistiques, résumés et comparaisons par assureur")
        st.warning("**💡 Explication (LIME)** — Comprendre pourquoi le modèle prédit ce résultat")
    with col_b:
        st.error("**🔍 Recherche (RAG)** — Retrouvez les avis les plus similaires à votre requête")
        st.success("**❓ Questions (QA)** — Posez des questions sur les avis d'un assureur")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prédiction":
    st.title("🔮 Prédiction de Sentiment")
    st.markdown("Entrez un avis client et le modèle prédit son sentiment et sa note estimée.")
    st.markdown("---")

    try:
        tfidf, clf, le = load_models()
        models_loaded = True
    except Exception as e:
        st.warning(f"⚠️ Modèles non trouvés. Lancez d'abord le notebook 04. ({e})")
        models_loaded = False

    # Zone de saisie
    user_input = st.text_area(
        "✍️ Entrez votre avis client ici :",
        placeholder="Ex : Le service client était très réactif et le remboursement a été rapide...",
        height=150
    )

    col1, col2 = st.columns([1, 3])
    predict_btn = col1.button("🚀 Analyser", type="primary", use_container_width=True)
    col2.markdown("")

    if predict_btn and user_input.strip():
        if not models_loaded:
            st.error("Modèles non disponibles.")
        else:
            with st.spinner("Analyse en cours..."):
                # Nettoyage et prédiction
                text_clean = clean_text(user_input)
                X = tfidf.transform([text_clean])
                pred_label = clf.predict(X)[0]
                pred_proba = clf.predict_proba(X)[0]
                sentiment  = le.inverse_transform([pred_label])[0]

                # Estimation de la note
                proba_dict = {le.classes_[i]: pred_proba[i] for i in range(len(le.classes_))}
                note_estimee = (
                    proba_dict.get('positif', 0) * 4.5 +
                    proba_dict.get('neutre', 0)  * 3.0 +
                    proba_dict.get('negatif', 0) * 1.5
                )

            st.markdown("---")
            st.markdown("## 📊 Résultats")

            # Affichage du sentiment
            col1, col2, col3 = st.columns(3)
            emoji_map = {'positif': '😊', 'neutre': '😐', 'negatif': '😞'}
            color_map = {'positif': 'green', 'neutre': 'orange', 'negatif': 'red'}

            col1.markdown(f"### Sentiment prédit")
            col1.markdown(
                f"<h2 style='color:{color_map[sentiment]}'>"
                f"{emoji_map[sentiment]} {sentiment.upper()}</h2>",
                unsafe_allow_html=True
            )

            col2.markdown("### Note estimée")
            stars = "⭐" * round(note_estimee)
            col2.markdown(f"<h2>{stars} {note_estimee:.1f}/5</h2>", unsafe_allow_html=True)

            col3.markdown("### Confiance")
            max_proba = max(pred_proba)
            col3.markdown(f"<h2>{max_proba*100:.1f}%</h2>", unsafe_allow_html=True)

            st.markdown("---")

            # Barres de probabilité
            st.markdown("### Probabilités par classe")
            for classe in le.classes_:
                proba = proba_dict.get(classe, 0)
                color = color_map.get(classe, 'blue')
                st.markdown(f"**{emoji_map.get(classe,'')} {classe.capitalize()}**")
                st.progress(float(proba))
                st.caption(f"{proba*100:.1f}%")

            # Mots clés détectés
            st.markdown("---")
            st.markdown("### 🔑 Mots clés analysés")
            mots = [m for m in text_clean.split() if len(m) > 3][:20]
            st.markdown(" ".join([f"`{m}`" for m in mots]))

    elif predict_btn and not user_input.strip():
        st.warning("⚠️ Veuillez entrer un avis avant de cliquer sur Analyser.")

    # Exemples prédéfinis
    st.markdown("---")
    st.markdown("### 💡 Exemples à tester")
    exemples = {
        "Avis positif":  "Excellent service client, remboursement rapide et conseiller très sympathique. Je recommande vivement !",
        "Avis négatif":  "Arnaque totale, impossible de résilier mon contrat. Service client inexistant, j'attends depuis 3 mois.",
        "Avis ambigu":   "Le prix est correct mais les délais de remboursement sont un peu longs. Pas de problème majeur."
    }
    for label, exemple in exemples.items():
        if st.button(f"📋 {label}", use_container_width=False):
            st.text_area("Exemple sélectionné :", value=exemple, height=80, key=f"ex_{label}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYSE ASSUREURS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analyse Assureurs":
    st.title("📊 Analyse des Assureurs")
    st.markdown("---")

    try:
        df = load_data()
    except Exception as e:
        st.error(f"Données non trouvées : {e}")
        st.stop()

    # Filtres
    col1, col2, col3 = st.columns(3)
    assureurs_dispo = sorted(df['assureur'].unique())
    assureur_sel = col1.selectbox("🏢 Sélectionner un assureur", ["Tous"] + assureurs_dispo)
    note_min, note_max = col2.slider("⭐ Plage de notes", 1, 5, (1, 5))
    min_avis = col3.number_input("📝 Nb minimum d'avis", min_value=1, max_value=500, value=50)

    # Filtrage
    df_filtered = df[df['note'].between(note_min, note_max)]
    if assureur_sel != "Tous":
        df_filtered = df_filtered[df_filtered['assureur'] == assureur_sel]

    st.markdown(f"**{len(df_filtered):,} avis filtrés**")
    st.markdown("---")

    if assureur_sel == "Tous":
        # Vue globale
        st.markdown("## 🏆 Classement des assureurs")

        stats = (
            df_filtered.groupby('assureur')
            .agg(
                note_moyenne=('note', 'mean'),
                nb_avis=('avis', 'count'),
                pct_positif=('sentiment_3', lambda x: (x=='positif').mean()*100),
                pct_negatif=('sentiment_3', lambda x: (x=='negatif').mean()*100)
            )
            .query(f'nb_avis >= {min_avis}')
            .sort_values('note_moyenne', ascending=False)
            .round(2)
        )

        # Tableau interactif
        st.dataframe(
            stats.style.background_gradient(subset=['note_moyenne'], cmap='RdYlGn')
                       .background_gradient(subset=['pct_positif'], cmap='Greens')
                       .background_gradient(subset=['pct_negatif'], cmap='Reds'),
            use_container_width=True
        )

        # Top 10 graphique
        top10 = stats.head(10)
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ['#2ecc71' if n >= 4 else '#f39c12' if n >= 3 else '#e74c3c'
                  for n in top10['note_moyenne']]
        bars = ax.barh(top10.index[::-1], top10['note_moyenne'][::-1], color=colors[::-1])
        ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Top 10 assureurs — Note moyenne', fontweight='bold')
        ax.set_xlabel('Note moyenne /5')
        ax.set_xlim(1, 5.5)
        for bar, val in zip(bars, top10['note_moyenne'][::-1]):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center')
        st.pyplot(fig)
        plt.close()

    else:
        # Vue détaillée d'un assureur
        st.markdown(f"## 🏢 {assureur_sel}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📝 Nb d'avis", f"{len(df_filtered):,}")
        col2.metric("⭐ Note moyenne", f"{df_filtered['note'].mean():.2f}/5")
        col3.metric("😊 % Positifs", f"{(df_filtered['sentiment_3']=='positif').mean()*100:.1f}%")
        col4.metric("😞 % Négatifs", f"{(df_filtered['sentiment_3']=='negatif').mean()*100:.1f}%")

        st.markdown("---")

        col_a, col_b = st.columns(2)

        # Distribution des notes
        with col_a:
            st.markdown("### Distribution des notes")
            fig, ax = plt.subplots(figsize=(6, 4))
            note_counts = df_filtered['note'].value_counts().sort_index()
            ax.bar(note_counts.index, note_counts.values,
                   color=sns.color_palette('RdYlGn', 5))
            ax.set_xlabel('Note')
            ax.set_ylabel('Nombre d\'avis')
            st.pyplot(fig)
            plt.close()

        # Sentiment pie
        with col_b:
            st.markdown("### Répartition du sentiment")
            fig, ax = plt.subplots(figsize=(6, 4))
            sent_counts = df_filtered['sentiment_3'].value_counts()
            colors_pie = {'positif': '#2ecc71', 'neutre': '#f39c12', 'negatif': '#e74c3c'}
            ax.pie(sent_counts.values,
                   labels=sent_counts.index,
                   autopct='%1.1f%%',
                   colors=[colors_pie.get(s, 'gray') for s in sent_counts.index])
            st.pyplot(fig)
            plt.close()

        # Résumé des avis
        st.markdown("---")
        st.markdown("### 📋 Résumé automatique")
        positifs = df_filtered[df_filtered['sentiment_3'] == 'positif']['avis'].head(3).tolist()
        negatifs = df_filtered[df_filtered['sentiment_3'] == 'negatif']['avis'].head(3).tolist()

        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.success("**Points positifs fréquents**")
            for avis in positifs:
                st.markdown(f"- *\"{avis[:120]}...\"*")
        with col_neg:
            st.error("**Points négatifs fréquents**")
            for avis in negatifs:
                st.markdown(f"- *\"{avis[:120]}...\"*")

        # Recherche dans les avis
        st.markdown("---")
        st.markdown("### 🔎 Rechercher dans les avis")
        keyword = st.text_input("Mot-clé :", placeholder="Ex: remboursement, résiliation, prix...")
        if keyword:
            mask = df_filtered['avis'].str.contains(keyword, case=False, na=False)
            results = df_filtered[mask][['note', 'avis', 'date_publication']].head(10)
            st.markdown(f"**{mask.sum()} avis contiennent « {keyword} »**")
            st.dataframe(results, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EXPLICATION (LIME)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Explication (LIME)":
    st.title("💡 Explication des prédictions — LIME")
    st.markdown("LIME explique **pourquoi** le modèle a prédit tel résultat en identifiant les mots qui ont le plus influencé la décision.")
    st.markdown("---")

    try:
        tfidf, clf, le = load_models()
        from lime.lime_text import LimeTextExplainer
        lime_ok = True
    except Exception as e:
        st.warning(f"⚠️ LIME ou modèles non disponibles : {e}\n\nInstallez : `pip install lime`")
        lime_ok = False

    user_input = st.text_area(
        "✍️ Entrez un avis à expliquer :",
        placeholder="Ex : Service client désastreux, j'attends mon remboursement depuis 2 mois...",
        height=120
    )

    num_features = st.slider("Nombre de mots à afficher", 5, 20, 10)

    if st.button("💡 Expliquer", type="primary") and user_input.strip():
        if not lime_ok:
            st.error("LIME non disponible.")
        else:
            with st.spinner("Calcul de l'explication LIME (quelques secondes)..."):
                text_clean = clean_text(user_input)

                def predict_proba(texts):
                    X = tfidf.transform(texts)
                    return clf.predict_proba(X)

                explainer = LimeTextExplainer(class_names=le.classes_)
                exp = explainer.explain_instance(
                    text_clean, predict_proba,
                    num_features=num_features,
                    num_samples=300
                )

                # Prédiction
                X = tfidf.transform([text_clean])
                pred = le.inverse_transform(clf.predict(X))[0]
                proba = clf.predict_proba(X)[0]

            st.markdown(f"**Sentiment prédit : `{pred.upper()}`** "
                        f"(confiance : {max(proba)*100:.1f}%)")
            st.markdown("---")

            # Explication sous forme de barres
            st.markdown("### 📊 Mots influents")
            explanation = exp.as_list()

            words_pos = [(w, s) for w, s in explanation if s > 0]
            words_neg = [(w, s) for w, s in explanation if s < 0]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Mots qui poussent vers positif**")
                for word, score in sorted(words_pos, key=lambda x: -x[1])[:8]:
                    st.markdown(f"- `{word}` → **+{score:.3f}**")

            with col2:
                st.markdown("**❌ Mots qui poussent vers négatif**")
                for word, score in sorted(words_neg, key=lambda x: x[1])[:8]:
                    st.markdown(f"- `{word}` → **{score:.3f}**")

            # Graphique
            fig, ax = plt.subplots(figsize=(10, 5))
            words_all = [w for w, _ in explanation]
            scores    = [s for _, s in explanation]
            colors_bar = ['#2ecc71' if s > 0 else '#e74c3c' for s in scores]
            ax.barh(words_all[::-1], scores[::-1], color=colors_bar[::-1])
            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.set_title('Importance des mots — LIME', fontweight='bold')
            ax.set_xlabel('Impact sur la prédiction')
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RECHERCHE RAG
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Recherche (RAG)":
    st.title("🔍 Recherche Sémantique — RAG")
    st.markdown("Trouvez les avis les plus similaires à votre requête grâce aux embeddings Word2Vec.")
    st.markdown("---")

    @st.cache_resource
    def load_w2v_and_vectors():
        from gensim.models import Word2Vec
        w2v = Word2Vec.load("word2vec_avis.model")
        vectors = np.load("w2v_train_vectors.npy")
        return w2v.wv, vectors

    @st.cache_data
    def load_train_data():
        return pd.read_csv("avis_train_clean.csv").dropna(subset=["avis_clean"]).reset_index(drop=True)


    try:
        wv, doc_vectors = load_w2v_and_vectors()
        df_rag = load_train_data()

        # On aligne les tailles
        min_len = min(len(df_rag), len(doc_vectors))
        df_rag = df_rag.iloc[:min_len].reset_index(drop=True)
        doc_vectors = doc_vectors[:min_len]

        rag_ok = True
    except Exception as e:
        st.warning(f"⚠️ Modèle Word2Vec ou vecteurs non trouvés : {e}\n\nLancez d'abord les notebooks 03 et 04.")
        rag_ok = False

    #try:
    #    wv, doc_vectors = load_w2v_and_vectors()
     #   df_rag = load_train_data()
     #   rag_ok = True
    #except Exception as e:
     #   st.warning(f"⚠️ Modèle Word2Vec ou vecteurs non trouvés : {e}\n\nLancez d'abord les notebooks 03 et 04.")
        #rag_ok = False

    query = st.text_input(
        "🔍 Entrez votre recherche :",
        placeholder="Ex: remboursement lent service médiocre"
    )

    col1, col2, col3 = st.columns(3)
    top_k = col1.slider("Nombre de résultats", 3, 20, 5)
    note_filter = col2.multiselect("Filtrer par note", [1, 2, 3, 4, 5], default=[1,2,3,4,5])
    assureur_filter = col3.text_input("Filtrer par assureur (optionnel)")

    if st.button("🔍 Rechercher", type="primary") and query.strip() and rag_ok:
        with st.spinner("Recherche sémantique en cours..."):
            from sklearn.metrics.pairwise import cosine_similarity

            def text_to_vec(text, model, size=100):
                words = clean_text(text).split()
                vecs = [model[w] for w in words if w in model]
                return np.mean(vecs, axis=0) if vecs else np.zeros(size)

            query_vec = text_to_vec(query, wv).reshape(1, -1)
            sims = cosine_similarity(query_vec, doc_vectors)[0]

            # Filtres
            mask = df_rag['note'].isin(note_filter)
            if assureur_filter:
                mask &= df_rag['assureur'].str.contains(assureur_filter, case=False, na=False)

            sims_filtered = sims.copy()
            sims_filtered[~mask] = -1

            top_indices = sims_filtered.argsort()[::-1][:top_k]

        st.markdown(f"### 📋 Top {top_k} avis similaires à : *\"{query}\"*")
        st.markdown("---")

        for rank, idx in enumerate(top_indices, 1):
            row = df_rag.iloc[idx]
            score = sims[idx]
            sentiment = row.get('sentiment_3', 'N/A')
            emoji = {'positif': '😊', 'neutre': '😐', 'negatif': '😞'}.get(sentiment, '❓')

            with st.expander(
                f"#{rank} | Score: {score:.3f} | {row['assureur']} | "
                f"{'⭐'*int(row['note'])} | {emoji} {sentiment}"
            ):
                st.markdown(f"**Avis complet :**")
                st.markdown(f"> {row['avis']}")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Note", f"{int(row['note'])}/5")
                col_b.metric("Similarité", f"{score:.3f}")
                col_c.metric("Assureur", row['assureur'])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — QA (Questions-Réponses)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "❓ Questions (QA)":
    st.title("❓ Questions & Réponses sur les avis")
    st.markdown("Posez une question sur les avis d'un assureur et obtenez une réponse générée automatiquement.")
    st.markdown("---")

    try:
        df_qa = load_data()
        qa_data_ok = True
    except Exception as e:
        st.error(f"Données non disponibles : {e}")
        qa_data_ok = False

    if qa_data_ok:
        assureur_qa = st.selectbox(
            "🏢 Choisir un assureur",
            sorted(df_qa['assureur'].unique())
        )

        df_assureur = df_qa[df_qa['assureur'] == assureur_qa]

        # Stats rapides
        col1, col2, col3 = st.columns(3)
        col1.metric("Nb avis", f"{len(df_assureur):,}")
        col2.metric("Note moyenne", f"{df_assureur['note'].mean():.2f}/5")
        col3.metric("% Positifs", f"{(df_assureur['sentiment_3']=='positif').mean()*100:.1f}%")

        st.markdown("---")

        # Questions prédéfinies
        st.markdown("### 💬 Questions fréquentes")
        questions_predefinies = [
            "Quels sont les principaux problèmes signalés ?",
            "Qu'est-ce que les clients apprécient le plus ?",
            "Comment est évalué le service client ?",
            "Y a-t-il des problèmes de remboursement ?",
            "Que disent les clients sur les prix et tarifs ?",
        ]

        question_sel = st.selectbox("Choisir une question :", ["-- Choisir --"] + questions_predefinies)
        question_libre = st.text_input("Ou posez votre propre question :", placeholder="Ex: Les délais de traitement sont-ils bons ?")

        question = question_libre if question_libre else (question_sel if question_sel != "-- Choisir --" else None)

        if st.button("❓ Obtenir une réponse", type="primary") and question:
            with st.spinner("Analyse des avis et génération de la réponse..."):

                # Approche extractive : trouver les avis pertinents par mots-clés
                keywords_map = {
                    "problème": ["problème", "probleme", "bug", "erreur", "dysfonction"],
                    "appréci": ["excellent", "parfait", "super", "bien", "satisfait", "recommande"],
                    "service client": ["conseiller", "téléphone", "contact", "accueil", "agent"],
                    "remboursement": ["remboursement", "rembours", "paiement", "versement", "délai"],
                    "prix": ["prix", "tarif", "coût", "cher", "cotisation", "économique"],
                    "délai": ["délai", "lent", "rapide", "attendre", "semaine", "mois"],
                }

                # Mots clés de la question
                question_lower = question.lower()
                relevant_keywords = []
                for key, synonyms in keywords_map.items():
                    if any(k in question_lower for k in [key] + synonyms):
                        relevant_keywords.extend(synonyms)

                if relevant_keywords:
                    pattern = '|'.join(relevant_keywords)
                    mask = df_assureur['avis'].str.contains(pattern, case=False, na=False)
                    df_relevant = df_assureur[mask]
                else:
                    df_relevant = df_assureur

                # Statistiques pour la réponse
                n_total = len(df_assureur)
                n_relevant = len(df_relevant)
                note_moy = df_relevant['note'].mean() if len(df_relevant) > 0 else df_assureur['note'].mean()
                pct_pos = (df_relevant['sentiment_3'] == 'positif').mean() * 100 if len(df_relevant) > 0 else 0
                pct_neg = (df_relevant['sentiment_3'] == 'negatif').mean() * 100 if len(df_relevant) > 0 else 0

                # Exemples d'avis positifs et négatifs pertinents
                ex_pos = df_relevant[df_relevant['sentiment_3'] == 'positif']['avis'].head(2).tolist()
                ex_neg = df_relevant[df_relevant['sentiment_3'] == 'negatif']['avis'].head(2).tolist()

            st.markdown("---")
            st.markdown(f"### 💬 Réponse pour **{assureur_qa}**")
            st.markdown(f"*Question : {question}*")
            st.markdown("---")

            # Réponse structurée
            st.markdown(f"""
            Sur **{n_total} avis** analysés pour {assureur_qa}, 
            **{n_relevant} avis** sont pertinents pour cette question.

            📊 **Statistiques** :
            - Note moyenne sur ces avis : **{note_moy:.2f}/5**
            - Avis positifs : **{pct_pos:.1f}%**
            - Avis négatifs : **{pct_neg:.1f}%**
            """)

            if ex_pos:
                st.success("**Ce que disent les clients satisfaits :**")
                for ex in ex_pos:
                    st.markdown(f"> *\"{ex[:250]}\"*")

            if ex_neg:
                st.error("**Ce que disent les clients insatisfaits :**")
                for ex in ex_neg:
                    st.markdown(f"> *\"{ex[:250]}\"*")

            if not ex_pos and not ex_neg:
                st.info("Pas assez d'avis pertinents trouvés pour cette question.")

            # Avis les plus récents
            st.markdown("---")
            st.markdown("### 📅 Avis les plus récents")
            recent = df_assureur.sort_values('date_publication', ascending=False).head(5)
            for _, row in recent.iterrows():
                emoji = {'positif': '😊', 'neutre': '😐', 'negatif': '😞'}.get(row.get('sentiment_3', ''), '❓')
                st.markdown(f"{emoji} {'⭐'*int(row['note'])} — *\"{row['avis'][:200]}\"*")
