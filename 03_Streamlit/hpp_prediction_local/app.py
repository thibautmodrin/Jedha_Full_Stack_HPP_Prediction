import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer

# Définir le chemin du modèle
model_path = 'best_model_logreg_f1_Sans_resampling.joblib'

# Vérifier si le fichier existe
if not os.path.exists(model_path):
    st.error(f"Le fichier modèle '{model_path}' n'existe pas. Veuillez vérifier le chemin du fichier.")
    model = None
else:
    # Charger le modèle entraîné
    try:
        model = joblib.load(model_path)
        st.success("Modèle chargé avec succès!")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        model = None

# Titre de l'application
st.title("🩺 Application de Prédiction d'HPP sévère")

# Uploader le fichier CSV
uploaded_file = st.file_uploader("Importer votre fichier CSV avec les variables nécessaires", type="csv")

if uploaded_file and model is not None:
    # Lire le CSV
    data = pd.read_csv(uploaded_file)
    st.write("### Données importées :")
    st.dataframe(data.head())

    # Vérifier si les données sont compatibles avec le modèle
    if st.button("Prédire"):
        with st.spinner('Prédictions en cours...'):
            try:
                # Faire la prédiction avec le pipeline joblib
                predictions = model.predict(data)
                prediction_probs = model.predict_proba(data)[:, 1]  # Probabilité classe positive

                # Ajouter les résultats au DataFrame
                data['Prédiction'] = predictions
                data['Probabilité_HPP'] = prediction_probs

                # Afficher les résultats
                st.write("### Résultats des Prédictions :")
                st.dataframe(data)
                
                # Afficher les métriques si disponibles
                if 'y' in data.columns:
                    st.write("### Métriques d'évaluation :")
                    precision = precision_score(data['y'], predictions, zero_division=0)
                    recall = recall_score(data['y'], predictions, zero_division=0)
                    f1 = f1_score(data['y'], predictions, zero_division=0)
                    st.write(f"Précision: {precision:.4f}")
                    st.write(f"Rappel: {recall:.4f}")
                    st.write(f"Score F1: {f1:.4f}")

                # Bouton de téléchargement des résultats
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Télécharger les résultats en CSV",
                    data=csv,
                    file_name='resultats_predictions.csv',
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {e}")
                st.write("Veuillez vérifier que votre fichier contient toutes les variables nécessaires au modèle.")
