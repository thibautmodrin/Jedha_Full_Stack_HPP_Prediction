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
                # Vérifier et traiter les valeurs manquantes
                if data.isna().any().any():
                    # Compter le nombre de lignes avant le dropna
                    rows_before = len(data)
                    
                    # Supprimer les lignes avec des valeurs manquantes
                    data = data.dropna()
                    
                    # Compter le nombre de lignes après le dropna
                    rows_after = len(data)
                    rows_removed = rows_before - rows_after
                    
                    # Afficher un message d'avertissement
                    st.warning(f"Des valeurs manquantes ont été détectées dans les données. {rows_removed} lignes contenant des valeurs manquantes ont été supprimées. Nombre de lignes restantes: {rows_after}.")
                
                # Extraire le préprocesseur du pipeline si le modèle est un pipeline
                if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                    preprocessor = model.named_steps['preprocessor']
                    # Prétraiter les données
                    X_processed = preprocessor.transform(data)
                    # Faire la prédiction avec le modèle
                    classifier = model.named_steps['classifier']
                    predictions = classifier.predict(X_processed)
                    prediction_probs = classifier.predict_proba(X_processed)[:, 1]
                else:
                    # Si le modèle n'a pas de préprocesseur explicite, retourner une erreur
                    st.error("Le modèle ne contient pas de préprocesseur. Un préprocesseur est nécessaire pour traiter les données avant la prédiction.")
                    raise ValueError("Modèle sans préprocesseur détecté. Impossible de continuer la prédiction.")

                # Ajouter les résultats au DataFrame
                data['Prédiction'] = predictions
                data['Probabilité_HPP (%)'] = prediction_probs * 100  # Convertir en pourcentage

                # Trier les résultats par probabilité d'HPP décroissante
                data_sorted = data.sort_values(by='Probabilité_HPP (%)', ascending=False)

                # Afficher les résultats
                st.write("### Résultats des Prédictions (triés par probabilité d'HPP décroissante) :")
                st.dataframe(data_sorted)
                
                # Afficher les métriques si disponibles
                if 'y' in data.columns:
                    st.write("### Métriques d'évaluation :")
                    precision = precision_score(data['y'], predictions, zero_division=0)
                    recall = recall_score(data['y'], predictions, zero_division=0)
                    f1 = f1_score(data['y'], predictions, zero_division=0)
                    st.write(f"Précision: {precision:.4f} - Mesure la proportion de cas positifs correctement identifiés parmi tous les cas prédits positifs.")
                    st.write(f"Rappel: {recall:.4f} - Mesure la proportion de cas positifs correctement identifiés parmi tous les cas réellement positifs.")
                    st.write(f"Score F1: {f1:.4f} - Moyenne harmonique entre précision et rappel, équilibrant ces deux métriques.")
                    
                    st.info("Pour la prédiction d'HPP sévère, un rappel élevé est particulièrement important car il indique la capacité du modèle à identifier correctement les cas à risque, minimisant ainsi les faux négatifs qui pourraient être dangereux dans un contexte médical.")

                # Bouton de téléchargement des résultats (triés)
                csv = data_sorted.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Télécharger les résultats en CSV",
                    data=csv,
                    file_name='resultats_predictions.csv',
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {e}")
                st.write("Veuillez vérifier que votre fichier contient toutes les variables nécessaires au modèle.")
