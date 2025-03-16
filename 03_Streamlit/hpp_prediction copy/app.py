import streamlit as st
import pandas as pd
import joblib

# Charger le modèle entraîné (remplace par ton fichier modèle .pkl)
model = joblib.load('modele_prediction_hpp.pkl')

# Titre de l'application
st.title("🩺 Application de Prédiction d'HPP sévère")

# Uploader le fichier CSV
uploaded_file = st.file_uploader("Importer votre fichier CSV avec les variables nécessaires", type="csv")

if uploaded_file:
    # Lire le CSV
    data = pd.read_csv(uploaded_file)
    st.write("### Données importées :")
    st.dataframe(data.head())

    # Vérifier si les données sont compatibles avec le modèle
    if st.button("Prédire"):
        with st.spinner('Prédictions en cours...'):
            # Faire la prédiction
            predictions = model.predict(data)
            prediction_probs = model.predict_proba(data)[:, 1]  # Probabilité classe positive

            # Ajouter les résultats au DataFrame
            data['Prédiction'] = predictions
            data['Probabilité_HPP'] = prediction_probs

            # Afficher les résultats
            st.write("### Résultats des Prédictions :")
            st.dataframe(data)

            # Bouton de téléchargement des résultats
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger les résultats en CSV",
                data=csv,
                file_name='resultats_predictions.csv',
                mime='text/csv',
            )
