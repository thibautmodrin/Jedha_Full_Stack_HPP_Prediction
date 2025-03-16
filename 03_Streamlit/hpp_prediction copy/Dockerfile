FROM python:3.11-slim

# Définition du répertoire de travail
WORKDIR /home/app

# Copie des fichiers du projet vers l'image Docker
COPY . /home/app

# Installation des dépendances
RUN pip install --upgrade pip
RUN pip install streamlit joblib scikit-learn pandas numpy xgboost imblearn

# Exposition du port
ENV PORT=80
EXPOSE ${PORT}

# Commande d'exécution de l'application Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=80", "--server.address=0.0.0.0"]
