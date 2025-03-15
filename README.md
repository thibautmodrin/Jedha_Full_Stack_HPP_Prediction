# Jedha_Full_Stack_HPP_Prediction

🌟 Prédiction d'Hémorragie du Post-Partum Sévère (HPP)

🚀 Projet de Machine Learning pour la Maternité de Bourgogne  
💪 Créateur : [Ton Nom]

## 🌐 Contexte
Ce projet est né d'une collaboration avec un statisticien travaillant à la maternité de Bourgogne, responsable de la compilation annuelle des données obstétricales régionales. Après avoir analysé le rapport annuel, l'HPP sévère est apparue comme une préoccupation majeure pour la maternité, justifiant la création d'un modèle prédictif basé sur des données disponibles avant l'accouchement.

## 🎯 Objectif
L'objectif principal est de prédire la survenue d'une HPP sévère dès l'arrivée d'une patiente à la maternité, à partir des données exclusivement recueillies avant l'accouchement. Différentes approches de gestion du déséquilibre de classes (classe minoritaire) ont été explorées, notamment :
- Régression logistique avec différents types de rééquilibrage (SMOTE, SMOTEENN, RandomUnderSampler).
- XGBoost et Random Forest avec optimisation hyperparamétrique.
- Réseau de neurones avec gestion du déséquilibre.

## 📝 Modélisation et Expériences

### 📌 1. Régression Logistique avec gestion du déséquilibre (SMOTE et variantes)

**Pourquoi cette approche ?**
- Simplicité d'interprétation et rapidité d'exécution.
- Gestion efficace du déséquilibre via SMOTE et ses variantes.

**Observations :**
- Bon rappel obtenu (>60%) mais précision insatisfaisante (<10%).
- SMOTEENN a légèrement amélioré les résultats.

### 📌 2. Random Forest et XGBoost

**Pourquoi ces modèles ?**
- Capacité à capturer les relations non-linéaires complexes.
- Bonne gestion naturelle des données déséquilibrées avec hyperparamétrage optimisé.

**Observations :**
- Difficulté à atteindre un bon équilibre précision-rappel.
- Recall élevé mais faible précision persistante.

### 📌 3. Réseau de Neurones

**Pourquoi cette approche ?**
- Possibilité de modéliser des interactions complexes entre les variables.
- Adaptation à des jeux de données larges.

**Observations :**
- Modèle complexe et performant en recall mais toujours limité en précision.
- Ressources computationnelles importantes nécessaires.

## 📊 Synthèse des résultats actuels

| Modèle | Recall | Précision | Points forts | Limitations |
|--------|--------|-----------|--------------|-------------|
| Logistic Regression (SMOTE) | 69% | ~8% | Bonne interprétabilité, rapide | Très faible précision |
| Random Forest / XGBoost | ~65% | ~9% | Modèles robustes | Difficulté à équilibrer précision-recall |
| Réseau de neurones | ~70% | ~6% | Modèle puissant | Complexité élevée, ressource computationnelle |

## 🔍 Conclusion intermédiaire

Les résultats montrent une grande difficulté à obtenir simultanément une bonne précision et un rappel élevé, reflétant la complexité du problème (classe extrêmement déséquilibrée). Des pistes d'amélioration incluent l'identification d'une autre variable cible ou l'utilisation de méthodes avancées spécifiques à l'imbalancing.

## 🤖 Tracking & Déploiement via Hugging Face

- **MLflow** - [Lien vers MLflow](https://ton-lien-mlflow.hf.space)
- **Application (Streamlit)** - [Lien vers ton application](https://huggingface.co/spaces/ton-app-streamlit)

## 🛠️ Installation et Utilisation

Cloner le projet depuis GitHub :

```bash
git clone https://github.com/ton-github/prediction-hpp-severe.git
cd prediction-hpp-severe
pip install -r requirements.txt
```

Exemple de prédiction :

```python
import joblib

model = joblib.load('hpp_severe_prediction_model.pkl')
patient_data = [[valeur1, valeur2, ..., valeurN]]
prediction = model.predict(patient_data)
print(prediction)
```

## 📂 Structure du projet GitHub

```
.
├── notebooks/          # Carnets Jupyter exploratoires
├── streamlit/          # Application Streamlit (POC)
├── mlflow/             # Expériences MLflow
├── data/               # Datasets utilisés
├── README.md           # Documentation du projet
└── requirements.txt    # Dépendances Python
```

## 📚 Ressources
- Lien vers le rapport annuel de la maternité (si disponible)
- Données anonymisées disponibles [ici](lien-vers-données)

---

✨ Merci à la maternité de Bourgogne pour les données fournies et leur collaboration précieuse dans ce projet ! 🌟

