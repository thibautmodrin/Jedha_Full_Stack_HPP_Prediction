# 🧠 Prédiction d’Hémorragie du Post-Partum Sévère (HPP)

🌟 Projet Data Science & MLOps mené avec une maternité  
👨‍💻 Thibaut Modrin – Data Scientist & Engineer Fullstack

---

## 🌐 Contexte

Ce projet est né d’une collaboration avec un statisticien d'une maternité, responsable de l’analyse annuelle des données obstétricales régionales.

Suite à l’étude de leur rapport, l’HPP sévère est apparue comme un enjeu clinique prioritaire, justifiant la mise en place d’un outil prédictif dès l’arrivée des patientes, pour mieux anticiper les risques et mobiliser les ressources.

---

## 🎯 Objectif

Développer un modèle de Machine Learning capable de prédire le risque d’HPP sévère avant l’accouchement, à partir des données disponibles à l’admission.

---

## ⚙️ Approches & Modélisation

Plusieurs techniques ont été testées, en raison de la très forte déséquilibration des classes (HPP sévère ≈ 2 %).

| Modèle                        | Rappel (Recall) | Précision (Precision) | Points forts               | Limites                        |
|------------------------------|------------------|------------------------|----------------------------|--------------------------------|
| Régression logistique + SMOTE | 69 %             | ~8 %                   | Simple & interprétable     | Très faible précision          |
| Random Forest                 | 65 %             | ~9 %                   | Non-linéaire & robuste     | Rappel ↔ Précision difficile   |
| XGBoost                       | 66 %             | ~9 %                   | Optimisation avancée       | Résultats similaires           |

> Techniques de rééquilibrage testées : SMOTE, SMOTEENN, RandomUnderSampler, surpondération.

---

## 📌 Interprétation métier

Dans un contexte clinique :

- Un bon **recall** (*rappel* – taux de détection des cas réels) est prioritaire, afin de ne pas rater une patiente à risque, quitte à déclencher de fausses alertes.
- Une faible **precision** (*précision* – fiabilité des alertes positives) est acceptable tant qu’elle ne surcharge pas les ressources.

Ce projet s’inscrit dans une logique de triage médical assisté par IA, où le coût d’un faux négatif est bien plus élevé que celui d’un faux positif.

---

## 🖥️ Application pour non-techniciens

L’interface Streamlit permet à un personnel soignant de :

1. Renseigner les données cliniques à l’admission.
2. Obtenir une prédiction immédiate du risque d’HPP sévère.
3. (À venir) Générer un rapport PDF de synthèse à archiver.

🎯 Objectif : faciliter une prise de décision rapide, sans intervention d’un data scientist.

---

## 📦 Stack technique

| Domaine              | Outils utilisés                            |
|----------------------|--------------------------------------------|
| Modélisation         | `scikit-learn`, `XGBoost`                  |
| Suivi expérimental   | `MLflow`                                   |
| Interface utilisateur| `Streamlit`                                |
| Orchestration        | `Python`, `joblib`, `pandas`, `numpy`      |
| Déploiement prévu    | `Docker`, `Hugging Face Spaces`            |

---

## 📊 Démonstrations

- 🎬 [Vidéo démo Streamlit (MP4)](https://github.com/thibautmodrin/Jedha_Full_Stack_HPP_Prediction/blob/main/03_Streamlit/Demo_Streamlit_HPP_Prediction.mp4)
- 🔍 [Suivi MLflow des modèles](https://thibautmodrin-mlflow.hf.space/)
- 🧩 [Diagramme Excalidraw du workflow](https://excalidraw.com/#json=rnFRCGx3gE_yHOTYRAbOv,cYMyt0FhFy2jG_s2-aF4bw)

---

## 🛠️ Installation locale

```bash
git clone https://github.com/thibautmodrin/Jedha_Full_Stack_HPP_Prediction.git
cd Jedha_Full_Stack_HPP_Prediction
pip install -r requirements.txt
```
---

## Exemple de prédiction en Python
```bash
import joblib
model = joblib.load('hpp_severe_prediction_model.pkl')
patient_data = [[valeur1, valeur2, ..., valeurN]]
prediction = model.predict(patient_data)
print(prediction)
```
---

## 🔐 Confidentialité & éthique
- Toutes les données ont été anonymisées avant traitement.

- Aucun identifiant personnel n’est stocké ni exploité.

- Le projet respecte les normes du RGPD en vigueur.

---

## 🛤️ Améliorations futures
- Intégration de données post-accouchement.

- Génération automatique de rapports PDF.

- Déploiement cloud sécurisé (Streamlit Cloud / AWS / Azure).

- Implémentation de tests unitaires.

- Intégration CI/CD (GitHub Actions).

---

## 🙏 Remerciements
Merci à l’équipe de la Maternité pour la mise à disposition des données et leur soutien dans l’exploration de solutions innovantes pour la santé publique.

---

## 📬 Contact
📧 Email : thibaut.modrin@gmail.com

🔗 LinkedIn : https://www.linkedin.com/in/thibautmodrin

🧰 Portfolio GitHub : https://github.com/thibautmodrin
