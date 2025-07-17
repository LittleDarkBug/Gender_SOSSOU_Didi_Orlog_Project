# Détection Automatique de Genre pour la Carte d'Identité Nationale du Togo

## Description du Projet

Ce projet vise à développer un système de classification automatique pour déterminer le genre (masculin/féminin) à partir des prénoms et noms de famille présents sur les cartes d'identité nationales du Togo. Cette solution s'appuie sur des techniques d'apprentissage automatique et d'ingénierie des caractéristiques linguistiques.

### Contexte et Objectifs

- **But** : Automatiser la détection du genre basée sur l'analyse des noms togolais
- **Contexte** : Améliorer les processus de traitement des documents d'identité
- **Approche** : Utilisation de modèles d'apprentissage automatique (RandomForest et CatBoost)
- **Auteur** : Didi Orlog SOSSOU

## Méthodologie

### Étapes Principales

1. **Chargement et exploration des données**
   - Import du dataset principal (`dataset_names.csv`) contenant prénoms, noms et genres
   - Analyse exploratoire des données d'entraînement
   - Chargement du dataset de test externe (`test_dataset.csv`)

2. **Ingénierie des caractéristiques (Feature Engineering)**
   - Extraction de caractéristiques linguistiques des prénoms :
     - Analyse des terminaisons genrées (suffixes féminins/masculins)
     - Comptage des voyelles et consonnes
     - Longueur des prénoms
     - Ratios et proportions linguistiques
   - Création de variables dérivées pour améliorer la performance des modèles

3. **Séparation des ensembles de données**
   - Division train/test (80%/20%) avec stratification
   - Préparation des features (X) et des labels (y)
   - Encodage : 1 = Masculin, 0 = Féminin

4. **Entraînement des modèles**
   - **RandomForest Classifier** : Ensemble de arbres de décision
   - **CatBoost Classifier** : Algorithme de gradient boosting optimisé

5. **Évaluation des performances**
   - Métriques calculées : Accuracy, Precision, Recall, F1-Score
   - Matrice de confusion pour analyse détaillée
   - Test sur dataset externe pour validation

6. **Prédiction**
   - Fonction `predict_gender()` pour prédictions en temps réel
   - Interface simple acceptant nom complet en entrée

## Résultats Obtenus

### Performance des Modèles

#### RandomForest Classifier
- **Ensemble d'entraînement** : Accuracy = 96.18%
- **Ensemble de test** : Accuracy = 95.86%
- **Test externe** : Accuracy = 97.83%
- Precision = 94.45%, Recall = 97.46%, F1-Score = 95.93%

#### CatBoost Classifier  
- **Ensemble d'entraînement** : Accuracy = 96.16%
- **Ensemble de test** : Accuracy = 95.89%
- **Test externe** : Accuracy = 100.00%
- Precision = 94.10%, Recall = 97.94%, F1-Score = 95.98%

**CatBoost** démontre une performance supérieure avec une précision parfaite sur le dataset de test externe.

## Exemple d'Utilisation

### Fonction de Prédiction

```python
def predict_gender(full_name):
    """
    Prédit le genre basé sur le nom complet en utilisant le classificateur entraîné.
    
    Args:
        full_name (str): Le nom complet (prénom suivi du nom de famille)
        
    Returns:
        int: 1 si le modèle prédit Masculin, 0 si le modèle prédit Féminin
    """
    # Traitement et prédiction du nom
    # ... (logique d'ingénierie des caractéristiques)
    return prediction

# Exemple d'usage
nom_complet = "Ruth AYITTEY"
genre_predit = predict_gender(nom_complet)
print(f"Sexe de {nom_complet} : {'Homme' if genre_predit == 1 else 'Femme'}")
# Sortie: Sexe de Ruth AYITTEY : Femme
```

## Prérequis et Installation

### Dépendances Requises

```bash
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install catboost
pip install jupyter
```

### Installation et Exécution

1. **Cloner le repository**
```bash
git clone https://github.com/LittleDarkBug/Gender_SOSSOU_Didi_Orlog_Project.git
cd Gender_SOSSOU_Didi_Orlog_Project
```

2. **Lancer le notebook Jupyter**
```bash
jupyter notebook Gender_SOSSOU_Didi_Orlog_Notebook.ipynb
```

### Description des Fichiers

- **`Gender_SOSSOU_Didi_Orlog_Notebook.ipynb`** : Notebook Jupyter contenant l'intégralité du pipeline de machine learning
- **`dataset_names.csv`** : Dataset principal avec colonnes `firstname`, `lastname`, `gender`
- **`test_dataset.csv`** : Dataset de validation externe avec colonnes `lastname`, `firstname`, `gender`
- **`README.md`** : Documentation complète du projet (ce fichier)

## Fonctionnalités Clés

### Ingénierie des Caractéristiques

Le système extrait automatiquement plusieurs types de caractéristiques linguistiques :

- **Terminaisons genrées** : Analyse des suffixes typiquement masculins ou féminins
- **Composition phonétique** : Ratio voyelles/consonnes, patterns sonores
- **Métriques de longueur** : Longueur des prénoms et noms
- **Caractéristiques contextuelles** : Combinaisons prénom-nom spécifiques

### Modèles d'Apprentissage

- **RandomForest** : Robuste aux données bruitées, interprétable
- **CatBoost** : Performance optimisée, gestion automatique des features catégorielles

## Auteur et Licence

### Auteur
**Didi Orlog SOSSOU**  
Email : didisossou@gmail.com

### Licence
Ce projet est développé à des fins éducatives et de recherche. Pour toute utilisation commerciale ou redistribution, veuillez contacter l'auteur.

### Contributions
Les contributions, suggestions d'amélioration et rapports de bugs sont les bienvenus. N'hésitez pas à ouvrir une issue ou soumettre une pull request.

---
