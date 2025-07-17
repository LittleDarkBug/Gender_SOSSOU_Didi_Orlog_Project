# Détection Automatique de Genre pour la Carte d'Identité Nationale du Togo

## Description du Projet

Ce projet vise à développer un système de **détection automatique du genre** basé sur l'analyse des prénoms et noms de famille pour les cartes d'identité nationales du Togo. L'objectif principal est de créer un modèle de machine learning capable de prédire le genre (masculin/féminin) d'une personne en se basant uniquement sur son prénom et nom de famille.

### Contexte
Dans le cadre de la modernisation des services d'état civil au Togo, ce système peut aider à automatiser et valider les informations de genre lors de l'émission des cartes d'identité nationales, réduisant ainsi les erreurs manuelles et accélérant le processus de traitement.

### Auteur
**Didi Orlog SOSSOU**

## Méthodologie

### Étape 1: Chargement et Exploration des Données
- **Dataset principal**: `dataset_names.csv` contenant 60 000 enregistrements
- **Structure**: prénom, nom de famille, genre (1 = Masculin, 0 = Féminin)
- **Vérification**: Aucun doublon détecté dans le dataset
- **Dataset de test**: `test_dataset.csv` avec 47 noms d'une promotion d'étudiants

### Étape 2: Feature Engineering (Ingénierie des Caractéristiques)
Le projet utilise des caractéristiques linguistiques avancées pour l'analyse des prénoms :

- **Terminaisons féminines** : Analyse des suffixes typiquement féminins (-a, -e, -ine, etc.)
- **Ratios de genre** : Proportions de caractéristiques masculines vs féminines
- **Douceur/Dureté phonétique** : Analyse des consonnes douces et dures
- **Densité vocalique** : Proportion de voyelles dans le prénom
- **Bigrammes** : Analyse des paires de lettres caractéristiques
- **Lettres récurrentes** : Détection de répétitions de lettres
- **Longueur et structure** : Longueur du prénom et patterns structurels

### Étape 3: Séparation des Ensembles
- **Ensemble d'entraînement** : 80% des données (48 000 enregistrements)
- **Ensemble de test** : 20% des données (12 000 enregistrements)
- Stratification pour maintenir l'équilibre des classes

### Étape 4: Entraînement des Modèles

#### RandomForest Classifier
- **Architecture** : Forêt aléatoire avec optimisation des hyperparamètres
- **Features** : Toutes les caractéristiques linguistiques extraites
- **Validation** : Validation croisée et métriques complètes

#### CatBoost Classifier
- **Architecture** : Gradient boosting optimisé pour les variables catégorielles
- **Avantages** : Gestion native des features catégorielles, robustesse
- **Performance** : Optimisation automatique des hyperparamètres

### Étape 5: Évaluation des Modèles

#### Métriques d'Évaluation
- **Précision (Accuracy)**
- **Précision (Precision)**
- **Rappel (Recall)**
- **Score F1**
- **Matrice de Confusion**

### Étape 6: Prédiction sur Nouveaux Noms
Implémentation d'une fonction de prédiction utilisable en production.

## Résultats Principaux

### Performance RandomForest
- **Ensemble d'entraînement** :
  - Précision : 99.98%
  - Precision : 99.96%
  - Rappel : 100.00%
  - Score F1 : 99.98%

- **Ensemble de test** :
  - Précision : **97.22%**
  - Precision : 96.74%
  - Rappel : 97.84%
  - Score F1 : 97.29%

- **Test sur données de classe** : **97.83%** de précision

### Performance CatBoost
- **Ensemble d'entraînement** :
  - Précision : 96.16%
  - Precision : 94.31%
  - Rappel : 98.26%
  - Score F1 : 96.24%

- **Ensemble de test** :
  - Précision : **95.89%**
  - Precision : 94.10%
  - Rappel : 97.94%
  - Score F1 : 95.98%

- **Test sur données de classe** : **100.00%** de précision

## Exemple d'Utilisation

### Fonction de Prédiction

```python
def predict_gender(full_name):
    """
    Prédit le genre basé sur le nom complet donné.
    
    Args:
        full_name (str): Le nom complet (prénom suivi du nom de famille)
    
    Returns:
        int: 1 si Masculin, 0 si Féminin
        
    Raises:
        ValueError: Si l'input ne contient pas prénom et nom
    """
    # Séparation prénom/nom
    parts = full_name.strip().split()
    if len(parts) < 2:
        raise ValueError("Veuillez fournir prénom et nom séparés par un espace.")
    
    first_name = parts[0].lower()
    last_name = " ".join(parts[1:])
    
    # Création du DataFrame d'entrée
    input_data = pd.DataFrame({
        'firstname': [first_name],
        'lastname': [last_name]
    })
    
    # Ingénierie des caractéristiques
    input_data = feature_engineering(input_data)
    input_data_ = input_data.drop(columns=['lastname', 'firstname'])
    
    # Prédiction
    prediction = catboost_classifier.predict(input_data_)
    return 1 if prediction[0] == 1 else 0

# Exemple d'utilisation
sexe = "Homme" if predict_gender("Ruth AYITTEY") == 1 else "Femme"
print(f"Sexe prédit : {sexe}")  # Output: Femme
```

## Prérequis et Installation

### Prérequis Python
- Python 3.7+
- pip ou conda pour la gestion des packages

### Dépendances Requises

```bash
pip install pandas scikit-learn catboost matplotlib seaborn jupyter
```

### Installation Détaillée

```bash
# Cloner le repository
git clone https://github.com/LittleDarkBug/Gender_SOSSOU_Didi_Orlog_Project.git
cd Gender_SOSSOU_Didi_Orlog_Project

# Installer les dépendances
pip install -r requirements.txt  # Si disponible
# Ou installer manuellement :
pip install pandas>=1.3.0 scikit-learn>=1.0.0 catboost>=1.0.0 matplotlib>=3.5.0 seaborn>=0.11.0 jupyter

# Lancer le notebook
jupyter notebook Gender_SOSSOU_Didi_Orlog_Notebook.ipynb
```

### Configuration Alternative avec Conda

```bash
conda create -n gender-detection python=3.9
conda activate gender-detection
conda install pandas scikit-learn matplotlib seaborn jupyter
pip install catboost
```

## Structure du Projet

```
Gender_SOSSOU_Didi_Orlog_Project/
│
├── README.md                              # Documentation principale
├── Gender_SOSSOU_Didi_Orlog_Notebook.ipynb   # Notebook principal
├── dataset_names.csv                      # Dataset d'entraînement (60K noms)
├── test_dataset.csv                       # Dataset de test (47 noms de classe)
│
└── .git/                                  # Contrôle de version Git
```

### Description des Fichiers

- **`Gender_SOSSOU_Didi_Orlog_Notebook.ipynb`** : Notebook Jupyter contenant l'analyse complète, le preprocessing, l'entraînement des modèles et l'évaluation
- **`dataset_names.csv`** : Dataset principal avec 60 000 noms étiquetés (prénom, nom, genre)
- **`test_dataset.csv`** : Dataset de validation avec 47 noms d'étudiants pour test en conditions réelles
- **`README.md`** : Documentation complète du projet (ce fichier)

## Utilisation du Modèle

### Chargement du Notebook
1. Ouvrir `Gender_SOSSOU_Didi_Orlog_Notebook.ipynb` dans Jupyter
2. Exécuter toutes les cellules pour entraîner les modèles
3. Utiliser la fonction `predict_gender()` pour de nouvelles prédictions

### Intégration en Production
Le modèle CatBoost peut être sauvegardé et chargé pour utilisation en production :

```python
# Sauvegarder le modèle
catboost_classifier.save_model('gender_model.cbm')

# Charger le modèle
from catboost import CatBoostClassifier
model = CatBoostClassifier()
model.load_model('gender_model.cbm')
```

## Limitations et Améliorations Possibles

### Limitations Actuelles
- Basé uniquement sur les noms togolais et ouest-africains
- Performance variable selon l'origine culturelle des noms
- Nécessite prénom ET nom de famille

### Améliorations Futures
- Élargir le dataset à d'autres cultures
- Intégrer des features phonétiques avancées
- Développer une API REST pour intégration système
- Ajouter la détection de confiance de prédiction
- Interface web pour utilisation interactive

## Auteurs et Contributions

### Auteur Principal
- **Didi Orlog SOSSOU** - Développement et implémentation complète

### Contributions
- Conception de l'architecture du modèle
- Développement des features linguistiques
- Implémentation des algorithmes RandomForest et CatBoost
- Évaluation et validation des performances
- Documentation et présentation

## Licence

Ce projet est développé dans un cadre éducatif et de recherche. Pour toute utilisation commerciale ou redistribution, veuillez contacter l'auteur.

---

## Contact

Pour toute question, suggestion ou collaboration :
- **Auteur** : Didi Orlog SOSSOU
- **Projet** : Détection Automatique de Genre - Carte d'Identité Nationale du Togo

---

*Dernière mise à jour : 2024*