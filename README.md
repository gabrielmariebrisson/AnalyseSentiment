# 📊 Analyse de Sentiment avec GloVe et CNN

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/gabrielmariebrisson/AnalyseSentiment/ci.yml?branch=main&style=for-the-badge&logo=github)](https://github.com/gabrielmariebrisson/AnalyseSentiment/actions)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)


Application web interactive d'analyse de sentiment basée sur un modèle de deep learning utilisant des embeddings GloVe et une architecture CNN (Convolutional Neural Network). Le projet permet d'analyser le sentiment de textes en temps réel via une interface Streamlit multilingue.

## 🎯 Description

Ce projet implémente un classifieur de sentiment pour tweets/textes en anglais, développé dans le cadre de la certification TensorFlow. Le modèle utilise :

- **Embeddings GloVe** (Global Vectors for Word Representation) de Stanford pour la représentation vectorielle des mots
- **Architecture CNN** avec couches Conv1D pour capturer les relations temporelles dans les séquences textuelles
- **Techniques de régularisation** (dropout) pour éviter le surajustement

### Performances

- **Précision sur les données de test** : 75%
- **Précision sur les données d'entraînement** : 79%
- **Temps d'exécution** : ~0.21 seconde par prédiction
- **Poids du modèle** : 52 Mo

## 🚀 Installation

### Prérequis

- Python 3.9 ou supérieur
- pip (gestionnaire de paquets Python)

### Étapes d'installation

1. **Cloner le dépôt** (ou télécharger les fichiers) :
   ```bash
   git clone https://github.com/gabrielmariebrisson/AnalyseSentiment.git
   cd AnalyseSentiment
   ```

2. **Créer un environnement virtuel** (recommandé) :
   ```bash
   python -m venv venv
   
   # Sur Linux/Mac
   source venv/bin/activate
   
   # Sur Windows
   venv\Scripts\activate
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

4. **Vérifier que les fichiers du modèle sont présents** :
   - `templates/assets/AnalyseSentiment.h5` (modèle TensorFlow)
   - `templates/assets/tokenizer.pkl` (tokenizer sérialisé)

## 💻 Utilisation

### Lancer l'application Streamlit

```bash
streamlit run app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

### Fonctionnalités

- **Analyse de sentiment en temps réel** : Entrez un texte en anglais et obtenez une prédiction (positif/négatif) avec un score de confiance
- **Interface multilingue** : Support de 10 langues (Français, Anglais, Espagnol, Allemand, Italien, Portugais, Japonais, Chinois, Arabe, Russe)
- **Visualisations** : Graphiques de précision et de perte du modèle
- **Documentation intégrée** : Présentation de l'architecture, des résultats et des coûts de développement

## 🐳 Déploiement avec Docker

L'application peut être déployée facilement avec Docker pour une installation et un déploiement simplifiés.

### Prérequis

- Docker installé sur votre système
- Voir [la documentation Docker](https://docs.docker.com/get-docker/) pour l'installation

### Construire l'image Docker

```bash
docker build -t analyse-sentiment .
```

### Lancer le conteneur

```bash
docker run -p 8501:8501 analyse-sentiment
```

L'application sera accessible à l'adresse : `http://localhost:8501`

### Options avancées

```bash
# Lancer en arrière-plan (détaché)
docker run -d -p 8501:8501 --name sentiment-app analyse-sentiment

# Voir les logs
docker logs sentiment-app

# Arrêter le conteneur
docker stop sentiment-app

# Redémarrer le conteneur
docker start sentiment-app
```

## 🔄 CI/CD avec GitHub Actions

Le projet inclut un workflow GitHub Actions qui exécute automatiquement les tests à chaque push sur la branche `main`.

### Workflow CI

Le workflow `.github/workflows/ci.yml` :
- S'active automatiquement sur les push et pull requests vers `main`
- Teste le code sur Python 3.9, 3.10 et 3.11
- Installe les dépendances et exécute la suite de tests avec `pytest`
- Utilise le cache pip pour accélérer les builds

### Vérifier le statut des tests

Les résultats des tests sont visibles dans l'onglet **Actions** de votre dépôt GitHub. Un badge de statut peut être ajouté au README pour afficher le statut des tests.

## 🏗️ Architecture

### Choix technologiques

#### GloVe (Global Vectors for Word Representation)

GloVe a été choisi pour sa capacité à capturer les relations sémantiques entre les mots en analysant les cooccurrences globales dans un corpus. Contrairement à Word2Vec, GloVe combine les avantages des méthodes globales (comme LSA) et locales (comme Word2Vec).

**Avantages** :
- Représentations vectorielles riches en informations sémantiques
- Bonne généralisation sur des mots non vus pendant l'entraînement
- Permet des opérations vectorielles (ex: roi - homme + femme ≈ reine)

#### Architecture CNN avec Conv1D

L'architecture utilise des couches de convolution 1D plutôt que des LSTM bidirectionnels pour plusieurs raisons :

**Avantages** :
- **Performance** : Temps d'entraînement plus rapide (13 minutes sur Google Colab)
- **Efficacité** : Moins de paramètres à entraîner
- **Capture de patterns locaux** : Les filtres convolutionnels détectent efficacement les n-grammes et patterns locaux dans les séquences

**Architecture du modèle** :
1. **Couche d'Embedding** : Intègre la matrice GloVe préentraînée (dimension ajustable)
2. **Couche Conv1D** : Capture les relations temporelles et les patterns locaux
3. **Couche Dense** : Agrège les informations pour la classification binaire

### Structure du projet

```
AnalyseSentiment/
├── app.py                 # Application Streamlit principale
├── Dockerfile             # Configuration Docker pour le déploiement
├── .dockerignore          # Fichiers exclus du build Docker
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration et constantes
│   ├── model_service.py  # Service de prédiction (SentimentModel)
│   └── utils.py          # Utilitaires (traduction, padding)
├── tests/
│   ├── conftest.py       # Fixtures pytest
│   ├── test_model.py     # Tests unitaires du modèle
│   └── test_utils.py     # Tests unitaires des utilitaires
├── templates/
│   └── assets/
│       ├── AnalyseSentiment.h5  # Modèle sauvegardé
│       ├── tokenizer.pkl        # Tokenizer sérialisé
│       └── images/              # Graphiques de résultats
├── .github/
│   └── workflows/
│       └── ci.yml         # Workflow GitHub Actions pour CI/CD
├── requirements.txt       # Dépendances Python
└── README.md             # Documentation du projet
```

## 🧪 Tests

Le projet inclut une suite de tests unitaires complète utilisant `pytest`.

### Lancer les tests

```bash
# Tous les tests
pytest tests/ -v

# Tests avec couverture de code
pytest tests/ --cov=src --cov-report=html

# Un fichier spécifique
pytest tests/test_model.py -v
pytest tests/test_utils.py -v
```

### Structure des tests

- **`test_model.py`** : Tests de la classe `SentimentModel` avec mocks de TensorFlow
- **`test_utils.py`** : Tests de la fonction `seq_pad_and_trunc` avec cas limites
- **`conftest.py`** : Fixtures partagées et mocks des dépendances (Streamlit, TensorFlow, deep-translator)

## 📊 Dataset

Le modèle a été entraîné sur le **dataset Sentiment140**, qui contient :
- **1,6 million de tweets** étiquetés par sentiment
- **Labels** : 0 (négatif) et 4 (positif)
- **Répartition** : 90% entraînement / 10% test

## 🔧 Développement

### Environnement d'entraînement

- **Plateforme** : Google Colab
- **Durée d'entraînement** : 13 minutes
- **Spécifications** :
  - Processeur : Intel Xeon (2.2 GHz)
  - RAM : 12.67 GiB

## 📝 Documentation du code

Le code suit les standards PEP 8 et utilise des docstrings au format **Google Style** pour toutes les classes et fonctions majeures. Exemple :

```python
def predict(self, sentence: str) -> Any:
    """Prédit le sentiment d'un texte donné.

    Args:
        sentence: Texte à analyser (en anglais de préférence).

    Returns:
        Tableau numpy de shape (1, 1) contenant la probabilité que le sentiment
        soit positif. Valeur proche de 0 = négatif, proche de 1 = positif.

    Raises:
        ValueError: Si le texte ne peut pas être tokenisé.
        RuntimeError: Si le modèle échoue lors de la prédiction.
    """
```

## 👤 Auteur

**Gabriel Marie-Brisson**

- Portfolio : [gabriel.mariebrisson.fr](https://gabriel.mariebrisson.fr)
- GitHub : [@gabrielmariebrisson](https://github.com/gabrielmariebrisson)

## 📄 License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- **Stanford NLP** pour les embeddings GloVe
- **TensorFlow/Keras** pour le framework de deep learning
- **Streamlit** pour l'interface web interactive
- **Google Colab** pour les ressources de calcul gratuites

---

⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !

