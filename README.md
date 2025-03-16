# Keyword Clustering Tool

Un outil de clustering de mots-clés basé sur OpenAI qui associe des clusters à des URLs pertinentes.

## Description

Cette API FastAPI permet de :
- Clusteriser une liste de mots-clés en groupes sémantiques
- Associer chaque cluster à l'URL la plus pertinente
- Classifier les URLs par typologie de pages (optionnel)
- Filtrer les résultats par cluster, URL, typologie de page et score de similarité
- Télécharger les résultats filtrés au format CSV
- Générer un fichier CSV avec les résultats complets

## Prérequis

- Python 3.8+
- Une clé API OpenAI

## Installation

### Méthode 1 : Installation locale

1. Cloner le dépôt :
```bash
git clone https://github.com/BenSlashr/kw-clustering.git
cd kw-clustering
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Lancer l'application :
```bash
uvicorn main:app --reload
```

L'API sera disponible à l'adresse : http://localhost:8000

### Méthode 2 : Utilisation avec Docker

1. Construire l'image Docker :
```bash
docker build -t kw-clustering .
```

2. Lancer le conteneur :
```bash
docker run -p 8000:8000 kw-clustering
```

L'API sera disponible à l'adresse : http://localhost:8000

## Utilisation de l'API

### Interface Web

Accédez à l'interface Swagger UI à l'adresse : http://localhost:8000/docs

### Format des fichiers d'entrée

1. **Fichier CSV des mots-clés** :
   - Première colonne : liste des mots-clés
   - Séparateurs acceptés : virgule (,) ou point-virgule (;)

   Exemple avec virgule :
   ```
   mot-clé
   référencement naturel
   SEO
   optimisation moteur de recherche
   ```

2. **Fichier CSV des URLs** :
   - Première colonne : URL
   - Deuxième colonne : Contenu de la page
   - Troisième colonne (optionnelle) : Type de page (typologie)
   - Séparateurs acceptés : virgule (,) ou point-virgule (;)

   Exemple avec virgule :
   ```
   url,contenu,typologie
   https://example.com/seo,Contenu sur le référencement naturel et l'optimisation pour les moteurs de recherche,blog
   https://example.com/social,Contenu sur les réseaux sociaux et le marketing digital,blog
   https://example.com/produit,Description d'un produit spécifique,produit
   https://example.com/categorie,Liste de produits d'une catégorie,catégorie
   ```
   
   Exemple avec point-virgule :
   ```
   url;contenu;typologie
   https://example.com/seo;Contenu sur le référencement naturel et l'optimisation pour les moteurs de recherche;blog
   https://example.com/social;Contenu sur les réseaux sociaux et le marketing digital;blog
   https://example.com/produit;Description d'un produit spécifique;produit
   https://example.com/categorie;Liste de produits d'une catégorie;catégorie
   ```
   
### Paramètres de l'API

- **keywords_file** : Fichier CSV contenant les mots-clés
- **urls_file** : Fichier CSV contenant les URLs et leur contenu
- **api_key** : Clé API OpenAI
- **clustering_algorithm** : Algorithme de clustering (kmeans ou dbscan)
- **n_clusters** : Nombre de clusters (pour KMeans)
- **eps** : Paramètre epsilon (pour DBSCAN)
- **min_samples** : Paramètre min_samples (pour DBSCAN)

### Format du fichier de sortie

Le fichier CSV de sortie contient :
- Colonne A : Mot-clé
- Colonne B : Cluster
- Colonne C : URL associée

## Méthodes d'Embedding

L'outil propose désormais deux méthodes pour générer les embeddings :

1. **OpenAI** :
   - Utilise l'API OpenAI et le modèle `text-embedding-ada-002`
   - Nécessite une clé API OpenAI
   - Avantages : Très performant, embeddings de haute qualité
   - Inconvénients : Nécessite une connexion internet et un compte OpenAI, coûts associés à l'utilisation de l'API

2. **Sentence Transformers** (nouveau) :
   - Utilise la bibliothèque Sentence Transformers pour générer des embeddings localement
   - Ne nécessite pas de clé API
   - Modèles disponibles :
     - `all-MiniLM-L6-v2` : Modèle léger et rapide (par défaut)
     - `all-mpnet-base-v2` : Modèle plus précis mais plus lent
     - `paraphrase-multilingual-MiniLM-L12-v2` : Modèle multilingue
   - Avantages : Fonctionne hors ligne, gratuit, pas de limite d'utilisation
   - Inconvénients : Peut être plus lent sur des machines avec peu de ressources, qualité d'embedding potentiellement inférieure à OpenAI

Choisissez la méthode qui convient le mieux à vos besoins en fonction de vos contraintes (connexion internet, budget, performances requises).

## Interface utilisateur

L'application dispose d'une interface utilisateur intuitive accessible à l'adresse http://localhost:8000 qui permet de :

1. **Télécharger les fichiers d'entrée** :
   - Fichier CSV des mots-clés
   - Fichier CSV des URLs avec leur contenu et typologie (optionnelle)

2. **Configurer les paramètres** :
   - Méthode d'embedding (OpenAI ou Sentence Transformers)
   - Algorithme de clustering (KMeans ou DBSCAN)
   - Paramètres spécifiques à l'algorithme choisi

3. **Visualiser les résultats** :
   - Aperçu des clusters de mots-clés
   - URLs associées à chaque cluster
   - Scores de similarité
   - Types de pages (si spécifiés)

4. **Filtrer les résultats** :
   - Par cluster
   - Par URL
   - Par type de page
   - Par score de similarité minimum

5. **Télécharger les résultats** :
   - Téléchargement des résultats complets
   - Téléchargement des résultats filtrés
   - Format CSV pour une analyse ultérieure

## Exemples d'utilisation

### Avec cURL

```bash
curl -X 'POST' \
  'http://localhost:8000/cluster-keywords/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'keywords_file=@keywords.csv' \
  -F 'urls_file=@urls.csv' \
  -F 'api_key=votre-cle-api-openai' \
  -F 'clustering_algorithm=kmeans' \
  -F 'n_clusters=5'
```

### Avec Python Requests

```python
import requests

url = "http://localhost:8000/cluster-keywords/"

files = {
    'keywords_file': open('keywords.csv', 'rb'),
    'urls_file': open('urls.csv', 'rb')
}

data = {
    'api_key': 'votre-cle-api-openai',
    'clustering_algorithm': 'kmeans',
    'n_clusters': 5
}

response = requests.post(url, files=files, data=data)

with open('results.csv', 'wb') as f:
    f.write(response.content)
```

## Fonctionnement technique

1. **Génération d'embeddings** : Les mots-clés et le contenu des pages sont convertis en vecteurs numériques via l'API OpenAI ou Sentence Transformers
2. **Clustering** : Les mots-clés sont regroupés en utilisant KMeans ou DBSCAN en fonction de la similarité de leurs embeddings
3. **Classification par typologie** : Les URLs sont classifiées selon leur typologie de page (si spécifiée dans le fichier d'entrée)
4. **Association aux URLs** : Pour chaque cluster, l'API trouve les URLs les plus pertinentes dont le contenu a l'embedding le plus proche du centroïde du cluster
5. **Filtrage des résultats** : Possibilité de filtrer les résultats par cluster, URL, typologie de page et score de similarité
6. **Génération du résultat** : Un fichier CSV est créé avec les mots-clés, leur cluster, les URLs associées, les scores de similarité et les typologies de pages

## Licence

MIT