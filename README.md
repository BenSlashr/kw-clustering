# Keyword Clustering Tool

Un outil de clustering de mots-clés basé sur OpenAI qui associe des clusters à des URLs pertinentes.

## Description

Cette API FastAPI permet de :
- Clusteriser une liste de mots-clés en groupes sémantiques
- Associer chaque cluster à l'URL la plus pertinente
- Générer un fichier CSV avec les résultats

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
   - Séparateurs acceptés : virgule (,) ou point-virgule (;)

   Exemple avec virgule :
   ```
   url,contenu
   https://example.com/seo,Contenu sur le référencement naturel et l'optimisation pour les moteurs de recherche
   https://example.com/social,Contenu sur les réseaux sociaux et le marketing digital
   ```
   
   Exemple avec point-virgule :
   ```
   url;contenu
   https://example.com/seo;Contenu sur le référencement naturel et l'optimisation pour les moteurs de recherche
   https://example.com/social;Contenu sur les réseaux sociaux et le marketing digital
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

1. **Génération d'embeddings** : Les mots-clés et le contenu des pages sont convertis en vecteurs numériques via l'API OpenAI
2. **Clustering** : Les mots-clés sont regroupés en utilisant KMeans ou DBSCAN en fonction de la similarité de leurs embeddings
3. **Association aux URLs** : Pour chaque cluster, l'API trouve l'URL dont le contenu a l'embedding le plus proche du centroïde du cluster
4. **Génération du résultat** : Un fichier CSV est créé avec les mots-clés, leur cluster et l'URL associée

## Licence

MIT