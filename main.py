import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import openai
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
import tempfile
import uuid
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Keyword Clustering API",
    description="API for clustering keywords and associating them with relevant URLs",
    version="1.0.0"
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize OpenAI client
def get_openai_client(api_key):
    openai.api_key = api_key
    return openai

# Function to generate embeddings using OpenAI
def generate_embeddings(texts, openai_client):
    """Generate embeddings for a list of texts using OpenAI API"""
    try:
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = []
        # Process in batches to avoid API limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = openai_client.Embedding.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            batch_embeddings = [item["embedding"] for item in response["data"]]
            embeddings.extend(batch_embeddings)
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

# Function to perform clustering
def perform_clustering(embeddings, algorithm="kmeans", n_clusters=None, eps=None, min_samples=None):
    """Cluster embeddings using either KMeans or DBSCAN"""
    try:
        logger.info(f"Performing clustering using {algorithm}")
        if algorithm.lower() == "kmeans":
            if n_clusters is None:
                # Estimate number of clusters if not provided
                n_clusters = max(2, min(20, len(embeddings) // 10))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
        elif algorithm.lower() == "dbscan":
            if eps is None:
                eps = 0.5
            if min_samples is None:
                min_samples = 5
                
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(embeddings)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
        
        return clusters
    except Exception as e:
        logger.error(f"Error performing clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error performing clustering: {str(e)}")

# Function to find the most relevant URL for each cluster
def associate_clusters_with_urls(keyword_embeddings, url_embeddings, clusters, urls):
    """Associate each cluster with the most relevant URL"""
    try:
        logger.info("Associating clusters with URLs")
        cluster_to_url = {}
        
        # For each unique cluster
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Skip noise points from DBSCAN
                continue
                
            # Get indices of keywords in this cluster
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            
            # Get the centroid of the cluster
            cluster_centroid = np.mean([keyword_embeddings[i] for i in cluster_indices], axis=0)
            
            # Calculate similarity between cluster centroid and each URL
            similarities = cosine_similarity([cluster_centroid], url_embeddings)[0]
            
            # Get the URL with highest similarity
            most_similar_url_idx = np.argmax(similarities)
            cluster_to_url[cluster_id] = urls[most_similar_url_idx]
        
        return cluster_to_url
    except Exception as e:
        logger.error(f"Error associating clusters with URLs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error associating clusters with URLs: {str(e)}")

# Function to process the clustering task
def process_clustering_task(
    keywords_file_path,
    urls_file_path,
    output_file_path,
    api_key,
    clustering_algorithm,
    n_clusters,
    eps,
    min_samples
):
    try:
        # Initialize OpenAI client
        openai_client = get_openai_client(api_key)
        
        # Read input files
        keywords_df = pd.read_csv(keywords_file_path)
        urls_df = pd.read_csv(urls_file_path)
        
        # Validate input data
        if len(keywords_df.columns) < 1:
            raise ValueError("Keywords CSV must have at least one column")
        
        if len(urls_df.columns) < 2:
            raise ValueError("URLs CSV must have at least two columns (URL and content)")
        
        # Extract keywords, URLs, and content
        keywords = keywords_df.iloc[:, 0].tolist()
        urls = urls_df.iloc[:, 0].tolist()
        url_contents = urls_df.iloc[:, 1].tolist()
        
        # Generate embeddings
        keyword_embeddings = generate_embeddings(keywords, openai_client)
        url_content_embeddings = generate_embeddings(url_contents, openai_client)
        
        # Perform clustering
        clusters = perform_clustering(
            keyword_embeddings, 
            algorithm=clustering_algorithm,
            n_clusters=n_clusters,
            eps=eps,
            min_samples=min_samples
        )
        
        # Associate clusters with URLs
        cluster_to_url = associate_clusters_with_urls(
            keyword_embeddings,
            url_content_embeddings,
            clusters,
            urls
        )
        
        # Create output DataFrame
        result_df = pd.DataFrame({
            'Keyword': keywords,
            'Cluster': clusters
        })
        
        # Add URL column
        result_df['URL'] = result_df['Cluster'].map(
            lambda x: cluster_to_url.get(x, "No URL associated") if x != -1 else "Noise"
        )
        
        # Save to CSV
        result_df.to_csv(output_file_path, index=False)
        logger.info(f"Results saved to {output_file_path}")
        
        return output_file_path
    except Exception as e:
        logger.error(f"Error processing clustering task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing clustering task: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Serve the HTML interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/cluster-keywords/", summary="Cluster keywords and associate them with URLs")
async def cluster_keywords(
    background_tasks: BackgroundTasks,
    keywords_file: UploadFile = File(..., description="CSV file with keywords"),
    urls_file: UploadFile = File(..., description="CSV file with URLs and their content"),
    api_key: str = Form(..., description="OpenAI API key"),
    clustering_algorithm: str = Form("kmeans", description="Clustering algorithm: 'kmeans' or 'dbscan'"),
    n_clusters: Optional[int] = Form(None, description="Number of clusters (for KMeans)"),
    eps: Optional[float] = Form(None, description="Epsilon parameter (for DBSCAN)"),
    min_samples: Optional[int] = Form(None, description="Min samples parameter (for DBSCAN)")
):
    # Create temporary directory for file processing
    temp_dir = tempfile.mkdtemp()
    
    # Generate unique filenames
    keywords_path = os.path.join(temp_dir, f"keywords_{uuid.uuid4()}.csv")
    urls_path = os.path.join(temp_dir, f"urls_{uuid.uuid4()}.csv")
    output_path = os.path.join(temp_dir, f"results_{uuid.uuid4()}.csv")
    
    # Save uploaded files
    try:
        with open(keywords_path, "wb") as f:
            f.write(await keywords_file.read())
        
        with open(urls_path, "wb") as f:
            f.write(await urls_file.read())
    except Exception as e:
        logger.error(f"Error saving uploaded files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving uploaded files: {str(e)}")
    
    # Process clustering in background
    try:
        result_path = process_clustering_task(
            keywords_path,
            urls_path,
            output_path,
            api_key,
            clustering_algorithm,
            n_clusters,
            eps,
            min_samples
        )
        
        return FileResponse(
            path=result_path,
            filename="clustered_keywords.csv",
            media_type="text/csv"
        )
    except Exception as e:
        logger.error(f"Error in clustering process: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
