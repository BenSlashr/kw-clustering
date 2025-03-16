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
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a list to store logs that will be accessible from the frontend
frontend_logs = []

def add_frontend_log(message, level="info"):
    """Add a log message that will be accessible from the frontend"""
    timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
    log_entry = {"timestamp": timestamp, "message": message, "level": level}
    frontend_logs.append(log_entry)
    # Also log to the standard logger
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    return log_entry

# Clear frontend logs
def clear_frontend_logs():
    frontend_logs.clear()

# Get all frontend logs
def get_logs():
    return {"logs": frontend_logs}

# Initialize FastAPI app
app = FastAPI(
    title="Keyword Clustering API",
    description="API for clustering keywords and associating them with URLs",
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

# Initialize Sentence Transformer model
def get_sentence_transformer(model_name="all-MiniLM-L6-v2"):
    """Initialize a Sentence Transformer model"""
    try:
        logger.info(f"Loading Sentence Transformer model: {model_name}")
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        logger.error(f"Error loading Sentence Transformer model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading Sentence Transformer model: {str(e)}")

# Function to generate embeddings using OpenAI
def generate_openai_embeddings(texts, openai_client):
    """Generate embeddings for a list of texts using OpenAI API"""
    try:
        logger.info(f"Generating OpenAI embeddings for {len(texts)} texts")
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
        logger.error(f"Error generating OpenAI embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating OpenAI embeddings: {str(e)}")

# Function to generate embeddings using Sentence Transformers
def generate_sentence_transformer_embeddings(texts, model):
    """Generate embeddings for a list of texts using Sentence Transformers"""
    try:
        add_frontend_log(f"Generating Sentence Transformer embeddings for {len(texts)} texts", "info")
        
        # Log the first few texts for debugging
        if texts:
            sample_text = texts[0][:100] + "..." if len(texts[0]) > 100 else texts[0]
            add_frontend_log(f"Sample text: {sample_text}", "info")
        
        # NOUVELLE APPROCHE: Traiter chaque texte individuellement pour éviter les problèmes
        add_frontend_log("Starting encoding with Sentence Transformer (safe mode)", "info")
        
        all_embeddings = []
        for i, text in enumerate(texts):
            try:
                # Encode chaque texte individuellement
                single_embedding = model.encode(text)
                
                # Vérifier si le résultat est un scalaire ou un vecteur
                if np.isscalar(single_embedding):
                    # Si c'est un scalaire, le convertir en liste
                    single_embedding = [float(single_embedding)]
                elif isinstance(single_embedding, np.ndarray):
                    # Si c'est un tableau numpy, le convertir en liste
                    single_embedding = single_embedding.tolist()
                
                all_embeddings.append(single_embedding)
                
                # Afficher la progression
                if (i+1) % 100 == 0 or i+1 == len(texts):
                    add_frontend_log(f"Processed {i+1}/{len(texts)} texts", "info")
                    
            except Exception as e:
                add_frontend_log(f"Error encoding text {i}: {str(e)}", "error")
                # Utiliser un vecteur de zéros comme fallback
                if i > 0 and all_embeddings:
                    # Utiliser la même dimension que le premier embedding réussi
                    dim = len(all_embeddings[0])
                    all_embeddings.append([0.0] * dim)
                else:
                    # Si c'est le premier texte qui échoue, utiliser une dimension par défaut
                    all_embeddings.append([0.0] * 384)  # Dimension par défaut pour all-MiniLM-L6-v2
        
        add_frontend_log(f"Encoding complete. Generated {len(all_embeddings)} embeddings", "info")
        
        # Vérifier la cohérence des dimensions
        dimensions = [len(emb) for emb in all_embeddings]
        if len(set(dimensions)) > 1:
            add_frontend_log(f"WARNING: Inconsistent embedding dimensions detected: {set(dimensions)}", "warning")
            # Normaliser les dimensions en utilisant la dimension la plus fréquente
            most_common_dim = max(set(dimensions), key=dimensions.count)
            add_frontend_log(f"Normalizing all embeddings to dimension {most_common_dim}", "info")
            
            for i, emb in enumerate(all_embeddings):
                if len(emb) != most_common_dim:
                    # Remplir avec des zéros ou tronquer selon le besoin
                    if len(emb) < most_common_dim:
                        all_embeddings[i] = emb + [0.0] * (most_common_dim - len(emb))
                    else:
                        all_embeddings[i] = emb[:most_common_dim]
        
        # Log embedding details
        add_frontend_log(f"Final embeddings count: {len(all_embeddings)}", "info")
        if all_embeddings:
            add_frontend_log(f"First embedding type: {type(all_embeddings[0])}", "info")
            add_frontend_log(f"First embedding length: {len(all_embeddings[0])}", "info")
            
        return all_embeddings
    except Exception as e:
        error_msg = f"Error generating Sentence Transformer embeddings: {str(e)}"
        add_frontend_log(error_msg, "error")
        add_frontend_log(f"Error occurred at line: {e.__traceback__.tb_lineno}", "error")
        raise HTTPException(status_code=500, detail=error_msg)

# Function to generate embeddings based on selected method
def generate_embeddings(texts, embedding_method, api_key=None, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings using the selected method"""
    if embedding_method.lower() == "openai":
        if not api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key is required for OpenAI embeddings")
        openai_client = get_openai_client(api_key)
        return generate_openai_embeddings(texts, openai_client)
    elif embedding_method.lower() == "sentence-transformers":
        model = get_sentence_transformer(model_name)
        return generate_sentence_transformer_embeddings(texts, model)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported embedding method: {embedding_method}")

# Function to perform clustering
def perform_clustering(embeddings, algorithm="kmeans", n_clusters=None, eps=None, min_samples=None):
    """Cluster embeddings using either KMeans or DBSCAN"""
    try:
        add_frontend_log(f"Performing clustering using {algorithm}", "info")
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
        
        add_frontend_log("Clustering completed successfully", "info")
        return clusters
    except Exception as e:
        error_msg = f"Error performing clustering: {str(e)}"
        add_frontend_log(error_msg, "error")
        add_frontend_log(f"Error occurred at line: {e.__traceback__.tb_lineno}", "error")
        raise HTTPException(status_code=500, detail=error_msg)

# Function to find the most relevant URL for each cluster
def associate_clusters_with_urls(keyword_embeddings, url_embeddings, clusters, urls, page_types=None, top_n=3):
    """Associate each cluster with the most relevant URLs"""
    try:
        add_frontend_log("Associating clusters with URLs", "info")
        cluster_to_urls = {}
        
        # Log the shapes and types for debugging
        add_frontend_log(f"Keyword embeddings type: {type(keyword_embeddings)}, length: {len(keyword_embeddings)}", "info")
        add_frontend_log(f"URL embeddings type: {type(url_embeddings)}, length: {len(url_embeddings)}", "info")
        
        # Ensure keyword_embeddings is a list of lists
        if isinstance(keyword_embeddings[0], float):
            add_frontend_log("Converting keyword embeddings from 1D to 2D", "info")
            keyword_embeddings = [[e] for e in keyword_embeddings]
        
        # Ensure url_embeddings is a list of lists
        if isinstance(url_embeddings[0], float):
            add_frontend_log("Converting URL embeddings from 1D to 2D", "info")
            url_embeddings = [[e] for e in url_embeddings]
        
        # Check if page types are provided
        has_page_types = page_types is not None and len(page_types) == len(urls)
        if has_page_types:
            add_frontend_log("Page types will be included in the results", "info")
        else:
            add_frontend_log("No page types provided or length mismatch. Proceeding without page type classification.", "info")
            page_types = ["unknown"] * len(urls)
        
        # For each unique cluster
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Skip noise points from DBSCAN
                continue
                
            # Get indices of keywords in this cluster
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            
            # Get the centroid of the cluster
            try:
                # Convert to numpy arrays for calculation
                cluster_vectors = np.array([keyword_embeddings[i] for i in cluster_indices])
                cluster_centroid = np.mean(cluster_vectors, axis=0)
                add_frontend_log(f"Calculated centroid for cluster {cluster_id}, shape: {cluster_centroid.shape}", "info")
            except Exception as e:
                add_frontend_log(f"Error calculating centroid: {str(e)}", "error")
                # Fallback: just use the first vector in the cluster
                cluster_centroid = np.array(keyword_embeddings[cluster_indices[0]])
                add_frontend_log(f"Using fallback centroid with shape: {cluster_centroid.shape}", "warning")
            
            # Ensure both inputs to cosine_similarity are 2D arrays
            if len(np.array(cluster_centroid).shape) == 1:
                cluster_centroid = np.array([cluster_centroid])
                add_frontend_log(f"Reshaped centroid to 2D: {cluster_centroid.shape}", "info")
                
            # Convert url_embeddings to numpy array
            url_embeddings_array = np.array(url_embeddings)
            add_frontend_log(f"URL embeddings array shape: {url_embeddings_array.shape}", "info")
            
            # Ensure url_embeddings is 2D
            if len(url_embeddings_array.shape) == 1:
                url_embeddings_array = url_embeddings_array.reshape(1, -1)
                add_frontend_log(f"Reshaped URL embeddings to 2D: {url_embeddings_array.shape}", "info")
            
            # Calculate similarity between cluster centroid and each URL
            try:
                similarities = cosine_similarity(cluster_centroid, url_embeddings_array)[0]
                add_frontend_log(f"Calculated similarities for cluster {cluster_id}, shape: {similarities.shape}", "info")
            except Exception as e:
                add_frontend_log(f"Error calculating similarities: {str(e)}", "error")
                add_frontend_log(f"Centroid shape: {cluster_centroid.shape}, URL embeddings shape: {url_embeddings_array.shape}", "error")
                raise
            
            # Get the top N URLs with highest similarity
            top_indices = np.argsort(similarities)[::-1][:top_n]
            top_similarities = similarities[top_indices]
            
            # Store the top URLs and their similarity scores
            cluster_to_urls[cluster_id] = [
                {"url": urls[idx], "similarity": float(score), "page_type": page_types[idx]} 
                for idx, score in zip(top_indices, top_similarities)
            ]
            
            add_frontend_log(f"Cluster {cluster_id} associated with top URL: {urls[top_indices[0]][:50]}... (score: {top_similarities[0]:.4f})", "info")
        
        add_frontend_log("Clusters associated with URLs successfully", "info")
        return cluster_to_urls
    except Exception as e:
        error_msg = f"Error associating clusters with URLs: {str(e)}"
        add_frontend_log(error_msg, "error")
        add_frontend_log(f"Error occurred at line: {e.__traceback__.tb_lineno}", "error")
        raise HTTPException(status_code=500, detail=error_msg)

# Function to process the clustering task
def process_clustering_task(
    keywords_file_path,
    urls_file_path,
    output_file_path,
    embedding_method,
    api_key,
    st_model_name,
    clustering_algorithm,
    n_clusters,
    eps,
    min_samples
):
    try:
        # Clear previous logs
        clear_frontend_logs()
        add_frontend_log("Starting clustering task", "info")
        
        # Read input files with auto-detection of separator (comma or semicolon)
        add_frontend_log("Reading keywords file", "info")
        keywords_df = None
        urls_df = None
        
        # Try different encodings and separators for keywords file
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            for sep in [',', ';']:
                try:
                    keywords_df = pd.read_csv(keywords_file_path, sep=sep, encoding=encoding)
                    if not keywords_df.empty:
                        add_frontend_log(f"Successfully read keywords file with encoding {encoding} and separator '{sep}'", "info")
                        break
                except Exception as e:
                    add_frontend_log(f"Failed to read keywords file with encoding {encoding} and separator '{sep}': {str(e)}", "warning")
            if keywords_df is not None and not keywords_df.empty:
                break
        
        if keywords_df is None or keywords_df.empty:
            error_msg = "Could not read keywords CSV file with any encoding or separator"
            add_frontend_log(error_msg, "error")
            raise ValueError(error_msg)
        
        # Try different encodings and separators for URLs file
        add_frontend_log("Reading URLs file", "info")
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            for sep in [',', ';']:
                try:
                    urls_df = pd.read_csv(urls_file_path, sep=sep, encoding=encoding)
                    if not urls_df.empty and len(urls_df.columns) >= 2:
                        add_frontend_log(f"Successfully read URLs file with encoding {encoding} and separator '{sep}'", "info")
                        break
                except Exception as e:
                    add_frontend_log(f"Failed to read URLs file with encoding {encoding} and separator '{sep}': {str(e)}", "warning")
            if urls_df is not None and not urls_df.empty and len(urls_df.columns) >= 2:
                break
        
        # If we couldn't read the URLs file with standard methods, try a more flexible approach
        if urls_df is None or urls_df.empty or len(urls_df.columns) < 2:
            add_frontend_log("Standard CSV parsing failed. Trying custom parsing for URLs file.", "warning")
            try:
                # Read the file as text
                with open(urls_file_path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
                
                # Parse manually
                urls = []
                contents = []
                page_types = []
                for line in lines:
                    if line.strip() and ';' in line:
                        parts = line.split(';')
                        url = parts[0].strip()
                        # Join all other parts as content, ignoring empty trailing parts
                        content = ';'.join([p for p in parts[1:] if p.strip()])
                        if url and content:
                            urls.append(url)
                            contents.append(content)
                            # Try to extract page type from the content
                            page_type = None
                            for keyword in ['typologie', 'type']:
                                if keyword in content.lower():
                                    page_type = content.split(keyword)[1].strip().split(';')[0].strip()
                                    break
                            page_types.append(page_type if page_type else "unknown")
                
                if urls and contents:
                    urls_df = pd.DataFrame({'url': urls, 'content': contents, 'page_type': page_types})
                    add_frontend_log(f"Successfully parsed URLs file manually, found {len(urls)} entries", "info")
                else:
                    error_msg = "No valid URL and content pairs found in the file"
                    add_frontend_log(error_msg, "error")
                    raise ValueError(error_msg)
            except Exception as e:
                error_msg = f"Custom parsing failed: {str(e)}"
                add_frontend_log(error_msg, "error")
                raise ValueError(f"Could not parse URLs CSV file: {str(e)}")
        
        # Validate input data
        if len(keywords_df.columns) < 1:
            error_msg = "Keywords CSV must have at least one column"
            add_frontend_log(error_msg, "error")
            raise ValueError(error_msg)
        
        if len(urls_df.columns) < 2:
            error_msg = "URLs CSV must have at least two columns (URL and content)"
            add_frontend_log(error_msg, "error")
            raise ValueError(error_msg)
            
        # Check if page type column exists
        has_page_type = False
        page_type_column = None
        for col in urls_df.columns:
            if col.lower() in ['page_type', 'typologie', 'type']:
                has_page_type = True
                page_type_column = col
                add_frontend_log(f"Found page type column: {col}", "info")
                break
                
        # If page type column exists, ensure it's properly formatted
        if has_page_type:
            # Fill missing values with 'unknown'
            urls_df[page_type_column] = urls_df[page_type_column].fillna('unknown')
            add_frontend_log(f"Page types found: {urls_df[page_type_column].unique().tolist()}", "info")
        else:
            add_frontend_log("No page type column found in URLs file. Proceeding without page type classification.", "info")
        
        # Extract keywords, URLs, and content
        keywords = keywords_df.iloc[:, 0].tolist()
        urls = urls_df.iloc[:, 0].tolist()
        url_contents = urls_df.iloc[:, 1].tolist()
        page_types = urls_df[page_type_column].tolist() if has_page_type else None
        
        add_frontend_log(f"Extracted {len(keywords)} keywords and {len(urls)} URLs", "info")
        
        # Generate embeddings based on selected method
        add_frontend_log(f"Using embedding method: {embedding_method}", "info")
        
        add_frontend_log("Generating keyword embeddings", "info")
        keyword_embeddings = generate_embeddings(
            keywords, 
            embedding_method, 
            api_key, 
            st_model_name
        )
        add_frontend_log("Keyword embeddings generated successfully", "info")
        
        add_frontend_log("Generating URL content embeddings", "info")
        url_content_embeddings = generate_embeddings(
            url_contents, 
            embedding_method, 
            api_key, 
            st_model_name
        )
        add_frontend_log("URL content embeddings generated successfully", "info")
        
        # Perform clustering
        add_frontend_log(f"Performing clustering using {clustering_algorithm}", "info")
        clusters = perform_clustering(
            keyword_embeddings, 
            algorithm=clustering_algorithm,
            n_clusters=n_clusters,
            eps=eps,
            min_samples=min_samples
        )
        add_frontend_log("Clustering completed successfully", "info")
        
        # Associate clusters with URLs
        add_frontend_log("Associating clusters with URLs", "info")
        cluster_to_urls = associate_clusters_with_urls(
            keyword_embeddings,
            url_content_embeddings,
            clusters,
            urls,
            page_types
        )
        add_frontend_log("Clusters associated with URLs successfully", "info")
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'Keyword': keywords,
            'Cluster': clusters
        })
        
        # Add URL and similarity columns for top URLs
        top_n = 3
        for i in range(top_n):
            result_df[f'URL{i+1}'] = result_df['Cluster'].map(
                lambda x: cluster_to_urls.get(x, [])[i]['url'] 
                if x != -1 and len(cluster_to_urls.get(x, [])) > i else ""
            )
            result_df[f'Similarity{i+1}'] = result_df['Cluster'].map(
                lambda x: cluster_to_urls.get(x, [])[i]['similarity'] 
                if x != -1 and len(cluster_to_urls.get(x, [])) > i else 0.0
            )
            result_df[f'PageType{i+1}'] = result_df['Cluster'].map(
                lambda x: cluster_to_urls.get(x, [])[i]['page_type'] 
                if x != -1 and len(cluster_to_urls.get(x, [])) > i else ""
            )
        
        # Save to CSV
        result_df.to_csv(output_file_path, index=False)
        add_frontend_log(f"Results saved to {output_file_path}", "info")
        
        return output_file_path
    except Exception as e:
        error_msg = f"Error processing clustering task: {str(e)}"
        add_frontend_log(error_msg, "error")
        add_frontend_log(f"Error occurred at line: {e.__traceback__.tb_lineno}", "error")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Serve the HTML interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/logs")
def get_logs():
    return {"logs": frontend_logs}

@app.post("/api/logs/clear")
def clear_logs():
    clear_frontend_logs()
    return {"status": "success", "message": "Logs cleared"}

@app.post("/cluster-keywords/", summary="Cluster keywords and associate them with URLs")
async def cluster_keywords(
    background_tasks: BackgroundTasks,
    keywords_file: UploadFile = File(..., description="CSV file with keywords"),
    urls_file: UploadFile = File(..., description="CSV file with URLs and their content"),
    embedding_method: str = Form("openai", description="Embedding method: 'openai' or 'sentence-transformers'"),
    api_key: Optional[str] = Form(None, description="OpenAI API key (required for OpenAI embeddings)"),
    st_model_name: str = Form("all-MiniLM-L6-v2", description="Sentence Transformers model name"),
    clustering_algorithm: str = Form("kmeans", description="Clustering algorithm: 'kmeans' or 'dbscan'"),
    n_clusters: Optional[int] = Form(None, description="Number of clusters (for KMeans)"),
    eps: Optional[float] = Form(None, description="Epsilon parameter (for DBSCAN)"),
    min_samples: Optional[int] = Form(None, description="Min samples parameter (for DBSCAN)")
):
    # Validate embedding method
    if embedding_method.lower() == "openai" and not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key is required when using OpenAI embeddings")
    
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
            embedding_method,
            api_key,
            st_model_name,
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
