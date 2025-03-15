import requests
import os
import argparse
from dotenv import load_dotenv

# Load environment variables (for API key)
load_dotenv()

def test_clustering_api(
    api_url,
    keywords_file,
    urls_file,
    api_key,
    clustering_algorithm="kmeans",
    n_clusters=None
):
    """
    Test the keyword clustering API with provided files
    """
    # Prepare the API request
    files = {
        'keywords_file': open(keywords_file, 'rb'),
        'urls_file': open(urls_file, 'rb')
    }
    
    data = {
        'api_key': api_key,
        'clustering_algorithm': clustering_algorithm
    }
    
    # Add optional parameters if provided
    if n_clusters:
        data['n_clusters'] = n_clusters
    
    print(f"Sending request to {api_url}...")
    print(f"Files: {keywords_file}, {urls_file}")
    print(f"Algorithm: {clustering_algorithm}")
    
    # Send the request
    try:
        response = requests.post(api_url, files=files, data=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Save the response content to a file
            output_file = "clustered_keywords_result.csv"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"Success! Results saved to {output_file}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
    finally:
        # Close the file handlers
        for f in files.values():
            f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the keyword clustering API")
    parser.add_argument("--api-url", default="http://localhost:8000/cluster-keywords/", help="API endpoint URL")
    parser.add_argument("--keywords-file", default="examples/keywords.csv", help="Path to keywords CSV file")
    parser.add_argument("--urls-file", default="examples/urls.csv", help="Path to URLs CSV file")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--algorithm", default="kmeans", choices=["kmeans", "dbscan"], help="Clustering algorithm")
    parser.add_argument("--n-clusters", type=int, help="Number of clusters (for KMeans)")
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key:
        print("Error: OpenAI API key is required. Provide it with --api-key or set OPENAI_API_KEY environment variable.")
        exit(1)
    
    test_clustering_api(
        args.api_url,
        args.keywords_file,
        args.urls_file,
        args.api_key,
        args.algorithm,
        args.n_clusters
    )
