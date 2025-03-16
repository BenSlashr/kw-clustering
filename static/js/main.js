document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const form = document.getElementById('clustering-form');
    const keywordsFileInput = document.getElementById('keywords-file');
    const urlsFileInput = document.getElementById('urls-file');
    const apiKeyInput = document.getElementById('api-key');
    const algorithmSelect = document.getElementById('algorithm');
    const nClustersInput = document.getElementById('n-clusters');
    const epsInput = document.getElementById('eps');
    const minSamplesInput = document.getElementById('min-samples');
    const keywordsFileName = document.getElementById('keywords-filename');
    const urlsFileName = document.getElementById('urls-filename');
    const loadingIndicator = document.getElementById('loading-indicator');
    const resultSection = document.getElementById('result-section');
    const downloadBtn = document.getElementById('download-btn');
    const alertContainer = document.getElementById('alert-container');
    const dbscanParams = document.getElementById('dbscan-params');
    const kmeansParams = document.getElementById('kmeans-params');
    const previewContainer = document.getElementById('preview-container');
    const previewTable = document.getElementById('preview-table');
    const previewBody = document.getElementById('preview-body');
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Local storage key for API key
    const API_KEY_STORAGE = 'openai_api_key';
    
    // Check if API key is stored in local storage
    if (localStorage.getItem(API_KEY_STORAGE)) {
        apiKeyInput.value = localStorage.getItem(API_KEY_STORAGE);
    }
    
    // File input change handlers
    keywordsFileInput.addEventListener('change', (e) => {
        const fileName = e.target.files[0]?.name || 'Aucun fichier choisi';
        keywordsFileName.textContent = fileName;
        
        // Preview keywords file if it's a CSV
        if (e.target.files[0] && e.target.files[0].type === 'text/csv') {
            previewCSV(e.target.files[0], 'keywords');
        }
    });
    
    urlsFileInput.addEventListener('change', (e) => {
        const fileName = e.target.files[0]?.name || 'Aucun fichier choisi';
        urlsFileName.textContent = fileName;
        
        // Preview URLs file if it's a CSV
        if (e.target.files[0] && e.target.files[0].type === 'text/csv') {
            previewCSV(e.target.files[0], 'urls');
        }
    });
    
    // Algorithm change handler
    algorithmSelect.addEventListener('change', (e) => {
        if (e.target.value === 'kmeans') {
            kmeansParams.classList.remove('hidden');
            dbscanParams.classList.add('hidden');
        } else {
            kmeansParams.classList.add('hidden');
            dbscanParams.classList.remove('hidden');
        }
    });
    
    // Handle embedding method change
    const embeddingMethodSelect = document.getElementById('embedding-method');
    const openaiParams = document.getElementById('openai-params');
    const stParams = document.getElementById('st-params');

    embeddingMethodSelect.addEventListener('change', function() {
        if (this.value === 'openai') {
            openaiParams.classList.remove('hidden');
            stParams.classList.add('hidden');
        } else if (this.value === 'sentence-transformers') {
            openaiParams.classList.add('hidden');
            stParams.classList.remove('hidden');
        }
    });

    // Tab navigation
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked tab and corresponding content
            tab.classList.add('active');
            const tabId = tab.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
    
    // Form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Validate inputs
        const keywordsFile = document.getElementById('keywords-file').files[0];
        const urlsFile = document.getElementById('urls-file').files[0];
        const embeddingMethod = document.getElementById('embedding-method').value;
        const apiKey = document.getElementById('api-key').value;
        const stModel = document.getElementById('st-model').value;
        const algorithm = document.getElementById('algorithm').value;
        const nClusters = document.getElementById('n-clusters').value;
        const eps = document.getElementById('eps').value;
        const minSamples = document.getElementById('min-samples').value;
        
        // Validate required fields
        if (!keywordsFile) {
            showAlert('Veuillez sélectionner un fichier de mots-clés', 'danger');
            return;
        }
        
        if (!urlsFile) {
            showAlert('Veuillez sélectionner un fichier d\'URLs', 'danger');
            return;
        }
        
        if (embeddingMethod === 'openai' && !apiKey) {
            showAlert('Veuillez fournir une clé API OpenAI', 'danger');
            return;
        }
        
        // Save API key to local storage if checkbox is checked
        const saveApiKey = document.getElementById('save-api-key').checked;
        if (saveApiKey && apiKey) {
            localStorage.setItem('openai_api_key', apiKey);
        }
        
        // Show loading state
        loadingIndicator.classList.remove('hidden');
        resultSection.classList.add('hidden');
        clearAlert();
        downloadBtn.classList.add('hidden');
        
        // Prepare form data
        const formData = new FormData();
        formData.append('keywords_file', keywordsFile);
        formData.append('urls_file', urlsFile);
        formData.append('embedding_method', embeddingMethod);
        
        if (embeddingMethod === 'openai') {
            formData.append('api_key', apiKey);
        } else if (embeddingMethod === 'sentence-transformers') {
            formData.append('st_model_name', stModel);
        }
        
        formData.append('clustering_algorithm', algorithm);
        
        if (nClusters) {
            formData.append('n_clusters', nClusters);
        }
        
        if (eps) {
            formData.append('eps', eps);
        }
        
        if (minSamples) {
            formData.append('min_samples', minSamples);
        }
        
        try {
            // Send request to API
            const response = await fetch('/cluster-keywords/', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Erreur: ${response.status} - ${await response.text()}`);
            }
            
            // Get response as blob
            const blob = await response.blob();
            
            // Create object URL for download
            const url = window.URL.createObjectURL(blob);
            downloadBtn.href = url;
            downloadBtn.download = 'clustered_keywords.csv';
            downloadBtn.classList.remove('hidden');
            
            // Read and display preview of results
            const csvText = await blob.text();
            displayResultPreview(csvText);
            
            // Show result section
            resultSection.classList.remove('hidden');
            showAlert('Clustering terminé avec succès !', 'success');
        } catch (error) {
            showAlert(`Erreur lors du clustering: ${error.message}`, 'danger');
            console.error('Error:', error);
        } finally {
            loadingIndicator.classList.add('hidden');
        }
    });
    
    // Function to validate form
    function validateForm() {
        if (!keywordsFileInput.files[0]) {
            showAlert('Veuillez sélectionner un fichier de mots-clés', 'danger');
            return false;
        }
        
        if (!urlsFileInput.files[0]) {
            showAlert('Veuillez sélectionner un fichier d\'URLs', 'danger');
            return false;
        }
        
        if (!apiKeyInput.value) {
            showAlert('Veuillez entrer une clé API OpenAI', 'danger');
            return false;
        }
        
        return true;
    }
    
    // Function to display alerts
    function showAlert(message, type) {
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.textContent = message;
        
        alertContainer.innerHTML = '';
        alertContainer.appendChild(alert);
        
        // Scroll to alert
        alertContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Function to clear alerts
    function clearAlert() {
        alertContainer.innerHTML = '';
    }
    
    // Function to preview CSV files
    function previewCSV(file, type) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            const content = e.target.result;
            const lines = content.split('\\n');
            
            if (lines.length > 0) {
                // Clear previous preview
                previewBody.innerHTML = '';
                
                // Get headers
                const headers = lines[0].split(',');
                
                // Create header row
                const headerRow = document.createElement('tr');
                headers.forEach(header => {
                    const th = document.createElement('th');
                    th.textContent = header;
                    headerRow.appendChild(th);
                });
                
                // Add header row to table
                document.getElementById('preview-header').innerHTML = '';
                document.getElementById('preview-header').appendChild(headerRow);
                
                // Add data rows (limit to 10)
                const maxRows = Math.min(lines.length, 11);
                for (let i = 1; i < maxRows; i++) {
                    if (lines[i].trim() === '') continue;
                    
                    const rowData = lines[i].split(',');
                    const row = document.createElement('tr');
                    
                    rowData.forEach(cell => {
                        const td = document.createElement('td');
                        td.textContent = cell;
                        row.appendChild(td);
                    });
                    
                    previewBody.appendChild(row);
                }
                
                // Show preview
                previewContainer.classList.remove('hidden');
                document.getElementById('preview-title').textContent = 
                    type === 'keywords' ? 'Aperçu des mots-clés' : 'Aperçu des URLs';
                
                // Switch to preview tab
                tabs.forEach(t => t.classList.remove('active'));
                tabContents.forEach(c => c.classList.remove('active'));
                document.querySelector('[data-tab="preview-tab"]').classList.add('active');
                document.getElementById('preview-tab').classList.add('active');
            }
        };
        
        reader.readAsText(file);
    }
    
    // Function to display result preview
    function displayResultPreview(csvText) {
        const lines = csvText.split('\\n');
        
        if (lines.length > 0) {
            // Clear previous preview
            previewBody.innerHTML = '';
            
            // Get headers
            const headers = lines[0].split(',');
            
            // Create header row
            const headerRow = document.createElement('tr');
            headers.forEach(header => {
                const th = document.createElement('th');
                th.textContent = header;
                headerRow.appendChild(th);
            });
            
            // Add header row to table
            document.getElementById('preview-header').innerHTML = '';
            document.getElementById('preview-header').appendChild(headerRow);
            
            // Add data rows (limit to 10)
            const maxRows = Math.min(lines.length, 11);
            for (let i = 1; i < maxRows; i++) {
                if (lines[i].trim() === '') continue;
                
                const rowData = lines[i].split(',');
                const row = document.createElement('tr');
                
                rowData.forEach(cell => {
                    const td = document.createElement('td');
                    td.textContent = cell;
                    row.appendChild(td);
                });
                
                previewBody.appendChild(row);
            }
            
            // Show preview
            previewContainer.classList.remove('hidden');
            document.getElementById('preview-title').textContent = 'Aperçu des résultats';
            
            // Switch to preview tab
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            document.querySelector('[data-tab="preview-tab"]').classList.add('active');
            document.getElementById('preview-tab').classList.add('active');
        }
    }
    
    // Function to clear form
    document.getElementById('reset-btn').addEventListener('click', () => {
        form.reset();
        keywordsFileName.textContent = 'Aucun fichier choisi';
        urlsFileName.textContent = 'Aucun fichier choisi';
        clearAlert();
        previewContainer.classList.add('hidden');
        resultSection.classList.add('hidden');
    });
});
