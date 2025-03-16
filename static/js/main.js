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
    const logsContent = document.getElementById('logs-content');
    const refreshLogsBtn = document.getElementById('refresh-logs-btn');
    const clearLogsBtn = document.getElementById('clear-logs-btn');
    const downloadFilteredBtn = document.getElementById('download-filtered-btn');
    
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

    // Tab switching
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and content
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked tab and corresponding content
            tab.classList.add('active');
            const tabId = tab.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
            
            // If logs tab is selected, fetch logs
            if (tabId === 'logs-tab') {
                fetchLogs();
            }
        });
    });
    
    // Function to fetch logs from the server
    async function fetchLogs() {
        try {
            const response = await fetch('/api/logs');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            displayLogs(data.logs);
        } catch (error) {
            console.error('Error fetching logs:', error);
            logsContent.innerHTML = `<div class="log-error">Erreur lors de la récupération des logs: ${error.message}</div>`;
        }
    }
    
    // Function to display logs
    function displayLogs(logs) {
        if (!logs || logs.length === 0) {
            logsContent.innerHTML = '<div class="log-info">Aucun log disponible.</div>';
            return;
        }
        
        const logsHtml = logs.map(log => {
            const logClass = log.level === 'error' ? 'log-error' : 
                            log.level === 'warning' ? 'log-warning' : 'log-info';
            return `<div class="${logClass}">
                <span class="log-timestamp">[${log.timestamp}]</span>
                <span class="log-message">${log.message}</span>
            </div>`;
        }).join('');
        
        logsContent.innerHTML = logsHtml;
        
        // Scroll to bottom to show latest logs
        const logsContainer = document.getElementById('logs-container');
        logsContainer.scrollTop = logsContainer.scrollHeight;
    }
    
    // Refresh logs button
    refreshLogsBtn.addEventListener('click', fetchLogs);
    
    // Clear logs button
    clearLogsBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/api/logs/clear', { method: 'POST' });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            logsContent.innerHTML = '<div class="log-info">Logs effacés.</div>';
        } catch (error) {
            console.error('Error clearing logs:', error);
            logsContent.innerHTML = `<div class="log-error">Erreur lors de l'effacement des logs: ${error.message}</div>`;
        }
    });
    
    // Form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Validate inputs
        const keywordsFile = document.getElementById('keywords-file').files[0];
        const urlsFile = document.getElementById('urls-file').files[0];
        const apiKey = document.getElementById('api-key').value;
        const embeddingMethod = document.getElementById('embedding-method').value;
        const stModelName = document.getElementById('st-model-name') ? document.getElementById('st-model-name').value : "all-MiniLM-L6-v2";
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
        
        // Switch to logs tab to show progress
        tabs.forEach(t => t.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        document.querySelector('.tab[data-tab="logs-tab"]').classList.add('active');
        document.getElementById('logs-tab').classList.add('active');
        
        // Clear previous logs
        logsContent.innerHTML = '<div class="log-info">Démarrage du traitement...</div>';
        
        // Start polling for logs
        const logInterval = setInterval(fetchLogs, 1000);
        
        // Prepare form data
        const formData = new FormData();
        formData.append('keywords_file', keywordsFile);
        formData.append('urls_file', urlsFile);
        formData.append('embedding_method', embeddingMethod);
        
        if (embeddingMethod === 'openai') {
            formData.append('api_key', apiKey);
        } else if (embeddingMethod === 'sentence-transformers') {
            formData.append('st_model_name', stModelName);
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
            clearInterval(logInterval);
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
        const lines = csvText.split('\n');
        
        if (lines.length > 0) {
            // Clear previous preview
            previewBody.innerHTML = '';
            
            // Detect separator (comma or semicolon)
            const firstLine = lines[0];
            let separator = ',';
            if (firstLine.includes(';') && (firstLine.split(';').length > firstLine.split(',').length)) {
                separator = ';';
            }
            
            // Get headers
            const headers = firstLine.split(separator);
            
            // Create header row
            const headerRow = document.createElement('tr');
            headers.forEach(header => {
                const th = document.createElement('th');
                th.textContent = header.trim();
                headerRow.appendChild(th);
            });
            
            // Add header row to table
            document.getElementById('preview-header').innerHTML = '';
            document.getElementById('preview-header').appendChild(headerRow);
            
            // Store all rows for filtering
            window.allResultRows = [];
            
            // Add data rows
            for (let i = 1; i < lines.length; i++) {
                if (lines[i].trim() === '') continue;
                
                const rowData = lines[i].split(separator);
                const row = document.createElement('tr');
                
                // Store row data for filtering
                const rowObj = {};
                
                rowData.forEach((cell, index) => {
                    const td = document.createElement('td');
                    td.textContent = cell.trim();
                    row.appendChild(td);
                    
                    // Store cell data with its header
                    if (headers[index]) {
                        rowObj[headers[index].trim()] = cell.trim();
                    }
                });
                
                // Add row to the table
                previewBody.appendChild(row);
                
                // Store row for filtering
                window.allResultRows.push({
                    element: row,
                    data: rowObj
                });
            }
            
            // Initialize filters
            initializeFilters(headers);
            
            // Initialize visible rows for download
            window.visibleRows = window.allResultRows.slice();
            
            // Show download button
            downloadFilteredBtn.style.display = 'inline-block';
            
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
    
    // Function to initialize filters
    function initializeFilters(headers) {
        const clusterFilter = document.getElementById('cluster-filter');
        const urlFilter = document.getElementById('url-filter');
        const pageTypeFilter = document.getElementById('page-type-filter');
        const similarityFilter = document.getElementById('similarity-filter');
        const similarityValue = document.getElementById('similarity-value');
        const applyFiltersBtn = document.getElementById('apply-filters');
        const resetFiltersBtn = document.getElementById('reset-filters');
        
        // Clear existing options
        clusterFilter.innerHTML = '<option value="">Tous les clusters</option>';
        urlFilter.innerHTML = '<option value="">Toutes les URLs</option>';
        pageTypeFilter.innerHTML = '<option value="">Tous les types</option>';
        
        // Get unique clusters, URLs and page types
        const clusters = new Set();
        const urls = new Set();
        const pageTypes = new Set();
        
        window.allResultRows.forEach(row => {
            if (row.data['Cluster']) {
                clusters.add(row.data['Cluster']);
            }
            
            // Add all URLs (primary and alternatives)
            for (let key in row.data) {
                if (key === 'URL' || key.startsWith('URL')) {
                    if (row.data[key] && row.data[key] !== 'Noise' && row.data[key] !== 'No URL associated' && row.data[key] !== '') {
                        urls.add(row.data[key]);
                    }
                }
                // Add all page types
                if (key === 'PageType1' || key === 'PageType2' || key === 'PageType3') {
                    if (row.data[key] && row.data[key] !== '') {
                        pageTypes.add(row.data[key]);
                    }
                }
            }
        });
        
        // Add cluster options
        [...clusters].sort((a, b) => Number(a) - Number(b)).forEach(cluster => {
            const option = document.createElement('option');
            option.value = cluster;
            option.textContent = `Cluster ${cluster}`;
            clusterFilter.appendChild(option);
        });
        
        // Add URL options
        [...urls].sort().forEach(url => {
            const option = document.createElement('option');
            option.value = url;
            option.textContent = url.length > 50 ? url.substring(0, 47) + '...' : url;
            urlFilter.appendChild(option);
        });
        
        // Add page type options
        [...pageTypes].sort().forEach(pageType => {
            const option = document.createElement('option');
            option.value = pageType;
            option.textContent = pageType;
            pageTypeFilter.appendChild(option);
        });
        
        // Update similarity value display
        similarityFilter.addEventListener('input', () => {
            similarityValue.textContent = `${similarityFilter.value}%`;
        });
        
        // Apply filters
        applyFiltersBtn.addEventListener('click', applyFilters);
        
        // Reset filters
        resetFiltersBtn.addEventListener('click', () => {
            clusterFilter.value = '';
            urlFilter.value = '';
            pageTypeFilter.value = '';
            similarityFilter.value = 0;
            similarityValue.textContent = '0%';
            applyFilters();
        });
    }
    
    // Function to apply filters
    function applyFilters() {
        const clusterFilter = document.getElementById('cluster-filter').value;
        const urlFilter = document.getElementById('url-filter').value;
        const pageTypeFilter = document.getElementById('page-type-filter').value;
        const similarityFilter = parseFloat(document.getElementById('similarity-filter').value) / 100;
        
        // Keep track of visible rows for download
        window.visibleRows = [];
        
        window.allResultRows.forEach(row => {
            let showRow = true;
            
            // Apply cluster filter
            if (clusterFilter && row.data['Cluster'] !== clusterFilter) {
                showRow = false;
            }
            
            // Apply URL filter
            if (urlFilter && showRow) {
                let urlMatch = false;
                for (let key in row.data) {
                    if (key === 'URL' || key.startsWith('URL')) {
                        if (row.data[key] === urlFilter) {
                            urlMatch = true;
                            break;
                        }
                    }
                }
                if (!urlMatch) {
                    showRow = false;
                }
            }
            
            // Apply page type filter
            if (pageTypeFilter && showRow) {
                let pageTypeMatch = false;
                for (let key in row.data) {
                    if (key === 'PageType1' || key === 'PageType2' || key === 'PageType3') {
                        if (row.data[key] === pageTypeFilter) {
                            pageTypeMatch = true;
                            break;
                        }
                    }
                }
                if (!pageTypeMatch) {
                    showRow = false;
                }
            }
            
            // Apply similarity filter
            if (similarityFilter > 0 && showRow) {
                let similarityMatch = false;
                for (let key in row.data) {
                    if (key === 'Similarity' || key.startsWith('Similarity')) {
                        if (parseFloat(row.data[key]) >= similarityFilter) {
                            similarityMatch = true;
                            break;
                        }
                    }
                }
                if (!similarityMatch) {
                    showRow = false;
                }
            }
            
            // Show or hide row
            row.element.style.display = showRow ? '' : 'none';
            
            // Add to visible rows if shown
            if (showRow) {
                window.visibleRows.push(row);
            }
        });
        
        // Update download button visibility
        if (window.visibleRows.length > 0) {
            downloadFilteredBtn.style.display = 'inline-block';
        } else {
            downloadFilteredBtn.style.display = 'none';
        }
    }
    
    // Function to clear form
    document.getElementById('reset-btn').addEventListener('click', () => {
        form.reset();
        keywordsFileName.textContent = 'Aucun fichier choisi';
        urlsFileName.textContent = 'Aucun fichier choisi';
        kmeansParams.classList.remove('hidden');
        dbscanParams.classList.add('hidden');
        openaiParams.classList.remove('hidden');
        stParams.classList.add('hidden');
        clearAlert();
        previewContainer.classList.add('hidden');
        resultSection.classList.add('hidden');
    });
    
    // Function to download filtered results
    function downloadFilteredResults() {
        if (!window.visibleRows || window.visibleRows.length === 0) return;
        
        // Get headers from the table
        const headers = [];
        const headerRow = document.querySelector('#preview-header tr');
        if (headerRow) {
            const headerCells = headerRow.querySelectorAll('th');
            headerCells.forEach(cell => {
                headers.push(cell.textContent);
            });
        }
        
        // Create CSV content
        let csvContent = headers.join(',') + '\n';
        
        // Add visible rows data
        window.visibleRows.forEach(row => {
            const rowValues = [];
            headers.forEach(header => {
                rowValues.push(row.data[header] || '');
            });
            csvContent += rowValues.join(',') + '\n';
        });
        
        // Create download link
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', 'resultats_filtres.csv');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    
    // Add event listener for download filtered button
    downloadFilteredBtn.addEventListener('click', downloadFilteredResults);
});
