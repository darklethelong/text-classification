# PowerShell script for deploying Complaint Detection API on Windows
# Usage: .\deploy.ps1

# Display banner
Write-Host "================================================"
Write-Host "Complaint Detection API Deployment (Windows)"
Write-Host "================================================"

# Check Docker installation
try {
    docker --version | Out-Null
} catch {
    Write-Host "Docker is not installed or not in PATH. Please install Docker Desktop for Windows." -ForegroundColor Red
    exit 1
}

# Check Docker Compose
$dockerComposeCommand = $null
try {
    # Try docker-compose first
    docker-compose --version | Out-Null
    $dockerComposeCommand = "docker-compose"
} catch {
    try {
        # Try docker compose (new style)
        docker compose version | Out-Null
        $dockerComposeCommand = "docker compose"
    } catch {
        Write-Host "Docker Compose is not installed. Please install Docker Desktop with Docker Compose." -ForegroundColor Red
        exit 1
    }
}

Write-Host "Using Docker Compose command: $dockerComposeCommand" -ForegroundColor Green

# Check for model directory
$modelDir = "outputs\models_20250321_222124"
$modelFile = "$modelDir\lstm_cnn_roberta_full.pt"
$vocabFile = "$modelDir\lstm_cnn_roberta_full_vocab.json"
$metadataFile = "$modelDir\lstm_cnn_roberta_full_metadata.json"

if (-not (Test-Path $modelDir)) {
    Write-Host "Creating model directory: $modelDir" -ForegroundColor Yellow
    New-Item -Path $modelDir -ItemType Directory -Force | Out-Null
}

# Check for model files
$missingFiles = @()
if (-not (Test-Path $modelFile)) { $missingFiles += "Model file: $modelFile" }
if (-not (Test-Path $vocabFile)) { $missingFiles += "Vocabulary file: $vocabFile" }
if (-not (Test-Path $metadataFile)) { $missingFiles += "Metadata file: $metadataFile" }

if ($missingFiles.Count -gt 0) {
    Write-Host "Warning: The following model files are missing:" -ForegroundColor Yellow
    foreach ($file in $missingFiles) {
        Write-Host "- $file" -ForegroundColor Yellow
    }
    
    $proceed = Read-Host "Do you want to proceed anyway? (y/N)"
    if ($proceed -ne "y" -and $proceed -ne "Y") {
        Write-Host "Deployment cancelled." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Proceeding with deployment, but service may not work correctly without model files." -ForegroundColor Yellow
    Write-Host "Make sure to mount the correct model files as volumes." -ForegroundColor Yellow
}

# Build and start containers
Write-Host "Building and starting Docker containers..." -ForegroundColor Cyan
Invoke-Expression "$dockerComposeCommand down"
Invoke-Expression "$dockerComposeCommand build"
Invoke-Expression "$dockerComposeCommand up -d"

# Check if container is running
Start-Sleep -Seconds 5  # Give containers time to start
$containersRunning = Invoke-Expression "$dockerComposeCommand ps -q"

if ($containersRunning) {
    Write-Host "Deployment successful!" -ForegroundColor Green
    Write-Host "API is now running at http://localhost:8000" -ForegroundColor Green
    Write-Host "API documentation: http://localhost:8000/docs" -ForegroundColor Green
} else {
    Write-Host "Error: Deployment failed. Check logs with '$dockerComposeCommand logs'" -ForegroundColor Red
    exit 1
} 