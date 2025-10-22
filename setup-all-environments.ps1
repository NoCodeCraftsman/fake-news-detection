# ========================================
# COMPREHENSIVE ENVIRONMENT SETUP SCRIPT
# For Professional Fake News Detection Project
# ========================================

# Navigate to project directory
cd "L:/Important/MCA/Mini Project/fake_news_detection"

Write-Host "🚀 Setting up professional ML project environments..." -ForegroundColor Cyan
Write-Host "Project Structure: Multi-environment ML pipeline" -ForegroundColor White
Write-Host ""

# ========================================
# ENV_PREPROCESSING: Advanced Text Processing
# ========================================

Write-Host "📝 Setting up ENV_PREPROCESSING..." -ForegroundColor Yellow
python -m venv env_preprocessing
.\env_preprocessing\Scripts\Activate.ps1

# Core data processing
pip install pandas==2.0.3 numpy==1.24.4 scipy==1.13.1 scikit-learn==1.3.0
pip install openpyxl==3.1.2 pyarrow==14.0.2

# Advanced NLP preprocessing
pip install nltk==3.8.1 spacy==3.7.2 textblob==0.17.1
pip install beautifulsoup4==4.12.2 html5lib==1.1
pip install contractions==0.1.73 langdetect==1.0.9
pip install ftfy==6.1.1 unidecode==1.3.7 pyspellchecker==0.7.2

# Utilities
pip install tqdm==4.66.0 regex==2023.10.3

# Download language models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('vader_lexicon')"

deactivate
Write-Host "✅ ENV_PREPROCESSING setup complete" -ForegroundColor Green
Write-Host ""

# ========================================
# ENV_NLP: Model Training (Already exists - upgrade)
# ========================================

Write-Host "🤖 Upgrading ENV_NLP for advanced training..." -ForegroundColor Yellow
.\env_nlp\Scripts\Activate.ps1

# Ensure latest versions for training
pip install --upgrade torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install --upgrade transformers==4.35.0 datasets==2.14.4 accelerate==0.24.0
pip install --upgrade tensorboard==2.15.0 wandb==0.16.0
pip install --upgrade scikit-learn==1.3.0 pandas==2.0.3 numpy==1.24.4

# Training optimization tools
pip install optuna==3.4.0 hyperopt==0.2.7

deactivate
Write-Host "✅ ENV_NLP upgrade complete" -ForegroundColor Green
Write-Host ""

# ========================================
# ENV_API: FastAPI Backend (Upgrade existing)
# ========================================

Write-Host "🌐 Setting up/upgrading ENV_API..." -ForegroundColor Yellow
if (Test-Path "env_api") {
    .\env_api\Scripts\Activate.ps1
} else {
    python -m venv env_api
    .\env_api\Scripts\Activate.ps1
}

# FastAPI stack
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0
pip install pydantic==2.5.0 python-multipart==0.0.6

# ML serving
pip install torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.35.0 pandas==2.0.3 numpy==1.24.4

# Security and utilities
pip install python-jose[cryptography]==3.3.0 passlib[bcrypt]==1.7.4
pip install aiofiles==23.2.1 python-dotenv==1.0.0

deactivate
Write-Host "✅ ENV_API setup complete" -ForegroundColor Green
Write-Host ""

# ========================================
# ENV_FRONTEND: Streamlit Interface (Upgrade existing)
# ========================================

Write-Host "🎨 Setting up/upgrading ENV_FRONTEND..." -ForegroundColor Yellow
if (Test-Path "env_frontend") {
    .\env_frontend\Scripts\Activate.ps1
} else {
    python -m venv env_frontend
    .\env_frontend\Scripts\Activate.ps1
}

# Streamlit and visualization
pip install streamlit==1.28.2 plotly==5.17.0
pip install matplotlib==3.7.0 seaborn==0.12.0
pip install pandas==2.0.3 numpy==1.24.4
pip install requests==2.31.0 pillow==10.1.0

# Additional UI components
pip install streamlit-option-menu==0.3.6 streamlit-aggrid==0.3.4

deactivate
Write-Host "✅ ENV_FRONTEND setup complete" -ForegroundColor Green
Write-Host ""

# ========================================
# ENV_EXPLAIN: Model Explainability (Upgrade existing)
# ========================================

Write-Host "🔍 Setting up/upgrading ENV_EXPLAIN..." -ForegroundColor Yellow
if (Test-Path "env_explain") {
    .\env_explain\Scripts\Activate.ps1
} else {
    python -m venv env_explain
    .\env_explain\Scripts\Activate.ps1
}

# Explainability libraries
pip install shap==0.43.0 lime==0.2.0.1
pip install captum==0.6.0 interpret==0.4.2

# ML and visualization
pip install torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.35.0 pandas==2.0.3 numpy==1.24.4
pip install matplotlib==3.7.0 plotly==5.17.0 seaborn==0.12.0

deactivate
Write-Host "✅ ENV_EXPLAIN setup complete" -ForegroundColor Green
Write-Host ""

# ========================================
# ENV_PIPELINE: Integration & MLOps (Upgrade existing)
# ========================================

Write-Host "⚙️ Setting up/upgrading ENV_PIPELINE..." -ForegroundColor Yellow
if (Test-Path "env_pipeline") {
    .\env_pipeline\Scripts\Activate.ps1
} else {
    python -m venv env_pipeline
    .\env_pipeline\Scripts\Activate.ps1
}

# MLOps and pipeline tools
pip install mlflow==2.8.1 prefect==2.14.9
pip install pytest==7.4.3 pytest-cov==4.1.0

# Data validation and monitoring
pip install great-expectations==0.18.8 evidently==0.4.5

# Basic ML for testing
pip install pandas==2.0.3 numpy==1.24.4 scikit-learn==1.3.0

deactivate
Write-Host "✅ ENV_PIPELINE setup complete" -ForegroundColor Green
Write-Host ""

# ========================================
# VERIFICATION SCRIPT
# ========================================

Write-Host "🔧 Verifying all environments..." -ForegroundColor Magenta

# Test each environment
$environments = @("env_preprocessing", "env_nlp", "env_api", "env_frontend", "env_explain", "env_pipeline")

foreach ($env in $environments) {
    Write-Host "Testing $env..." -ForegroundColor Gray
    if (Test-Path $env) {
        .\$env\Scripts\Activate.ps1
        python -c "import sys; print(f'✅ {sys.prefix.split('\\')[-1]} - Python {sys.version[:5]}')"
        deactivate
    } else {
        Write-Host "⚠️ $env not found" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "🎉 MULTI-ENVIRONMENT SETUP COMPLETE!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 ENVIRONMENT SUMMARY:" -ForegroundColor Cyan
Write-Host "├── env_preprocessing : Advanced text cleaning & preprocessing" -ForegroundColor White
Write-Host "├── env_nlp          : Model training with GPU support" -ForegroundColor White  
Write-Host "├── env_api          : FastAPI backend services" -ForegroundColor White
Write-Host "├── env_frontend     : Streamlit web interface" -ForegroundColor White
Write-Host "├── env_explain      : Model interpretability (SHAP/LIME)" -ForegroundColor White
Write-Host "└── env_pipeline     : MLOps and integration testing" -ForegroundColor White
Write-Host ""
Write-Host "🚀 Ready for production-grade ML development!" -ForegroundColor Green