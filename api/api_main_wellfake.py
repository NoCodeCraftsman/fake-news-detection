
# """
# FastAPI Backend for Fake News Detection - WELFake Dataset
# Professional-grade API using fine-tuned BERT and RoBERTa models trained on WELFake dataset
# Includes ensemble averaging, temperature scaling, and uncertainty threshold
# """
# from fastapi import FastAPI, HTTPException, Depends, Request
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import logging
# import time
# from typing import Dict, List, Optional
# import json
# from pathlib import Path
# from datetime import datetime
# import sys
# import re

# # Constants for ensemble and calibration
# ENSEMBLE_MODELS = ["bert-base-uncased", "roberta-base"]
# TEMPERATURE = 2.0
# CONFIDENCE_THRESHOLD = 0.7

# # Project root
# PROJECT_ROOT = Path("L:/Important/MCA/Mini Project/fake_news_detection")
# sys.path.append(str(PROJECT_ROOT / "src"))

# # Setup logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# logger = logging.getLogger(__name__)

# # Initialize FastAPI app
# app = FastAPI(
#     title="Fake News Detection API - WELFake Dataset",
#     version="2.2.0",
#     description="API using BERT/RoBERTa models fine-tuned on WELFake dataset",
#     docs_url="/docs",
#     redoc_url="/redoc"
# )

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
#     allow_credentials=True,
# )

# # Security
# security = HTTPBearer(auto_error=False)

# # Models for requests and responses
# class NewsRequest(BaseModel):
#     text: str = Field(..., min_length=10, max_length=5000, description="News text to classify")
#     title: str = Field(default="", max_length=500, description="News title (optional)")
#     model_type: str = Field(default="ensemble", description="Model type: ensemble, bert, or roberta")
#     explain: bool = Field(default=True, description="Include explanation in response")

# class PredictionResponse(BaseModel):
#     prediction: str
#     confidence: float
#     probabilities: Dict[str, float]
#     model_used: str
#     model_paths: Dict[str, str]
#     processing_time: float
#     explanation: Optional[Dict]
#     timestamp: str
#     input_stats: Dict[str, int]

# class HealthResponse(BaseModel):
#     status: str
#     version: str
#     uptime: float
#     total_predictions: int
#     models_loaded: List[str]
#     dataset_info: str

# # Globals
# models_cache = {}
# tokenizers_cache = {}
# model_paths_cache = {}
# start_time = time.time()
# prediction_count = 0
# request_counts: Dict[str, List[float]] = {}
# RATE_LIMIT = 100  # per hour

# def find_best_checkpoint(trial_dir):
#     """Find the best checkpoint in a trial directory"""
#     steps = []
#     for sub in trial_dir.iterdir():
#         m = re.match(r"checkpoint-(\d+)$", sub.name)
#         if m and sub.is_dir():
#             steps.append((int(m.group(1)), sub))
#     return max(steps, key=lambda x: x[0])[1] if steps else None

# def find_fine_tuned_model(model_type: str) -> Optional[Path]:
#     """Find the best fine-tuned model checkpoint from hyperparameter tuning results"""
#     results_dir = PROJECT_ROOT / "results"

#     if not results_dir.exists():
#         logger.warning(f"Results directory not found: {results_dir}")
#         return None

#     # Look through trial directories for matching model
#     for trial_dir in sorted(results_dir.glob("trial_*")):
#         checkpoint = find_best_checkpoint(trial_dir)
#         if not checkpoint:
#             continue

#         # Check if this checkpoint matches our model type
#         config_file = checkpoint / "config.json"
#         if not config_file.exists():
#             continue

#         try:
#             with open(config_file, 'r') as f:
#                 config = json.load(f)
#                 if model_type in config.get('_name_or_path', ''):
#                     logger.info(f"Found fine-tuned model for {model_type}: {checkpoint}")
#                     return checkpoint
#         except Exception as e:
#             logger.debug(f"Error reading config from {config_file}: {e}")
#             continue

#     logger.warning(f"No fine-tuned model found for {model_type}")
#     return None

# def load_model(model_type: str):
#     """Load model and tokenizer, preferring fine-tuned version"""
#     if model_type not in models_cache:
#         # Try to find fine-tuned model first
#         checkpoint_path = find_fine_tuned_model(model_type)

#         if checkpoint_path and checkpoint_path.exists():
#             # Load fine-tuned model
#             try:
#                 tokenizer = AutoTokenizer.from_pretrained(model_type)  # Use original tokenizer
#                 model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
#                 model.eval()

#                 models_cache[model_type] = model
#                 tokenizers_cache[model_type] = tokenizer
#                 model_paths_cache[model_type] = str(checkpoint_path)

#                 logger.info(f"Loaded fine-tuned {model_type} from {checkpoint_path}")

#             except Exception as e:
#                 logger.error(f"Error loading fine-tuned {model_type}: {e}")
#                 raise HTTPException(503, f"Failed to load fine-tuned {model_type} model: {str(e)}")
#         else:
#             # Fallback to base model if no fine-tuned version found
#             logger.warning(f"No fine-tuned {model_type} found, loading base model")
#             try:
#                 tokenizer = AutoTokenizer.from_pretrained(model_type)
#                 model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=2)
#                 model.eval()

#                 models_cache[model_type] = model
#                 tokenizers_cache[model_type] = tokenizer
#                 model_paths_cache[model_type] = f"base-{model_type}"

#             except Exception as e:
#                 logger.error(f"Error loading base {model_type}: {e}")
#                 raise HTTPException(503, f"Failed to load {model_type} model: {str(e)}")

#     return models_cache[model_type], tokenizers_cache[model_type], model_paths_cache[model_type]

# def check_rate_limit(req: Request):
#     """Check rate limiting"""
#     ip = req.client.host
#     now = time.time()
#     request_counts.setdefault(ip, [])
#     request_counts[ip] = [t for t in request_counts[ip] if now - t < 3600]
#     if len(request_counts[ip]) >= RATE_LIMIT:
#         raise HTTPException(429, "Rate limit exceeded. Max 100 requests per hour.")
#     request_counts[ip].append(now)

# def prepare_input_text(text: str, title: str = "") -> str:
#     """Prepare input text, combining title and text if both provided"""
#     if title.strip():
#         # Combine title and text similar to preprocessing
#         combined = f"{title.strip()} [SEP] {text.strip()}"
#     else:
#         combined = text.strip()

#     return combined

# async def predict_single_model(text: str, model_type: str) -> Dict:
#     """Predict using a single model"""
#     try:
#         model, tokenizer, model_path = load_model(model_type)

#         # Tokenize input
#         inputs = tokenizer(
#             text, 
#             return_tensors="pt", 
#             truncation=True, 
#             padding=True, 
#             max_length=512
#         )

#         # Get prediction
#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits = outputs.logits

#         # Apply temperature scaling
#         scaled_logits = logits / TEMPERATURE
#         probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)[0]

#         # Get prediction
#         pred_idx = int(probabilities.argmax())
#         confidence = float(probabilities[pred_idx])

#         classes = ["real", "fake"]
#         prediction = classes[pred_idx]

#         # Apply uncertainty threshold
#         if confidence < CONFIDENCE_THRESHOLD:
#             prediction = "uncertain"

#         return {
#             "prediction": prediction,
#             "confidence": confidence,
#             "probabilities": {
#                 "real": float(probabilities[0]),
#                 "fake": float(probabilities[1])
#             },
#             "model_path": model_path,
#             "success": True
#         }

#     except Exception as e:
#         logger.error(f"Error in {model_type} prediction: {e}")
#         return {
#             "success": False,
#             "error": str(e)
#         }

# async def predict_ensemble(text: str) -> Dict:
#     """Predict using ensemble of models"""
#     try:
#         logits_sum = None
#         successful_models = []
#         model_paths = {}

#         for model_type in ENSEMBLE_MODELS:
#             try:
#                 model, tokenizer, model_path = load_model(model_type)
#                 model_paths[model_type] = model_path

#                 # Tokenize input
#                 inputs = tokenizer(
#                     text, 
#                     return_tensors="pt", 
#                     truncation=True, 
#                     padding=True, 
#                     max_length=512
#                 )

#                 # Get logits
#                 with torch.no_grad():
#                     outputs = model(**inputs)
#                     logits = outputs.logits

#                 # Sum logits for ensemble
#                 if logits_sum is None:
#                     logits_sum = logits
#                 else:
#                     logits_sum += logits

#                 successful_models.append(model_type)

#             except Exception as e:
#                 logger.warning(f"Failed to use {model_type} in ensemble: {e}")
#                 continue

#         if not successful_models:
#             return {
#                 "success": False,
#                 "error": "No models available for ensemble prediction"
#             }

#         # Average logits and apply temperature scaling
#         avg_logits = logits_sum / len(successful_models)
#         scaled_logits = avg_logits / TEMPERATURE
#         probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)[0]

#         # Get prediction
#         pred_idx = int(probabilities.argmax())
#         confidence = float(probabilities[pred_idx])

#         classes = ["real", "fake"]
#         prediction = classes[pred_idx]

#         # Apply uncertainty threshold
#         if confidence < CONFIDENCE_THRESHOLD:
#             prediction = "uncertain"

#         return {
#             "prediction": prediction,
#             "confidence": confidence,
#             "probabilities": {
#                 "real": float(probabilities[0]),
#                 "fake": float(probabilities[1])
#             },
#             "model_paths": model_paths,
#             "successful_models": successful_models,
#             "success": True
#         }

#     except Exception as e:
#         logger.error(f"Error in ensemble prediction: {e}")
#         return {
#             "success": False,
#             "error": str(e)
#         }

# @app.get("/health", response_model=HealthResponse)
# async def health():
#     """Health check endpoint"""
#     loaded_models = list(models_cache.keys())

#     return HealthResponse(
#         status="healthy",
#         version="2.2.0",
#         uptime=time.time() - start_time,
#         total_predictions=prediction_count,
#         models_loaded=loaded_models,
#         dataset_info="WELFake Dataset - Fine-tuned BERT/RoBERTa"
#     )

# @app.post("/predict", response_model=PredictionResponse)
# async def predict(
#     req: NewsRequest, 
#     request: Request, 
#     creds: HTTPAuthorizationCredentials = Depends(security)
# ):
#     """Predict if news is real or fake"""
#     global prediction_count

#     # Check rate limiting
#     check_rate_limit(request)

#     start_time_pred = time.time()

#     # Prepare input text
#     input_text = prepare_input_text(req.text, req.title)

#     # Get prediction based on model type
#     if req.model_type == "ensemble":
#         result = await predict_ensemble(input_text)
#         model_used = f"ensemble-{'-'.join(ENSEMBLE_MODELS)}"
#         model_paths = result.get("model_paths", {})
#     elif req.model_type in ["bert-base-uncased", "roberta-base"]:
#         result = await predict_single_model(input_text, req.model_type)
#         model_used = req.model_type
#         model_paths = {req.model_type: result.get("model_path", "")}
#     else:
#         raise HTTPException(400, f"Invalid model type: {req.model_type}. Use 'ensemble', 'bert-base-uncased', or 'roberta-base'")

#     # Check if prediction was successful
#     if not result.get("success", False):
#         raise HTTPException(500, result.get("error", "Prediction failed"))

#     # Increment counter
#     prediction_count += 1

#     # Prepare explanation
#     explanation = None
#     if req.explain:
#         explanation = {
#             "method": "ensemble_temperature_scaling" if req.model_type == "ensemble" else "single_model_temperature_scaling",
#             "temperature": TEMPERATURE,
#             "confidence_threshold": CONFIDENCE_THRESHOLD,
#             "models_used": result.get("successful_models", [req.model_type]),
#             "uncertainty_handling": "predictions below threshold marked as uncertain"
#         }

#     # Input statistics
#     input_stats = {
#         "text_length": len(req.text),
#         "title_length": len(req.title),
#         "combined_length": len(input_text),
#         "word_count": len(input_text.split())
#     }

#     processing_time = time.time() - start_time_pred

#     return PredictionResponse(
#         prediction=result["prediction"],
#         confidence=result["confidence"],
#         probabilities=result["probabilities"],
#         model_used=model_used,
#         model_paths=model_paths,
#         processing_time=processing_time,
#         explanation=explanation,
#         timestamp=datetime.now().isoformat(),
#         input_stats=input_stats
#     )

# @app.get("/models")
# async def get_models():
#     """Get information about loaded models"""
#     model_info = {}

#     for model_type in ENSEMBLE_MODELS:
#         try:
#             checkpoint_path = find_fine_tuned_model(model_type)
#             model_info[model_type] = {
#                 "checkpoint_available": checkpoint_path is not None,
#                 "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
#                 "loaded": model_type in models_cache,
#                 "cache_path": model_paths_cache.get(model_type)
#             }
#         except Exception as e:
#             model_info[model_type] = {
#                 "error": str(e),
#                 "loaded": False
#             }

#     return {
#         "ensemble_models": ENSEMBLE_MODELS,
#         "model_details": model_info,
#         "temperature": TEMPERATURE,
#         "confidence_threshold": CONFIDENCE_THRESHOLD
#     }

# @app.post("/reload-models")
# async def reload_models():
#     """Reload all models (clears cache)"""
#     global models_cache, tokenizers_cache, model_paths_cache

#     models_cache.clear()
#     tokenizers_cache.clear()
#     model_paths_cache.clear()

#     logger.info("Model cache cleared - models will be reloaded on next request")

#     return {
#         "status": "success",
#         "message": "Model cache cleared. Models will be reloaded on next prediction request."
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
"""
FastAPI Backend for Fake News Detection - WELFake Dataset
Uses fine-tuned BERT and RoBERTa models with corrected class ordering
"""
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import time
from typing import Dict, List, Optional
import json
from pathlib import Path
from datetime import datetime
import re

# Ensemble configuration
ENSEMBLE_MODELS = ["bert-base-uncased", "roberta-base"]
TEMPERATURE = 2.0
CONFIDENCE_THRESHOLD = 0.7

# Project root
PROJECT_ROOT = Path("L:/Important/MCA/Mini Project/fake_news_detection")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Fake News Detection API - WELFake",
    version="2.3.1",
    description="Ensemble BERT/RoBERTa fine-tuned on WELFake",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Security
security = HTTPBearer(auto_error=False)

# Request & Response models
class NewsRequest(BaseModel):
    title: Optional[str] = Field(default="", max_length=500)
    text: str = Field(..., min_length=10, max_length=5000)
    model_type: str = Field(default="ensemble", description="ensemble|bert-base-uncased|roberta-base")
    explain: bool = Field(default=True)

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str
    paths: Dict[str, str]
    processing_time: float
    explanation: Optional[Dict]
    timestamp: str
    input_stats: Dict[str, int]

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    total_predictions: int
    models_loaded: List[str]
    dataset_info: str

# Globals
models_cache: Dict[str, torch.nn.Module] = {}
tokenizers_cache: Dict[str, AutoTokenizer] = {}
model_paths_cache: Dict[str, str] = {}
start_time = time.time()
prediction_count = 0
request_counts: Dict[str, List[float]] = {}
RATE_LIMIT = 100  # per hour

def find_best_checkpoint(trial_dir: Path) -> Optional[Path]:
    best = None
    max_step = -1
    for sub in trial_dir.iterdir():
        m = re.match(r"checkpoint-(\d+)$", sub.name)
        if m and sub.is_dir():
            step = int(m.group(1))
            if step > max_step:
                max_step = step
                best = sub
    return best

def find_fine_tuned_model(model_name: str) -> Optional[Path]:
    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        return None
    for trial in sorted(results_dir.glob("trial_*")):
        ckpt = find_best_checkpoint(trial)
        if not ckpt: continue
        cfg = ckpt / "config.json"
        if not cfg.exists(): continue
        cfg_data = json.loads(cfg.read_text())
        if model_name in cfg_data.get("_name_or_path", ""):
            return ckpt
    return None

def load_model(model_name: str):
    if model_name not in models_cache:
        ckpt = find_fine_tuned_model(model_name)
        if ckpt and ckpt.exists():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(ckpt)
            model.eval()
            models_cache[model_name] = model
            tokenizers_cache[model_name] = tokenizer
            model_paths_cache[model_name] = str(ckpt)
            logger.info(f"Loaded fine-tuned {model_name} from {ckpt}")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            model.eval()
            models_cache[model_name] = model
            tokenizers_cache[model_name] = tokenizer
            model_paths_cache[model_name] = f"hf://{model_name}"
            logger.warning(f"No fine-tuned checkpoint for {model_name}, using base model")
    return models_cache[model_name], tokenizers_cache[model_name], model_paths_cache[model_name]

def check_rate_limit(req: Request):
    ip = req.client.host
    now = time.time()
    request_counts.setdefault(ip, [])
    request_counts[ip] = [t for t in request_counts[ip] if now - t < 3600]
    if len(request_counts[ip]) >= RATE_LIMIT:
        raise HTTPException(429, "Rate limit exceeded")
    request_counts[ip].append(now)

def prepare_input_text(text: str, title: str = "") -> str:
    if title:
        return f"{title.strip()} [SEP] {text.strip()}"
    return text.strip()

async def predict_single(text: str, model_name: str) -> Dict:
    model, tokenizer, path = load_model(model_name)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    scaled = logits / TEMPERATURE
    probs = torch.nn.functional.softmax(scaled, dim=-1)[0]
    idx = int(probs.argmax())
    confidence = float(probs[idx])
    # index 0 = fake, index 1 = real
    classes = ["fake", "real"]
    pred = classes[idx] if confidence >= CONFIDENCE_THRESHOLD else "uncertain"
    return {
        "prediction": pred,
        "confidence": confidence,
        "probabilities": {"fake": float(probs[0]), "real": float(probs[1])},
        "model_path": path,
        "success": True
    }

async def predict_ensemble(text: str) -> Dict:
    logits_sum = None
    paths = {}
    used = []
    for m in ENSEMBLE_MODELS:
        res = await predict_single(text, m)
        if not res["success"]:
            continue
        paths[m] = res["model_path"]
        used.append(m)
        model, tokenizer, _ = load_model(m)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            out = model(**inputs).logits
        logits_sum = out if logits_sum is None else logits_sum + out
    if logits_sum is None:
        return {"success": False, "error": "No models available"}
    avg = logits_sum / len(used)
    scaled = avg / TEMPERATURE
    probs = torch.nn.functional.softmax(scaled, dim=-1)[0]
    idx = int(probs.argmax())
    confidence = float(probs[idx])
    # index 0 = fake, index 1 = real
    classes = ["fake", "real"]
    pred = classes[idx] if confidence >= CONFIDENCE_THRESHOLD else "uncertain"
    return {
        "prediction": pred,
        "confidence": confidence,
        "probabilities": {"fake": float(probs[0]), "real": float(probs[1])},
        "paths": paths,
        "successful_models": used,
        "success": True
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version="2.3.1",
        uptime=time.time() - start_time,
        total_predictions=prediction_count,
        models_loaded=list(models_cache.keys()),
        dataset_info="WELFake preprocessed test set"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(req: NewsRequest, request: Request, creds: HTTPAuthorizationCredentials = Depends(security)):
    global prediction_count
    check_rate_limit(request)
    start = time.time()
    text = prepare_input_text(req.text, req.title)
    if req.model_type == "ensemble":
        res = await predict_ensemble(text)
        model_used = "ensemble"
        paths = res.get("paths", {})
    else:
        res = await predict_single(text, req.model_type)
        model_used = req.model_type
        paths = {req.model_type: res.get("model_path")}
    if not res["success"]:
        raise HTTPException(500, res.get("error", "Prediction failed"))
    prediction_count += 1
    explanation = None
    if req.explain:
        explanation = {
            "method": "ensemble" if model_used == "ensemble" else "single_model",
            "temperature": TEMPERATURE,
            "confidence_threshold": CONFIDENCE_THRESHOLD
        }
    input_stats = {
        "title_length": len(req.title),
        "text_length": len(req.text),
        "combined_length": len(text),
        "word_count": len(text.split())
    }
    return PredictionResponse(
        prediction=res["prediction"],
        confidence=res["confidence"],
        probabilities=res["probabilities"],
        model_used=model_used,
        paths=paths,
        processing_time=time.time() - start,
        explanation=explanation,
        timestamp=datetime.now().isoformat(),
        input_stats=input_stats
    )

@app.get("/models")
async def get_models():
    info = {}
    for m in ENSEMBLE_MODELS:
        ckpt = find_fine_tuned_model(m)
        info[m] = {
            "fine_tuned": bool(ckpt),
            "checkpoint": str(ckpt) if ckpt else None,
            "loaded": m in models_cache
        }
    return {"ensemble": ENSEMBLE_MODELS, "details": info}

@app.post("/reload-models")
async def reload_models():
    models_cache.clear()
    tokenizers_cache.clear()
    model_paths_cache.clear()
    logger.info("Cleared model cache")
    return {"status": "cache_cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)