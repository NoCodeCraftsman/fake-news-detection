import logging
import os
from pathlib import Path
from datetime import datetime

def setup_logging(log_name="app", environment="nlp"):
    """
    Setup logging for different environments
    Args:
        log_name: Name of the log file
        environment: Environment name (nlp, api, frontend, etc.)
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create environment-specific log file
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"{log_name}_{environment}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(f"{log_name}_{environment}")