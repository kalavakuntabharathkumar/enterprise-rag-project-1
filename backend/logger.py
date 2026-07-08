import logging
import os
from backend.config import Config

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Setup logger with file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create logs directory if it doesn't exist
    os.makedirs(Config.LOGS_PATH, exist_ok=True)

    # File handler
    file_handler = logging.FileHandler(os.path.join(Config.LOGS_PATH, log_file))
    file_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    return logger

# Create loggers
app_logger = setup_logger('app', 'app.log')
query_logger = setup_logger('query', 'queries.log')