import logging
import os
from datetime import datetime

def setup_logger(name: str, log_dir: str = "fl/logging/logs", level=logging.INFO, to_console=True):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)  # Console shows only warnings+
            console_formatter = logging.Formatter("%(levelname)s: %(message)s")
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

    return logger