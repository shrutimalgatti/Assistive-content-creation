import os
from dotenv import load_dotenv
import logging
import sys

load_dotenv()

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO").upper()
DETAILED_FRAMEWORK_LOGGING = os.getenv("DETAILED_FRAMEWORK_LOGGING", "False").lower() == "true"

SAVE_IMAGES_LOCALLY: bool = True
LOCAL_IMAGE_SAVE_PATH: str = "local_image_results"

SAVE_PROMPT_TO_FILE: bool = True

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")

def setup_logging():

    root_logger = logging.getLogger()
    root_logger.setLevel(LOGGING_LEVEL)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)

    if not DETAILED_FRAMEWORK_LOGGING:
        noisy_loggers = [
            "google.adk",
            "google.generativeai",
            "urllib3",
            "httpx",
        ]
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
            logging.info(f"Set logger '{logger_name}' to WARNING level.") # Confirm setting
        
        logging.getLogger("google_genai.types").setLevel(logging.ERROR)
        logging.info("Set logger 'google_genai.types' to ERROR level.")

    logging.info(f"Logging configured. Base level: {LOGGING_LEVEL}, Detailed framework logs: {DETAILED_FRAMEWORK_LOGGING}")