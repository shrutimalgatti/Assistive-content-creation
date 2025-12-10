import logging
import sys
from utils import configs

# For convenience, we can re-export the configs from here so other tool
# files can continue to import from `settings` if they need to.
GOOGLE_API_KEY = configs.GOOGLE_API_KEY
SAVE_PROMPT_TO_FILE = configs.SAVE_PROMPT_TO_FILE
SAVE_LOCALLY = configs.SAVE_LOCALLY
LOCAL_SAVE_PATH = configs.LOCAL_SAVE_PATH
if not configs.GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")

def setup_logging():

    root_logger = logging.getLogger() 
    root_logger.setLevel(configs.LOGGING_LEVEL)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)

    if not configs.DETAILED_FRAMEWORK_LOGGING:
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

    logging.info(f"Logging configured. Base level: {configs.LOGGING_LEVEL}, Detailed framework logs: {configs.DETAILED_FRAMEWORK_LOGGING}")