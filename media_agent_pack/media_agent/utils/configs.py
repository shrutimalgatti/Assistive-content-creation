
import os
from dotenv import load_dotenv

load_dotenv()

ROOT_MODEL_NAME = "gemini-1.5-pro-latest"
IMAGE_AGENT_MODEL_NAME = "gemini-1.5-pro-latest"
AUDIO_AGENT_MODEL_NAME = "gemini-1.5-pro-latest"
TEXT_AGENT_MODEL_NAME = "gemini-1.5-pro-latest"

# Models for tools
IMAGE_TOOL_MODEL_NAME = "gemini-2.5-flash-image"
PROMPT_ENHANCEMENT_MODEL_NAME = "gemini-1.5-flash-latest"

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO").upper()
DETAILED_FRAMEWORK_LOGGING = os.getenv("DETAILED_FRAMEWORK_LOGGING", "False").lower() == "true"

SAVE_LOCALLY: bool = os.getenv("SAVE_LOCALLY", "True").lower() == "true"
LOCAL_SAVE_PATH: str = os.getenv("LOCAL_SAVE_PATH", "local_results")
SAVE_PROMPT_TO_FILE: bool = os.getenv("SAVE_PROMPT_TO_FILE", "True").lower() == "true"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
