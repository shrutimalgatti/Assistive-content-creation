
import os
from pathlib import Path

from dotenv import load_dotenv

# Anchor to media_agent/.env rather than searching upward from the current
# working directory, which depends on where `adk` happens to be launched from.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ROOT_MODEL_NAME = "gemini-2.5-pro"
IMAGE_AGENT_MODEL_NAME = "gemini-2.5-pro"
AUDIO_AGENT_MODEL_NAME = "gemini-2.5-pro"
TEXT_AGENT_MODEL_NAME = "gemini-2.5-pro"

# Models for tools
IMAGE_TOOL_MODEL_NAME = "gemini-2.5-flash-image"
PROMPT_ENHANCEMENT_MODEL_NAME = "gemini-2.5-pro"
TEXT_TOOL_MODEL_NAME = "gemini-2.0-flash-001"

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO").upper()
DETAILED_FRAMEWORK_LOGGING = os.getenv("DETAILED_FRAMEWORK_LOGGING", "False").lower() == "true"

SAVE_LOCALLY: bool = os.getenv("SAVE_LOCALLY", "True").lower() == "true"
LOCAL_SAVE_PATH: str = os.getenv("LOCAL_SAVE_PATH", "local_results")
SAVE_PROMPT_TO_FILE: bool = os.getenv("SAVE_PROMPT_TO_FILE", "True").lower() == "true"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
