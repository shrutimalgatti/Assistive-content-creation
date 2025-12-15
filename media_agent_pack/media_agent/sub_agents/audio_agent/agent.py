import logging
from google.adk.agents import Agent
from media_agent.utils import configs
from ...tools.text_to_speech_tool import text_to_speech_tool
from .prompt import AUDIO_AGENT_INSTR

logger = logging.getLogger(__name__)

audio_agent = Agent(
    name="audio_agent",
    model=configs.AUDIO_AGENT_MODEL_NAME,
    description="Handles user requests to convert text into speech.",
    instruction=AUDIO_AGENT_INSTR,
    tools=[text_to_speech_tool],
)

logger.info(f"audio_agent '{audio_agent.name}' initialized.")