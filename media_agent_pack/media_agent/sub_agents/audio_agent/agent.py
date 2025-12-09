import logging
from google.adk.agents import Agent
from ...tools.text_to_speech_tool import text_to_speech_tool
from .prompt import AUDIO_AGENT_INSTR

logger = logging.getLogger(__name__)

audio_agent = Agent(
    name="audio_agent",
    model="gemini-2.0-flash-001",
    description="Handles user requests to convert text into spoken audio (text-to-speech).",
    instruction=AUDIO_AGENT_INSTR,
    tools=[
        text_to_speech_tool,
    ],
)

logger.info(f"audio_agent '{audio_agent.name}' initialized with tools.")