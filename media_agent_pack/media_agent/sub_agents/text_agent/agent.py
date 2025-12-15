import logging
from google.adk.agents import Agent
from media_agent.utils import configs
from ...tools.text_tools import text_restyle_tool
from .prompt import TEXT_AGENT_INSTR

logger = logging.getLogger(__name__)

text_agent = Agent(
    name="text_agent",
    model=configs.TEXT_AGENT_MODEL_NAME,
    description="Handles user requests to rewrite or change the style of a piece of text.",
    instruction=TEXT_AGENT_INSTR,
    tools=[text_restyle_tool],
)

logger.info(f"text_agent '{text_agent.name}' initialized.")
