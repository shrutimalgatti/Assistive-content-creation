import logging
from google.adk.agents import Agent
from ...tools.text_tools import text_restyle_tool
from .prompt import TEXT_AGENT_INSTR

logger = logging.getLogger(__name__)

text_agent = Agent(
    name="text_agent",
    model="gemini-2.0-flash-001",
    description="Handles user requests to rewrite or change the style of a piece of text.",
    instruction=TEXT_AGENT_INSTR,
    tools=[text_restyle_tool],
)

logger.info(f"text_agent '{text_agent.name}' initialized.")
