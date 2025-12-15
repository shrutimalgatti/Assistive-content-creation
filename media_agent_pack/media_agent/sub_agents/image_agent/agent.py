import logging
from google.adk.agents import Agent
from media_agent.utils import configs

from ...tools.image_generation import (
    ## prompt_enhance_tool,
    _enhance_prompt_for_image_gen,
    gemini_image_generation_tool,
    gemini_image_restyle_tool,
    gemini_image_edit_tool,
    clear_image_state_tool,
    save_artifact_to_state_tool,
)
from .prompt import IMAGE_AGENT_INSTR

logger = logging.getLogger(__name__)

image_agent = Agent(
    name="image_agent",
    model=configs.IMAGE_AGENT_MODEL_NAME,
    description="Handles all user requests related to image generation, editing, and re-styling.",
    instruction=IMAGE_AGENT_INSTR,
    tools=[
        ## prompt_enhance_tool,
        _enhance_prompt_for_image_gen,
        gemini_image_generation_tool,
        gemini_image_restyle_tool,
        gemini_image_edit_tool,
        clear_image_state_tool,
        save_artifact_to_state_tool,
    ],

)

logger.info(f"image_agent '{image_agent.name}' initialized.")
