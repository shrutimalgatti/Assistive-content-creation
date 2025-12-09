import logging
from google.adk.agents import Agent


from ...tools.image_generation import (
    prompt_enhance_tool,
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
    model="gemini-2.5-flash", 
    description="Handles user requests related to image generation and image prompt enhancement.",
    instruction=IMAGE_AGENT_INSTR,
    tools=[
        prompt_enhance_tool,
        gemini_image_generation_tool,
        gemini_image_restyle_tool,
        gemini_image_edit_tool,
        clear_image_state_tool,
        save_artifact_to_state_tool,
    
    ],

)

logger.info(f"image_agent '{image_agent.name}' initialized with tools.") 
