
# Import logging
import logging
import base64

# Import google libs
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext


# Import your sub-agents
from .sub_agents.image_agent.agent import image_agent
from .sub_agents.audio_agent.agent import audio_agent 
from .sub_agents.text_agent.agent import text_agent

# Import utils and config
from utils import configs


logger = logging.getLogger(__name__)

def _decode_b64_str(s: str) -> bytes:
    if isinstance(s, str) and s.startswith("data:"):
        parts = s.split(",", 1)
        if len(parts) == 2:
            s = parts[1]
    s = s.strip()
    padding = len(s) % 4
    if padding:
        s += "=" * (4 - padding)
    try:
        return base64.b64decode(s)
    except Exception as e:
        logger.error(f"Base64 decoding failed in callback: {e}", exc_info=True)
        raise

def _save_uploaded_image_to_state(callback_context: CallbackContext):
    """
    Checks user content for image uploads. If an image is found, it
    **clears all previous image data from the state** before saving the new image.
    This is critical to prevent the agent from using stale images from previous turns.
    """
    logger.info("--- Entering _save_uploaded_image_to_state callback ---")
    state = callback_context.state
    user_content = callback_context.user_content

    if not user_content or not user_content.parts:
        logger.info("Callback: No content or parts found, exiting.")
        return

    # Check if there are any images in the current turn's content.
    is_new_image_upload = any(
        hasattr(part, 'inline_data') and getattr(part.inline_data, 'mime_type', '').startswith('image/')
        for part in user_content.parts
    )

    # *** THE CRITICAL FIX ***
    # If a new image is detected, we MUST clear the state first.
    if is_new_image_upload:
        logger.info("New image upload detected. Clearing all previous image data from session state.")
        keys_to_clear = [
            "uploaded_image_b64",
            "uploaded_mask_b64",
            "uploaded_image_parts",
        ]
        cleared_keys = []
        for key in keys_to_clear:
            if key in state:
                del state[key]
                cleared_keys.append(key)
        if cleared_keys:
            logger.info(f"Successfully cleared stale keys: {cleared_keys}")

    # Now, proceed to process and save the new image data.
    image_parts_list = []
    first_image_b64 = None
    for i, part in enumerate(user_content.parts):
        if hasattr(part, 'inline_data') and getattr(part.inline_data, 'mime_type', '').startswith('image/'):
            mime_type = part.inline_data.mime_type
            logger.debug(f"Callback: Processing new image part {i} (mime: {mime_type})")
            part_data_container = part.inline_data
            current_data_bytes = None

            raw_bytes = getattr(part_data_container, 'data', None)
            if raw_bytes and isinstance(raw_bytes, (bytes, bytearray)):
                current_data_bytes = raw_bytes
            else:
                b64_data = getattr(part_data_container, 'b64_json', None)
                if b64_data and isinstance(b64_data, str):
                    try:
                        current_data_bytes = _decode_b64_str(b64_data)
                    except Exception as e:
                        logger.error(f"Callback: Failed to decode base64 from part {i}: {e}")
                        continue
                else:
                    logger.warning(f"Callback: Image part {i} found but no suitable data found.")
                    continue

            if current_data_bytes:
                current_b64_str = base64.b64encode(current_data_bytes).decode('utf-8')
                image_part_info = {"b64": current_b64_str, "mime_type": mime_type}
                image_parts_list.append(image_part_info)
                if first_image_b64 is None:
                    first_image_b64 = current_b64_str

    # If any new images were processed, save them to the (now clean) state.
    if image_parts_list:
        state["uploaded_image_parts"] = image_parts_list
        logger.info(f"Callback: Saved {len(image_parts_list)} new image parts to state['uploaded_image_parts'].")
    if first_image_b64:
        state["uploaded_image_b64"] = first_image_b64
        logger.info("Callback: Saved first image base64 string to state['uploaded_image_b64'].")

    logger.info("--- Exiting _save_uploaded_image_to_state callback ---")


# --- The rest of your agent.py file remains unchanged ---
root_agent = Agent(
    name="root_agent",
    model=configs.ROOT_MODEL_NAME,
    description="Routes requests to specialized sub-agents for images, audio, or text.",
    instruction=(
        "You are the main router agent. Your job is to delegate the user's request "
        "to the most suitable sub-agent. You know about the following agents:\n"
        "- 'image_agent': Handles image generation, editing, and re-styling.\n"
        "- 'audio_agent': Handles text-to-speech generation, including different voice styles.\n"
        "- 'text_agent': Handles rewriting and re-styling of text.\n"
        "When the user greets the agent (for example: 'hi', 'hello', 'hey'), respond with a concise welcome message that: \n"
        "  1) Welcomes the user,\n"
        "  2) Lists the agent's primary capabilities (image generation/editing, audio/text-to-speech, text rewriting/restyling), and\n"
        "  3) Asks a follow-up question asking what they would like to do next (for example: 'What would you like to do today — generate an image, edit an image, convert text to speech, or rewrite some text?').\n"
        "If the user message is not a greeting, analyze the user's request and delegate it to the correct sub-agent."
    ),
    tools=[],
    sub_agents=[image_agent, audio_agent, text_agent],
    before_agent_callback=_save_uploaded_image_to_state
)

logger.info(f"root_agent '{root_agent.name}' initialized with sub-agents '{[sa.name for sa in root_agent.sub_agents]}' and before_agent_callback.")
