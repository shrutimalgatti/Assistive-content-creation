import base64
import logging
import os
import pathlib
import uuid
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

from google import genai
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import BaseTool, FunctionTool, ToolContext
from google.genai import types
from PIL import Image

from . import image_prompt_examples, settings


# This is a helper function for decoding base64 strings.
def _decode_b64_str(s: str) -> bytes:
    if isinstance(s, str) and s.startswith("data:"):
        parts = s.split(",", 1)
        if len(parts) == 2:
            s = parts[1]
    s = s.strip()
    padding = len(s) % 4
    if padding:
        s += "=" * (4 - padding)
    return base64.b64decode(s)

logger = logging.getLogger(__name__)

def log_prompt_to_file(formatted_prompt: str, raw_llm_response: Any, final_enhanced_prompt: str) -> None:
    if not settings.SAVE_PROMPT_TO_FILE:
        return
    try:
        PROMPT_LOG_DIR = pathlib.Path("logs/prompts")
        PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)

        today = datetime.now().strftime("%Y%m%d")
        prompt_log_file = PROMPT_LOG_DIR / f"prompts_image_gen_{today}.log"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(prompt_log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'='*50}\n")
            f.write(f"TIMESTAMP: {timestamp}\n")
            f.write(f"{'='*50}\n\n")
            f.write("--- Formatted Prompt Sent to LLM ---\n")
            f.write(formatted_prompt) # Log the fully formatted prompt
            f.write("\n\n--- Raw LLM Response Object ---\n")
            try:
                response_str = str(raw_llm_response)
            except Exception as str_err:
                response_str = f"<Error converting response object to string: {str_err}>"
            f.write(response_str)
            f.write("\n\n--- Final Enhanced Prompt String Used ---\n")
            f.write(final_enhanced_prompt) # Log the final extracted prompt
            f.write(f"\n\n{'='*50}\n")

    except Exception as e:
        logger.error(f"Failed to log prompt to file: {e}")
        import traceback
        logger.error(f"Prompt logging error details: {traceback.format_exc()}")

async def _save_uploaded_image_to_state_callback(callback_context: CallbackContext):
    logger.info("--- Entering _save_uploaded_image_to_state_callback ---")
    state = callback_context.state
    user_content = callback_context.user_content

    if not user_content or not user_content.parts:
        logger.info("Callback: No content or parts found in user_content.")
        return

    image_parts_list = []
    first_image_b64 = None


    if "uploaded_image_b64" in state:
        del state["uploaded_image_b64"]
    if "uploaded_mask_b64" in state:
        del state["uploaded_mask_b64"]
    if "uploaded_image_parts" in state:
        del state["uploaded_image_parts"]
    logger.debug("Callback: Cleared existing image/mask state keys.")

    for i, part in enumerate(user_content.parts):
        if hasattr(part, 'inline_data') and getattr(part.inline_data, 'mime_type', '').startswith('image/'):
            mime_type = part.inline_data.mime_type
            logger.debug(f"Callback: Found image part {i} with mime_type: {mime_type}")
            part_data_container = part.inline_data
            current_data_bytes = None

            raw_bytes = getattr(part_data_container, 'data', None)
            if raw_bytes and isinstance(raw_bytes, (bytes, bytearray)):
                current_data_bytes = raw_bytes
                logger.debug(f"Callback: Extracted raw bytes ({len(current_data_bytes)}) from inline_data for part {i}.")
            else:

                b64_data = getattr(part_data_container, 'b64_json', None)
                if b64_data and isinstance(b64_data, str):
                    logger.debug(f"Callback: Found base64 string in inline_data for part {i}. Decoding...")
                    try:
                        current_data_bytes = _decode_b64_str(b64_data)
                        logger.debug(f"Callback: Decoded base64 part {i}. Length: {len(current_data_bytes)}")
                    except Exception as e:
                        logger.error(f"Callback: Failed to decode base64 from part {i}: {e}")
                        continue # Skip this part if decoding fails
                else:
                    logger.warning(f"Callback: Image part {i} found but no suitable data/b64_json.")
                    continue

            if current_data_bytes:
                try:
                    current_b64_str = base64.b64encode(current_data_bytes).decode('utf-8')
                    image_part_info = {"b64": current_b64_str, "mime_type": mime_type}
                    image_parts_list.append(image_part_info)
                    logger.debug(f"Callback: Added image part {i} info (mime: {mime_type}, b64 len: {len(current_b64_str)}) to list.")
                    logger.info(f"Callback: Successfully processed image data from user_content for part {i}.") # Added confirmation log

                    if first_image_b64 is None:
                        first_image_b64 = current_b64_str
                        logger.debug(f"Callback: Storing part {i} as the first image for single-edit compatibility.")

                except Exception as e:
                    logger.error(f"Callback: Error processing/encoding image part {i}: {e}")
                    continue

    if image_parts_list:
        state["uploaded_image_parts"] = image_parts_list
        logger.info(f"Callback: Saved list of {len(image_parts_list)} image parts to state['uploaded_image_parts'].")
    else:
        logger.info("Callback: No valid image parts found to save to the list.")

    if first_image_b64:
        state["uploaded_image_b64"] = first_image_b64
        logger.info(f"Callback: Saved first image base64 string (len: {len(first_image_b64)}) to state['uploaded_image_b64'].")
    else:
        if "uploaded_image_b64" in state:
            del state["uploaded_image_b64"]
        logger.info("Callback: No first image found, ensuring state['uploaded_image_b64'] is clear.")

    logger.info("--- Exiting _save_uploaded_image_to_state_callback ---")

save_uploaded_image_to_state_tool = FunctionTool(func=_save_uploaded_image_to_state_callback) 


async def _image_save_func(image_bytes: bytes, file_extension: str, tool_context: ToolContext) -> Dict[str, Any]:
    logger.debug("Entering _image_save_func (artifact save version)...")

    if file_extension and not file_extension.startswith('.'):
        file_extension = '.' + file_extension
    mime_type = f"image/{file_extension.lstrip('.')}" if file_extension else "image/png"

    filename = f"generated_image_{uuid.uuid4()}{file_extension if file_extension else '.png'}" 
    local_path = None # Initialize local_path to None
    if settings.SAVE_LOCALLY:
        save_dir = settings.LOCAL_SAVE_PATH
        try:
            os.makedirs(save_dir, exist_ok=True)
            local_path = os.path.join(save_dir, filename) 
            
            logger.info(f"Attempting to save image locally to: {local_path} (SAVE_IMAGES_LOCALLY is True)")
            with open(local_path, "wb") as f:
                f.write(image_bytes)
            logger.info(f"Successfully saved image locally to: {local_path}")
        except Exception as e:
            logger.warning(f"Local file save failed (path: {save_dir}): {e}", exc_info=False)
    else:
        logger.debug("Local image saving skipped (SAVE_LOCALLY is False).")

    artifact_version = None
    local_path_if_saved = local_path if settings.SAVE_LOCALLY and local_path else None
    try:
        logger.info(f"Saving {len(image_bytes)} bytes as ADK artifact (name: {filename}, mime: {mime_type})")
        artifact_part = types.Part(inline_data=types.Blob(data=image_bytes, mime_type=mime_type))
        logger.debug(f"--->>> PRE-SAVE ARTIFACT CALL (sync) for {filename}")
        artifact_version = await tool_context.save_artifact( 
            filename=filename, 
            artifact=artifact_part 
        )
        logger.info(f"Successfully saved artifact {filename} (version {artifact_version})")
    except Exception as e:
        logger.error(f"Failed to save ADK artifact: {e}", exc_info=True)
        error_msg = f"Failed to save image as artifact: {e}"
        if local_path_if_saved:
             logger.warning(f"Artifact save failed, but local copy exists at {local_path_if_saved}")
             return {
                 "confirmation": f"Image saved locally to {local_path_if_saved}. Failed to save as artifact: {e}", 
                 "local_path": local_path_if_saved,
                 "artifact_error": str(e)
             }
        else:
             return {"error": error_msg}

    confirmation_msg = f"Image artifact {filename} (version {artifact_version}) saved successfully."
    if local_path_if_saved:
        confirmation_msg += f" Local copy saved to {local_path_if_saved}."
        
    logger.info("Returning confirmation and artifact info to agent.")
    return {
        "filename": filename, 
        "artifact_version": artifact_version, 
        "confirmation": confirmation_msg
    }

image_save_tool = FunctionTool(func=_image_save_func)

async def _enhance_prompt_for_image_gen(desc: str) -> Optional[str]:
    logger.debug(f"Enhancing prompt for description: '{desc}' with Gemini Flash")
    try:
       
        client = genai.Client(api_key=settings.GOOGLE_API_KEY)

        prompt_text = (
            "You are a creative image generation assistant. Enhance the following user request "
            f"into a detailed and vivid image generation prompt. User request: '{desc}'\n"
            "Examples:\n"
            f"- {image_prompt_examples.ENHANCE_PROMPT_CHARACTER[0]}\n"
            f"- {image_prompt_examples.ENHANCE_PROMPT_YOUTUBE_THUMBNAIL[0]}\n"
            "Return ONLY the enhanced prompt string."
        )

        logger.debug("Sending prompt to genai.generate_content...")
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt_text,
            config=types.GenerateContentConfig(
                temperature=0.3,
            )
        )
        logger.debug(f"Received response from genai.generate_content: {response}")

        detailed_prompt = response.text
        if not detailed_prompt:
             logger.error(f"Gemini prompt enhancement failed: No text in response: {response}")
             return None

        logger.info(f"Enhanced prompt for {desc}: '{detailed_prompt}'")
        log_prompt_to_file(prompt_text, response, detailed_prompt.strip())

        return detailed_prompt.strip() # Return the string
    except Exception as e:
        logger.error(f"Gemini prompt enhancement failed: {e}", exc_info=True)
        return None # Indicate failure

prompt_enhance_tool = FunctionTool(func=_enhance_prompt_for_image_gen)

async def _generate_image_with_gemini(desc: str, tool_context: ToolContext) -> Dict[str, Any]:
    logger.debug(f"--- Entering _generate_image_with_gemini for prompt: '{desc}' ---")
    
    try:
        client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-image", # Use the multimodal image generation model
            contents=[desc],
        )

        logger.debug(f"Received response from genai.generate_content (image gen).")

        generated_data_bytes = None
        
        # FIX: Explicitly access the content parts from the first candidate
        content_parts = response.candidates[0].content.parts if (response.candidates and response.candidates[0].content and response.candidates[0].content.parts) else []

        for part in content_parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                generated_data_bytes = part.inline_data.data
                mime_type = part.inline_data.mime_type
                logger.info(f"Extracted image data: {len(generated_data_bytes)} bytes, mime: {mime_type}")
                break

        if not generated_data_bytes:
            # Added logging for missing image data
            logger.error(f"Gemini image generation failed: No image data in response parts. Raw response: {response}")
            return {"error": "Image generation failed: No image data returned from Gemini."}

        artifact_result = await _image_save_func(generated_data_bytes, ".png", tool_context)
        
        logger.info(f"Gemini Generate artifact save result keys: {artifact_result.keys()}")
        if "error" in artifact_result or "artifact_error" in artifact_result:
            logger.info(f"INSIDE IF")
            error_key = "error" if "error" in artifact_result else "artifact_error"
            confirm = artifact_result.get("confirmation", "Image generated but artifact save failed.") + f" Error: {artifact_result.get(error_key)}"
            return {"confirmation": confirm, "artifact_error": artifact_result.get(error_key)}
        else:
            logger.info(f"ELSE")
            return artifact_result
            
    except Exception as e:
        logger.error(f"Gemini image generation failed: {e}", exc_info=True)
        # Added check for 'tuple' error specific to the previous iteration logic
        if "'tuple' object has no attribute 'inline_data'" in str(e):
             return {"error": "Image generation processing error: Response structure was invalid."}
        return {"error": str(e)}

gemini_image_generation_tool = FunctionTool(func=_generate_image_with_gemini)


async def _edit_image_with_gemini(prompt: str, tool_context: ToolContext, use_mask: bool = False) -> Dict[str, Any]:

    logger.debug(f"--- Entering _edit_image_with_gemini for prompt: '{prompt}', mask: {use_mask} ---")

    image_parts = tool_context.state.get("uploaded_image_parts")
    image_b64_str = None
    if image_parts and isinstance(image_parts, list) and len(image_parts) > 0 and isinstance(image_parts[0], dict):
        image_b64_str = image_parts[0].get("b64")
    
    if not image_b64_str:
        logger.error("Cannot edit image: Image data structure in state is missing or unexpected.")
        return {"error": "Cannot edit image: Please upload an image first."}

    try:
        raw_image_bytes = _decode_b64_str(image_b64_str)
        # Use PIL to open the image from bytes for the API
        input_image = Image.open(BytesIO(raw_image_bytes))
        logger.debug(f"Successfully loaded input image (size: {input_image.size}).")
    except Exception as e:
        logger.error(f"Failed to decode or process image from state: {e}", exc_info=True)
        return {"error": f"Failed to process image data stored in session state: {e}"}

    contents = [
        input_image, 
        f"Edit the image: {prompt}" 
    ]

    try:
        client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        
        logger.debug("Sending image edit request to genai.generate_content...")
        response = client.models.generate_content(
            model="gemini-2.5-flash-image", # The multimodal image model
            contents=contents,
        )

        generated_data_bytes = None
        # FIX: Explicitly access the content parts from the first candidate
        content_parts = response.candidates[0].content.parts if (response.candidates and response.candidates[0].content and response.candidates[0].content.parts) else []

        for part in content_parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                generated_data_bytes = part.inline_data.data
                mime_type = part.inline_data.mime_type
                logger.info(f"Extracted edited image data: {len(generated_data_bytes)} bytes, mime: {mime_type}")
                break 

        if not generated_data_bytes:
            logger.error("Gemini image editing failed: No image data in response parts.")
            return {"error": "Image editing failed: No image data returned from Gemini."}

        artifact_result = await _image_save_func(generated_data_bytes, "_edited.png", tool_context)
        
        logger.info(f"Edit artifact save result keys: {artifact_result.keys()}")

        if "error" in artifact_result or "artifact_error" in artifact_result:
            error_key = "error" if "error" in artifact_result else "artifact_error"
            confirm = artifact_result.get("confirmation", "Image edited but artifact save failed.") + f" Error: {artifact_result.get(error_key)}"
            return {"confirmation": confirm, "artifact_error": artifact_result.get(error_key)}
        else:
            return artifact_result
            
    except Exception as e:
        logger.error(f"Gemini image editing failed: {e}", exc_info=True)
        return {"error": str(e)}

gemini_image_edit_tool = FunctionTool(func=_edit_image_with_gemini)

def _save_image_artifact_to_state(filename: str, version: int, tool_context: ToolContext) -> Dict[str, str]:
    logger.info(f"--- Entering _save_image_artifact_to_state for {filename} v{version} ---")
    state = tool_context.state

    try:
        logger.debug(f"Attempting to load artifact: {filename} v{version}")
        # The tool_context.load_artifact returns a single Part object
        artifact_part = tool_context.load_artifact(filename, version=version)

        if not artifact_part:
            logger.error(f"Artifact {filename} v{version} not found.")
            return {"error": f"Artifact {filename} v{version} not found."}
        
        # --- ROBUST DATA EXTRACTION LOGIC ---
        image_bytes = None
        mime_type = "image/png" # Default in case metadata is missing

        # Helper function for recursive search
        def recursive_find_bytes(obj):
            if isinstance(obj, (bytes, bytearray)):
                return obj
            if isinstance(obj, dict):
                for key, val in obj.items():
                    if key in ('data', 'bytes'):
                        if isinstance(val, (bytes, bytearray)): return val
                    result = recursive_find_bytes(val)
                    if result is not None: return result
            elif hasattr(obj, '__dict__'):
                for key, val in obj.__dict__.items():
                    if key in ('data', 'bytes'):
                        if isinstance(val, (bytes, bytearray)): return val
                    result = recursive_find_bytes(val)
                    if result is not None: return result
            elif isinstance(obj, list) or isinstance(obj, tuple):
                 for val in obj:
                    result = recursive_find_bytes(val)
                    if result is not None: return result
            return None

        # Attempt 1: Standard Part structure check (types.Part object)
        if hasattr(artifact_part, 'inline_data') and hasattr(artifact_part.inline_data, 'data'):
             image_bytes = artifact_part.inline_data.data
             mime_type = artifact_part.inline_data.mime_type
             logger.debug("Successfully extracted data via standard types.Part structure.")
        
        # Attempt 2: If it's a simple dictionary (e.g., loaded from a wrapper)
        elif isinstance(artifact_part, dict) and 'data' in artifact_part:
            image_bytes = artifact_part['data']
            mime_type = artifact_part.get('mime_type', 'image/png')
            logger.debug("Successfully extracted data via dictionary check.")
        
        # Attempt 3: Check for direct 'data' attribute (last resort for ADK object weirdness)
        elif hasattr(artifact_part, 'data') and isinstance(getattr(artifact_part, 'data'), (bytes, bytearray)):
             image_bytes = artifact_part.data
             # Try to get mime_type from attributes/keys, default to png
             mime_type = getattr(artifact_part, 'mime_type', artifact_part.get('mime_type', 'image/png'))
             logger.debug("Successfully extracted data via direct data attribute check.")

        # Attempt 4: FINAL fail-safe deep attribute check for raw data
        if not image_bytes:
            image_bytes = recursive_find_bytes(artifact_part)
            if image_bytes:
                 logger.debug("Successfully extracted data via FINAL recursive deep attribute check.")
            
        # Attempt 5: Logging full object structure if all else fails
        if not image_bytes:
             logger.error(f"Artifact {filename} v{version} failed all data extraction attempts. Object type: {type(artifact_part)}, Content (truncated): {str(artifact_part)[:200]}...")


        if not image_bytes:
            logger.error(f"Artifact {filename} v{version} has invalid data structure. Could not find image bytes.")
            return {"error": f"Artifact {filename} v{version} has invalid data structure."}
        # --- END ROBUST DATA EXTRACTION LOGIC ---


        logger.debug(f"Loaded artifact data: {len(image_bytes)} bytes, mime: {mime_type}")

        image_b64_str = base64.b64encode(image_bytes).decode('utf-8')
        image_part_info = {"b64": image_b64_str, "mime_type": mime_type}
        image_parts_list = [image_part_info]

        state["uploaded_image_parts"] = image_parts_list
        state["uploaded_image_b64"] = image_b64_str
        confirmation_msg = f"Successfully loaded artifact {filename} v{version} and updated session state for editing."
        logger.info(confirmation_msg)
        return {"confirmation": confirmation_msg}

    except ValueError as e:
        logger.error(f"Error loading artifact {filename} v{version}: {e}", exc_info=True)
        return {"error": f"Error accessing artifact service: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error saving artifact {filename} v{version} to state: {e}", exc_info=True)
        return {"error": f"Unexpected error processing artifact: {e}"}

save_artifact_to_state_tool = FunctionTool(func=_save_image_artifact_to_state)

async def _clear_image_state(tool_context: ToolContext) -> Dict[str, Any]:
    logger.debug("--- Entering _clear_image_state tool ---")
    state = tool_context.state
    keys_to_clear = [
        "uploaded_image_b64",
        "uploaded_mask_b64",
        "uploaded_image_parts",
    ]
    cleared_keys_log = []

    for key in keys_to_clear:
        if state.get(key) is not None:
            state[key] = None 
            logger.debug(f"Cleared key '{key}' in session state by setting to None.")
            cleared_keys_log.append(key)
        else:
            logger.debug(f"Key '{key}' not found or already None in session state, skipping.")

    confirmation_msg = f"Image session state cleared. Cleared keys: {cleared_keys_log}" if cleared_keys_log else "Image session state was already clear (or no relevant keys found)."
    logger.info(confirmation_msg)
    return {"confirmation": confirmation_msg}

clear_image_state_tool = FunctionTool(func=_clear_image_state)

async def _restyle_image_with_gemini(style_description: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Applies a new artistic style to an uploaded image based on a text description.
    """
    logger.debug(f"--- Entering _restyle_image_with_gemini for style: '{style_description}' ---")

    # Retrieve the uploaded image from state
    image_parts = tool_context.state.get("uploaded_image_parts")
    image_b64_str = None
    image_mime_type = None

    if image_parts and isinstance(image_parts, list) and len(image_parts) > 0 and isinstance(image_parts[0], dict):
        first_part = image_parts[0]
        image_b64_str = first_part.get("b64")
        image_mime_type = first_part.get("mime_type") # Get mime type directly from the saved part

    if not image_b64_str:
        logger.error("Cannot restyle image: No image data found in session state.")
        return {"error": "Cannot restyle image: Please upload an image first."}

    try:
        raw_image_bytes = _decode_b64_str(image_b64_str)
        # Use PIL to open the image from bytes for the API
        input_image = Image.open(BytesIO(raw_image_bytes))
        logger.debug(f"Successfully loaded input image for restyling (size: {input_image.size}).")
    except Exception as e:
        logger.error(f"Failed to decode or process image from state for restyling: {e}", exc_info=True)
        return {"error": f"Failed to process image data from session state: {e}"}

    # Construct the prompt for re-styling
    contents = [
        input_image, 
        f"Re-style this image to look like: {style_description}. Do not change the subject, only apply the artistic style." 
    ]

    try:
        client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        
        logger.debug("Sending image restyle request to genai.generate_content...")
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
        )

        generated_data_bytes = None
        # FIX: Explicitly access the content parts from the first candidate
        content_parts = response.candidates[0].content.parts if (response.candidates and response.candidates[0].content and response.candidates[0].content.parts) else []

        for part in content_parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                generated_data_bytes = part.inline_data.data
                break 

        if not generated_data_bytes:
            logger.error("Gemini image restyling failed: No image data in response.")
            return {"error": "Image restyling failed: No image data returned from Gemini."}

        # Save the new image as an artifact
        artifact_result = await _image_save_func(generated_data_bytes, "_restyled.png", tool_context)
        
        return artifact_result
            
    except Exception as e:
        logger.error(f"Gemini image restyling failed: {e}", exc_info=True)
        return {"error": str(e)}


gemini_image_restyle_tool = FunctionTool(func=_restyle_image_with_gemini)