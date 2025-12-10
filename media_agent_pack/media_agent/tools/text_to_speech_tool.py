import logging
import os
import uuid
from google.cloud import texttospeech
from google.adk.tools import ToolContext, FunctionTool
from google.genai import types
from . import settings

logger = logging.getLogger(__name__)

async def _audio_save_func(audio_bytes: bytes, tool_context: ToolContext) -> dict[str, any]:
    """Saves audio bytes as an ADK artifact and optionally locally."""
    logger.debug("Entering _audio_save_func...")
    
    file_extension = ".mp3"
    mime_type = "audio/mpeg"
    filename = f"generated_audio_{uuid.uuid4()}{file_extension}"

    # Save locally if enabled in settings
    if settings.SAVE_LOCALLY:
        save_dir = settings.LOCAL_SAVE_PATH
        try:
            os.makedirs(save_dir, exist_ok=True)
            local_path = os.path.join(save_dir, filename)
            
            logger.info(f"Attempting to save audio locally to: {local_path}")
            with open(local_path, "wb") as f:
                f.write(audio_bytes)
            logger.info(f"Successfully saved audio locally to: {local_path}")
        except Exception as e:
            logger.warning(f"Local audio file save failed: {e}", exc_info=False)

    # Save as ADK artifact
    try:
        logger.info(f"Saving {len(audio_bytes)} bytes as ADK audio artifact (name: {filename}, mime: {mime_type})")
        artifact_part = types.Part(inline_data=types.Blob(data=audio_bytes, mime_type=mime_type))
        artifact_version = await tool_context.save_artifact(
            filename=filename,
            artifact=artifact_part
        )
        logger.info(f"Successfully saved audio artifact {filename} (version {artifact_version})")
        
        confirmation_msg = f"Audio artifact {filename} (version {artifact_version}) saved successfully."
        return {
            "filename": filename,
            "artifact_version": artifact_version,
            "confirmation": confirmation_msg
        }
    except Exception as e:
        logger.error(f"Failed to save ADK audio artifact: {e}", exc_info=True)
        return {"error": f"Failed to save audio as artifact: {e}"}



async def _generate_audio_from_text(text: str, tool_context: ToolContext, voice_style: str = "neutral") -> dict[str, any]:
    """
    Generates audio from text, applying a specific voice style using supported SSML attributes.
    `voice_style` can be a general description like 'cheerful', 'sad', 'whispering'.
    """
    logger.debug(f"--- Entering _generate_audio_from_text with style: '{voice_style}' ---")

    try:
        client = texttospeech.TextToSpeechAsyncClient()

        # UPDATED SSML generation to remove the unsupported 'pitch' attribute.
        # We now rely on 'rate' and 'volume' for styling.
        if "cheerful" in voice_style.lower() or "happy" in voice_style.lower():
            ssml_text = f'<speak><prosody rate="fast">{text}</prosody></speak>'
        elif "sad" in voice_style.lower():
            ssml_text = f'<speak><prosody rate="slow">{text}</prosody></speak>'
        elif "whisper" in voice_style.lower():
             ssml_text = f'<speak><prosody rate="slow" volume="soft">{text}</prosody></speak>'
        else: # Neutral/default style
            ssml_text = f'<speak>{text}</speak>'

        synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", name="en-US-Studio-O"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        logger.debug("Sending request to Google Text-to-Speech API...")
        response = await client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        logger.debug("Received response from API.")
        
        audio_bytes = response.audio_content

        if not audio_bytes:
            logger.error("Text-to-Speech generation failed: No audio data in response.")
            return {"error": "Audio generation failed: No audio data returned from the API."}
        
        # Save the audio as an artifact
        artifact_result = await _audio_save_func(audio_bytes, tool_context)
        return artifact_result

    except Exception as e:
        logger.error(f"Google Text-to-Speech generation failed: {e}", exc_info=True)
        # Make the error message more user-friendly
        error_message = str(e)
        if "pitch" in error_message:
            error_message = "There was an error during audio generation related to unsupported voice attributes. The 'pitch' attribute is not supported for this voice."
        return {"error": error_message}

# This line should already be in your file, make sure it points to the updated function.
text_to_speech_tool = FunctionTool(func=_generate_audio_from_text)