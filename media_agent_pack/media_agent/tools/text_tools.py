import logging
from google import genai
from . import settings
from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

def _restyle_text(original_text: str, style_description: str) -> str:
    """
    Rewrites the original text to match the given style description.
    """
    logger.debug(f"Restyling text to style: '{style_description}'")
    try:
        client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        
        prompt = (
            f"You are an expert writer. Rewrite the following text to match the specified style.\n\n"
            f"**Style:**\n{style_description}\n\n"
            f"**Original Text:**\n{original_text}\n\n"
            f"**Rewritten Text:**"
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt,
            config=genai.types.GenerateContentConfig(temperature=0.7)
        )
        
        rewritten_text = response.text
        logger.info("Successfully restyled text.")
        return rewritten_text

    except Exception as e:
        logger.error(f"Text restyling failed: {e}", exc_info=True)
        return f"Error: Could not restyle the text. {e}"

text_restyle_tool = FunctionTool(func=_restyle_text)
