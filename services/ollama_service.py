"""
Ollama Service - Handles all ML/AI interactions with Ollama
"""
import requests
import logging
from typing import List
from config import OLLAMA_URL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)


class OllamaService:
    """Service for interacting with Ollama API"""
    
    def __init__(self, base_url: str = OLLAMA_URL, timeout: int = OLLAMA_TIMEOUT):
        """
        Initialize Ollama service
        
        Args:
            base_url: Ollama API endpoint
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout

    def _tags_url(self) -> str:
        """Build Ollama /api/tags endpoint from configured base URL."""
        return self.base_url.replace("/api/generate", "/api/tags")

    def _chat_url(self) -> str:
        """Build Ollama /api/chat endpoint from configured base URL."""
        return self.base_url.replace("/api/generate", "/api/chat")

    @staticmethod
    def _extract_text(payload: dict) -> str:
        """Extract model text from either generate or chat response shapes."""
        text = payload.get("response", "")
        if isinstance(text, str) and text.strip():
            return text.strip()

        message = payload.get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else ""
        if isinstance(content, str):
            return content.strip()

        return ""
    
    def call_model(self, prompt_text: str, model_name: str, temperature: float = 0.3) -> str:
        """
        Call Ollama model with a prompt
        
        Args:
            prompt_text: The prompt to send to the model
            model_name: Name of the model to use
            temperature: Temperature parameter (0.0 to 1.0)
            
        Returns:
            Model response as string
            
        Raises:
            requests.exceptions.ConnectionError: If Ollama is not running
            requests.exceptions.HTTPError: If API returns an error
        """
        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature
            }
        }
        chat_payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt_text}
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(self.base_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            generate_json = response.json()
            generate_text = self._extract_text(generate_json)
            if generate_text:
                return generate_text

            # Some model builds return empty text on /api/generate but work with /api/chat.
            logger.warning(
                f"Empty response from /api/generate for model '{model_name}', trying /api/chat"
            )
            chat_response = requests.post(self._chat_url(), json=chat_payload, timeout=self.timeout)
            chat_response.raise_for_status()
            chat_json = chat_response.json()
            chat_text = self._extract_text(chat_json)
            if chat_text:
                return chat_text

            logger.warning(
                f"Ollama returned empty output for model '{model_name}' on both generate and chat APIs"
            )
            return ""
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise
    
    def check_connection(self) -> bool:
        """
        Check if Ollama service is running
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(self._tags_url(), timeout=3)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> List[str]:
        """
        Get available models from local Ollama server.

        Returns:
            List of model names, empty list if unavailable
        """
        try:
            response = requests.get(self._tags_url(), timeout=5)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models", [])
            model_names = [m.get("name", "") for m in models if m.get("name")]
            return sorted(model_names)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not fetch Ollama models: {e}")
            return []


# Create singleton instance
ollama_service = OllamaService()
