"""ElevenLabs-compatible text-to-speech, served from local voices."""

from .api import create_app
from .settings import Settings

__all__ = ["create_app", "Settings"]
