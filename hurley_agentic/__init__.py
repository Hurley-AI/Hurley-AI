"""
hurley_agentic package.
"""

from .agent import Agent
from .tools import HurleyToolFactory, HurleyTool

# Define the __all__ variable for wildcard imports
__all__ = ['Agent', 'HurleyToolFactory', 'HurleyTool']

# Ensure package version is available
try:
    import importlib.metadata
    __version__ = importlib.metadata.version("hurley_agentic")
except Exception:
    __version__ = "0.0.0"  # fallback if not installed
