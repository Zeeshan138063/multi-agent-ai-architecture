"""
Adapters package - Contains model adapter implementations.

Available adapters:
- OpenAIAdapter: Integration with OpenAI's API services
- AnthropicAdapter: Integration with Anthropic's Claude API
- LocalModelAdapter: Support for local AI models and inference engines
"""

from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .local_model_adapter import LocalModelAdapter

__all__ = ['OpenAIAdapter', 'AnthropicAdapter', 'LocalModelAdapter']