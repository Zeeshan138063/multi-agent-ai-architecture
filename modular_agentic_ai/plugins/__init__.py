"""
Plugins package - Contains all plugin implementations for the modular AI system.

This package includes:
- Agents: Reasoning, Planning, and Execution agents
- Tools: Web Search, Code Execution, and API Client tools
- Memory: Vector and Graph memory systems

All plugins follow the interface contracts defined in core.interfaces.
"""

# Import plugin categories
from . import agents
from . import tools  
from . import memory

__all__ = ['agents', 'tools', 'memory']