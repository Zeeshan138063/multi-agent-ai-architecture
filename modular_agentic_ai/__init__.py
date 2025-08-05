"""
Modular Agentic AI Architecture

A comprehensive, plugin-based architecture for building scalable agentic AI systems.
This package provides a modular framework with hot-swappable components including
agents, tools, memory systems, and model adapters.

Key Features:
- Plugin-based architecture with interface-driven contracts
- Service-oriented design with loose coupling
- Event-driven communication via internal event bus
- Hot-swappable components for flexibility
- Multi-model support through adapters
- Comprehensive memory systems (vector and graph)
- Built-in tools for web search, code execution, and API integration

Components:
- Core: Engine, Registry, Event Bus, Interfaces
- Services: Agent Service, Memory Service, Planning Service
- Plugins: Agents, Tools, Memory Systems
- Adapters: OpenAI, Anthropic, Local Models

Author: Zeehsan (PS.Brij Kishore Pandey)
License: MIT
"""

from .core.engine import ExecutionEngine, ExecutionContext, TaskExecution
from .core.registry import ServiceRegistry, ComponentInfo
from .core.event_bus import EventBus, Event, EventPriority
from .core.interfaces import (
    AgentInterface, AgentMessage, AgentResponse,
    ToolInterface, ToolResult, ToolParameter, ToolExecutionStatus,
    MemoryInterface, MemoryEntry, MemoryQuery, MemorySearchResult
)

__version__ = "1.0.0"
__author__ = "Zeehsan (PS.Brij Kishore Pandey)"
__license__ = "MIT"

__all__ = [
    # Core components
    'ExecutionEngine', 'ExecutionContext', 'TaskExecution',
    'ServiceRegistry', 'ComponentInfo',
    'EventBus', 'Event', 'EventPriority',
    
    # Interfaces
    'AgentInterface', 'AgentMessage', 'AgentResponse',
    'ToolInterface', 'ToolResult', 'ToolParameter', 'ToolExecutionStatus',
    'MemoryInterface', 'MemoryEntry', 'MemoryQuery', 'MemorySearchResult',
    
    # Package info
    '__version__', '__author__', '__license__'
]


def get_version():
    """Get the package version."""
    return __version__


def get_system_info():
    """Get comprehensive system information."""
    return {
        'name': 'Modular Agentic AI Architecture',
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'description': 'Plugin-based architecture for scalable agentic AI systems',
        'components': {
            'core': ['Engine', 'Registry', 'Event Bus', 'Interfaces'],
            'services': ['Agent Service', 'Memory Service', 'Planning Service'],
            'plugins': ['Agents', 'Tools', 'Memory Systems'],
            'adapters': ['OpenAI', 'Anthropic', 'Local Models']
        },
        'features': [
            'Plugin-based architecture',
            'Hot-swappable components',
            'Event-driven communication',
            'Multi-model support',
            'Comprehensive memory systems',
            'Built-in tools and utilities'
        ]
    }