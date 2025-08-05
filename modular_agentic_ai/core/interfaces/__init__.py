"""
Core interfaces package - Defines contracts for all system components.
"""

from .agent_interface import AgentInterface, AgentMessage, AgentResponse
from .tool_interface import ToolInterface, ToolResult, ToolParameter, ToolExecutionStatus
from .memory_interface import MemoryInterface, MemoryEntry, MemoryQuery, MemorySearchResult

__all__ = [
    'AgentInterface', 'AgentMessage', 'AgentResponse',
    'ToolInterface', 'ToolResult', 'ToolParameter', 'ToolExecutionStatus', 
    'MemoryInterface', 'MemoryEntry', 'MemoryQuery', 'MemorySearchResult'
]