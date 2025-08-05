"""
Tool Interface - Abstract base class for all tools in the system.
Tools are utilities that agents can use to perform specific tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class ToolExecutionStatus(Enum):
    """Status of tool execution."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"


@dataclass
class ToolResult:
    """Result of tool execution."""
    status: ToolExecutionStatus
    data: Any
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str
    required: bool
    description: str
    default_value: Any = None


class ToolInterface(ABC):
    """Abstract interface that all tools must implement."""
    
    def __init__(self, tool_id: str, config: Dict[str, Any]):
        self.tool_id = tool_id
        self.config = config
        self.is_available = True
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """Return list of parameters this tool accepts."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return description of what this tool does."""
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate if parameters are correct for this tool."""
        pass
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Return comprehensive tool information."""
        return {
            "tool_id": self.tool_id,
            "description": self.get_description(),
            "parameters": [param.__dict__ for param in self.get_parameters()],
            "is_available": self.is_available,
            "config": self.config
        }
    
    async def initialize(self) -> bool:
        """Initialize the tool. Override if needed."""
        return True
    
    async def cleanup(self) -> bool:
        """Cleanup tool resources. Override if needed."""
        return True