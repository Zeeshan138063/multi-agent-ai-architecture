"""
Agent Interface - Abstract base class for all agents in the system.
Ensures all agents follow the same contract for consistency and hot-swapping.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class AgentMessage:
    """Represents a message in agent communication."""
    content: str
    sender: str
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class AgentResponse:
    """Represents an agent's response to a task."""
    result: Any
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class AgentInterface(ABC):
    """Abstract interface that all agents must implement."""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.is_active = True
    
    @abstractmethod
    async def process_task(self, task: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process a given task and return a response."""
        pass
    
    @abstractmethod
    async def communicate(self, message: AgentMessage) -> AgentResponse:
        """Handle communication from other agents."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this agent provides."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return current agent status and health."""
        pass
    
    async def initialize(self) -> bool:
        """Initialize the agent. Override if needed."""
        return True
    
    async def shutdown(self) -> bool:
        """Cleanup when agent is being stopped. Override if needed."""
        self.is_active = False
        return True