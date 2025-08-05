"""
Memory Interface - Abstract base class for all memory systems in the architecture.
Provides persistent storage and retrieval capabilities for agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemoryEntry:
    """Represents a single memory entry."""
    id: str
    content: Any
    timestamp: datetime
    tags: List[str]
    metadata: Dict[str, Any]
    importance_score: float = 0.5


@dataclass
class MemoryQuery:
    """Represents a memory search query."""
    query: str
    filters: Dict[str, Any] = None
    limit: int = 10
    similarity_threshold: float = 0.7


@dataclass
class MemorySearchResult:
    """Result of memory search operation."""
    entries: List[MemoryEntry]
    total_count: int
    search_time: float
    metadata: Dict[str, Any] = None


class MemoryInterface(ABC):
    """Abstract interface that all memory systems must implement."""
    
    def __init__(self, memory_id: str, config: Dict[str, Any]):
        self.memory_id = memory_id
        self.config = config
        self.is_connected = False
    
    @abstractmethod
    async def store(self, entry: MemoryEntry) -> bool:
        """Store a memory entry."""
        pass
    
    @abstractmethod
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry by ID."""
        pass
    
    @abstractmethod
    async def search(self, query: MemoryQuery) -> MemorySearchResult:
        """Search memory entries based on query."""
        pass
    
    @abstractmethod
    async def update(self, entry_id: str, updated_entry: MemoryEntry) -> bool:
        """Update an existing memory entry."""
        pass
    
    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        pass
    
    @abstractmethod
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        pass
    
    @abstractmethod
    async def create_embedding(self, content: str) -> List[float]:
        """Create vector embedding for content (if supported)."""
        pass
    
    async def connect(self) -> bool:
        """Connect to the memory system. Override if needed."""
        self.is_connected = True
        return True
    
    async def disconnect(self) -> bool:
        """Disconnect from the memory system. Override if needed."""
        self.is_connected = False
        return True
    
    def get_memory_type(self) -> str:
        """Return the type of memory system."""
        return self.__class__.__name__