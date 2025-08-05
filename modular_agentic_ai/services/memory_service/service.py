"""
Memory Service - Manages persistent storage and retrieval across the system.
Provides centralized memory management with multiple storage backends.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import uuid

from ...core.interfaces import MemoryInterface, MemoryEntry, MemoryQuery, MemorySearchResult
from ...core.event_bus import EventBus, EventPriority


class MemoryService:
    """
    Service for managing memory systems and providing unified memory interface.
    
    Responsibilities:
    - Multiple memory backend management
    - Memory routing and federation
    - Memory synchronization
    - Performance optimization
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self._logger = logging.getLogger(__name__)
        self._memory_backends: Dict[str, MemoryInterface] = {}
        self._memory_routing: Dict[str, str] = {}  # memory_type -> backend_id
        self._cache: Dict[str, MemoryEntry] = {}
        self._cache_ttl = timedelta(minutes=self.config.get('cache_ttl_minutes', 30))
        self._is_running = False
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> bool:
        """Initialize the memory service."""
        try:
            self._logger.info("Initializing Memory Service...")
            
            # Subscribe to relevant events
            self.event_bus.subscribe('memory.register', self._handle_memory_registration)
            self.event_bus.subscribe('memory.cleanup', self._handle_cleanup_request)
            
            # Start background cleanup task
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            self._is_running = True
            
            await self.event_bus.publish(
                'service.started',
                {'service': 'memory_service'},
                'memory_service',
                EventPriority.NORMAL
            )
            
            self._logger.info("Memory Service initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Memory Service initialization failed: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the memory service."""
        try:
            self._logger.info("Shutting down Memory Service...")
            
            # Cancel cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Disconnect all memory backends
            for backend_id, backend in self._memory_backends.items():
                await backend.disconnect()
            
            self._is_running = False
            
            await self.event_bus.publish(
                'service.stopped',
                {'service': 'memory_service'},
                'memory_service',
                EventPriority.NORMAL
            )
            
            self._logger.info("Memory Service shutdown completed")
            return True
            
        except Exception as e:
            self._logger.error(f"Error during Memory Service shutdown: {e}")
            return False
    
    async def register_memory_backend(self, backend_id: str, backend: MemoryInterface, 
                                    memory_types: List[str] = None) -> bool:
        """
        Register a memory backend with the service.
        
        Args:
            backend_id: Unique identifier for the backend
            backend: Memory backend instance
            memory_types: List of memory types this backend handles
            
        Returns:
            True if registration successful
        """
        try:
            if backend_id in self._memory_backends:
                self._logger.warning(f"Memory backend {backend_id} already registered")
                return False
            
            # Connect to the backend
            success = await backend.connect()
            if not success:
                self._logger.error(f"Failed to connect to memory backend {backend_id}")
                return False
            
            self._memory_backends[backend_id] = backend
            
            # Set up routing for memory types
            memory_types = memory_types or ['default']
            for memory_type in memory_types:
                self._memory_routing[memory_type] = backend_id
            
            await self.event_bus.publish(
                'memory.backend_registered',
                {
                    'backend_id': backend_id,
                    'memory_types': memory_types,
                    'backend_type': backend.get_memory_type()
                },
                'memory_service',
                EventPriority.NORMAL
            )
            
            self._logger.info(f"Memory backend {backend_id} registered successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Error registering memory backend {backend_id}: {e}")
            return False
    
    async def store_memory(self, content: Any, tags: List[str] = None, 
                          memory_type: str = 'default', 
                          importance_score: float = 0.5,
                          metadata: Dict[str, Any] = None) -> Optional[str]:
        """
        Store a memory entry.
        
        Args:
            content: Content to store
            tags: Associated tags
            memory_type: Type of memory for routing
            importance_score: Importance score (0.0 to 1.0)
            metadata: Additional metadata
            
        Returns:
            Memory entry ID if successful
        """
        try:
            # Get appropriate backend
            backend = self._get_backend_for_type(memory_type)
            if not backend:
                self._logger.error(f"No backend available for memory type: {memory_type}")
                return None
            
            # Create memory entry
            entry_id = str(uuid.uuid4())
            entry = MemoryEntry(
                id=entry_id,
                content=content,
                timestamp=datetime.now(),
                tags=tags or [],
                metadata=metadata or {},
                importance_score=importance_score
            )
            
            # Store in backend
            success = await backend.store(entry)
            if not success:
                self._logger.error(f"Failed to store memory entry {entry_id}")
                return None
            
            # Cache the entry
            self._cache[entry_id] = entry
            
            # Publish storage event
            await self.event_bus.publish(
                'memory.stored',
                {
                    'entry_id': entry_id,
                    'memory_type': memory_type,
                    'backend_id': self._memory_routing[memory_type],
                    'tags': tags,
                    'importance_score': importance_score
                },
                'memory_service'
            )
            
            self._logger.debug(f"Stored memory entry {entry_id}")
            return entry_id
            
        except Exception as e:
            self._logger.error(f"Error storing memory: {e}")
            return None
    
    async def retrieve_memory(self, entry_id: str, memory_type: str = 'default') -> Optional[MemoryEntry]:
        """
        Retrieve a specific memory entry.
        
        Args:
            entry_id: ID of the memory entry
            memory_type: Type of memory for routing
            
        Returns:
            Memory entry if found
        """
        try:
            # Check cache first
            if entry_id in self._cache:
                cached_entry = self._cache[entry_id]
                # Check if cache is still valid
                if datetime.now() - cached_entry.timestamp < self._cache_ttl:
                    return cached_entry
                else:
                    # Remove expired entry from cache
                    del self._cache[entry_id]
            
            # Get from backend
            backend = self._get_backend_for_type(memory_type)
            if not backend:
                return None
            
            entry = await backend.retrieve(entry_id)
            
            # Cache the retrieved entry
            if entry:
                self._cache[entry_id] = entry
            
            return entry
            
        except Exception as e:
            self._logger.error(f"Error retrieving memory {entry_id}: {e}")
            return None
    
    async def search_memory(self, query: str, memory_type: str = 'default', 
                          filters: Dict[str, Any] = None, limit: int = 10,
                          similarity_threshold: float = 0.7) -> MemorySearchResult:
        """
        Search memory entries.
        
        Args:
            query: Search query
            memory_type: Type of memory for routing
            filters: Additional filters
            limit: Maximum results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            Search results
        """
        try:
            backend = self._get_backend_for_type(memory_type)
            if not backend:
                return MemorySearchResult(
                    entries=[],
                    total_count=0,
                    search_time=0.0,
                    metadata={'error': f'No backend for memory type: {memory_type}'}
                )
            
            # Create search query
            memory_query = MemoryQuery(
                query=query,
                filters=filters or {},
                limit=limit,
                similarity_threshold=similarity_threshold
            )
            
            start_time = datetime.now()
            result = await backend.search(memory_query)
            search_time = (datetime.now() - start_time).total_seconds()
            
            # Update search time in result
            if result:
                result.search_time = search_time
            
            # Publish search event
            await self.event_bus.publish(
                'memory.searched',
                {
                    'query': query,
                    'memory_type': memory_type,
                    'results_count': len(result.entries) if result else 0,
                    'search_time': search_time
                },
                'memory_service'
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Error searching memory: {e}")
            return MemorySearchResult(
                entries=[],
                total_count=0,
                search_time=0.0,
                metadata={'error': str(e)}
            )
    
    async def update_memory(self, entry_id: str, updated_content: Any = None,
                          new_tags: List[str] = None, new_importance: float = None,
                          memory_type: str = 'default') -> bool:
        """
        Update an existing memory entry.
        
        Args:
            entry_id: ID of the entry to update
            updated_content: New content (optional)
            new_tags: New tags (optional)
            new_importance: New importance score (optional)
            memory_type: Type of memory for routing
            
        Returns:
            True if update successful
        """
        try:
            backend = self._get_backend_for_type(memory_type)
            if not backend:
                return False
            
            # Get current entry
            current_entry = await backend.retrieve(entry_id)
            if not current_entry:
                self._logger.error(f"Memory entry {entry_id} not found for update")
                return False
            
            # Create updated entry
            updated_entry = MemoryEntry(
                id=entry_id,
                content=updated_content if updated_content is not None else current_entry.content,
                timestamp=current_entry.timestamp,
                tags=new_tags if new_tags is not None else current_entry.tags,
                metadata=current_entry.metadata,
                importance_score=new_importance if new_importance is not None else current_entry.importance_score
            )
            
            # Update in backend
            success = await backend.update(entry_id, updated_entry)
            
            # Update cache
            if success and entry_id in self._cache:
                self._cache[entry_id] = updated_entry
            
            return success
            
        except Exception as e:
            self._logger.error(f"Error updating memory {entry_id}: {e}")
            return False
    
    async def delete_memory(self, entry_id: str, memory_type: str = 'default') -> bool:
        """
        Delete a memory entry.
        
        Args:
            entry_id: ID of the entry to delete
            memory_type: Type of memory for routing
            
        Returns:
            True if deletion successful
        """
        try:
            backend = self._get_backend_for_type(memory_type)
            if not backend:
                return False
            
            success = await backend.delete(entry_id)
            
            # Remove from cache
            if success and entry_id in self._cache:
                del self._cache[entry_id]
            
            if success:
                await self.event_bus.publish(
                    'memory.deleted',
                    {'entry_id': entry_id, 'memory_type': memory_type},
                    'memory_service'
                )
            
            return success
            
        except Exception as e:
            self._logger.error(f"Error deleting memory {entry_id}: {e}")
            return False
    
    def _get_backend_for_type(self, memory_type: str) -> Optional[MemoryInterface]:
        """Get the appropriate backend for a memory type."""
        backend_id = self._memory_routing.get(memory_type)
        if not backend_id:
            # Try default backend
            backend_id = self._memory_routing.get('default')
        
        return self._memory_backends.get(backend_id) if backend_id else None
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired cache entries."""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                current_time = datetime.now()
                expired_keys = []
                
                for entry_id, entry in self._cache.items():
                    if current_time - entry.timestamp > self._cache_ttl:
                        expired_keys.append(entry_id)
                
                for key in expired_keys:
                    del self._cache[key]
                
                if expired_keys:
                    self._logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
            except Exception as e:
                self._logger.error(f"Error during periodic cleanup: {e}")
    
    async def _handle_memory_registration(self, event):
        """Handle memory backend registration events."""
        self._logger.debug(f"Memory registration event: {event.data}")
    
    async def _handle_cleanup_request(self, event):
        """Handle cleanup request events."""
        self._logger.debug(f"Cleanup request event: {event.data}")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get memory service statistics."""
        backend_stats = {}
        for backend_id, backend in self._memory_backends.items():
            try:
                if hasattr(backend, 'get_memory_stats'):
                    backend_stats[backend_id] = asyncio.create_task(backend.get_memory_stats())
            except Exception as e:
                backend_stats[backend_id] = {'error': str(e)}
        
        return {
            'total_backends': len(self._memory_backends),
            'memory_routing': self._memory_routing,
            'cache_size': len(self._cache),
            'backend_stats': backend_stats
        }