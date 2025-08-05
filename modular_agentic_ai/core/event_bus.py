"""
Event Bus - Async communication system for decoupled messaging between components.
Implements publish-subscribe pattern for event-driven architecture.
"""

import asyncio
import uuid
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """Represents an event in the system."""
    id: str
    type: str
    data: Dict[str, Any]
    source: str
    timestamp: datetime
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EventBus:
    """
    Central event bus for async communication between system components.
    Supports pub-sub pattern with filtering and priority handling.
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_queue = asyncio.Queue()
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._logger = logging.getLogger(__name__)
        self._event_history: List[Event] = []
        self._max_history_size = 1000
    
    async def start(self):
        """Start the event bus processor."""
        if not self._running:
            self._running = True
            self._processor_task = asyncio.create_task(self._process_events())
            self._logger.info("Event bus started")
    
    async def stop(self):
        """Stop the event bus processor."""
        if self._running:
            self._running = False
            if self._processor_task:
                await self._event_queue.put(None)  # Signal to stop processing
                await self._processor_task
            self._logger.info("Event bus stopped")
    
    def subscribe(self, event_type: str, handler: Callable[[Event], None]):
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to listen for
            handler: Async function to handle the event
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(handler)
        self._logger.debug(f"Handler subscribed to event type: {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable[[Event], None]):
        """
        Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of events to stop listening for
            handler: Handler function to remove
        """
        if event_type in self._subscribers and handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]
            self._logger.debug(f"Handler unsubscribed from event type: {event_type}")
    
    async def publish(self, event_type: str, data: Dict[str, Any], source: str, 
                     priority: EventPriority = EventPriority.NORMAL) -> str:
        """
        Publish an event to the bus.
        
        Args:
            event_type: Type of the event
            data: Event data payload
            source: Source component publishing the event
            priority: Event priority level
            
        Returns:
            Event ID
        """
        event = Event(
            id=str(uuid.uuid4()),
            type=event_type,
            data=data,
            source=source,
            timestamp=datetime.now(),
            priority=priority
        )
        
        await self._event_queue.put(event)
        self._logger.debug(f"Event published: {event_type} from {source}")
        return event.id
    
    async def _process_events(self):
        """Internal event processor running in background."""
        self._logger.info("Event processor started")
        
        while self._running:
            try:
                # Get event from queue
                event = await self._event_queue.get()
                
                # Check for stop signal
                if event is None:
                    break
                
                # Add to history
                self._add_to_history(event)
                
                # Process the event
                await self._handle_event(event)
                
                # Mark task as done
                self._event_queue.task_done()
                
            except Exception as e:
                self._logger.error(f"Error processing event: {e}")
    
    async def _handle_event(self, event: Event):
        """Handle a single event by notifying subscribers."""
        if event.type in self._subscribers:
            handlers = self._subscribers[event.type][:]  # Copy to avoid modification during iteration
            
            # Execute handlers concurrently
            tasks = []
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(asyncio.create_task(handler(event)))
                    else:
                        # Handle sync functions
                        tasks.append(asyncio.create_task(
                            asyncio.to_thread(handler, event)
                        ))
                except Exception as e:
                    self._logger.error(f"Error creating task for handler: {e}")
            
            # Wait for all handlers to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    def _add_to_history(self, event: Event):
        """Add event to history with size management."""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)
    
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """
        Get event history, optionally filtered by type.
        
        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return
            
        Returns:
            List of historical events
        """
        history = self._event_history
        
        if event_type:
            history = [e for e in history if e.type == event_type]
        
        return history[-limit:] if limit else history
    
    def get_subscriber_count(self, event_type: str) -> int:
        """Get number of subscribers for an event type."""
        return len(self._subscribers.get(event_type, []))
    
    def get_all_event_types(self) -> List[str]:
        """Get all event types that have subscribers."""
        return list(self._subscribers.keys())
    
    async def wait_for_queue_empty(self):
        """Wait until all events in queue are processed."""
        await self._event_queue.join()