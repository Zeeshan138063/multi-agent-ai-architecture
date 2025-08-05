"""
Execution Engine - Central orchestrator for the modular agentic AI system.
Manages component lifecycle, coordinates execution flows, and handles system-wide operations.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass

from .event_bus import EventBus, EventPriority
from .registry import ServiceRegistry
from .interfaces import AgentInterface, ToolInterface, MemoryInterface


@dataclass
class ExecutionContext:
    """Context for execution sessions."""
    session_id: str
    start_time: datetime
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class TaskExecution:
    """Represents a task execution."""
    task_id: str
    task_type: str
    parameters: Dict[str, Any]
    assigned_agent: str
    status: str
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class ExecutionEngine:
    """
    Central orchestration engine for the modular agentic AI system.
    
    Responsibilities:
    - System initialization and shutdown
    - Component lifecycle management
    - Task execution coordination
    - Event-driven communication
    - Resource management
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.registry = ServiceRegistry()
        self.event_bus = EventBus()
        self._logger = logging.getLogger(__name__)
        self._is_running = False
        self._active_sessions: Dict[str, ExecutionContext] = {}
        self._task_queue = asyncio.Queue()
        self._task_processor: Optional[asyncio.Task] = None
        self._system_metrics = {
            'tasks_executed': 0,
            'active_sessions': 0,
            'errors_count': 0,
            'start_time': None
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the execution engine and all registered components.
        
        Returns:
            True if initialization successful
        """
        try:
            self._logger.info("Initializing Modular Agentic AI Engine...")
            
            # Start event bus
            await self.event_bus.start()
            
            # Initialize all registered components
            init_results = await self.registry.initialize_components()
            
            # Check if critical components initialized successfully
            failed_components = [cid for cid, success in init_results.items() if not success]
            if failed_components:
                self._logger.error(f"Failed to initialize components: {failed_components}")
                return False
            
            # Start task processor
            self._task_processor = asyncio.create_task(self._process_tasks())
            
            # Subscribe to system events
            self._setup_event_handlers()
            
            self._is_running = True
            self._system_metrics['start_time'] = datetime.now()
            
            # Publish system started event
            await self.event_bus.publish(
                'system.started',
                {'engine_id': id(self), 'components': list(init_results.keys())},
                'engine',
                EventPriority.HIGH
            )
            
            self._logger.info("Engine initialization completed successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Engine initialization failed: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the engine and all components.
        
        Returns:
            True if shutdown successful
        """
        try:
            self._logger.info("Shutting down Modular Agentic AI Engine...")
            
            # Mark as not running
            self._is_running = False
            
            # Publish shutdown event
            await self.event_bus.publish(
                'system.shutting_down',
                {'engine_id': id(self)},
                'engine',
                EventPriority.HIGH
            )
            
            # Stop task processor
            if self._task_processor and not self._task_processor.done():
                self._task_processor.cancel()
                try:
                    await self._task_processor
                except asyncio.CancelledError:
                    pass
            
            # Shutdown all components
            for component_id in list(self.registry._components.keys()):
                component = self.registry.get_component(component_id)
                if hasattr(component, 'shutdown'):
                    try:
                        if asyncio.iscoroutinefunction(component.shutdown):
                            await component.shutdown()
                        else:
                            component.shutdown()
                    except Exception as e:
                        self._logger.error(f"Error shutting down {component_id}: {e}")
                
                self.registry.unregister_component(component_id)
            
            # Stop event bus
            await self.event_bus.stop()
            
            self._logger.info("Engine shutdown completed")
            return True
            
        except Exception as e:
            self._logger.error(f"Error during engine shutdown: {e}")
            return False
    
    async def execute_task(self, task_type: str, parameters: Dict[str, Any], 
                          session_id: Optional[str] = None, 
                          preferred_agent: Optional[str] = None) -> TaskExecution:
        """
        Execute a task using the available agents and tools.
        
        Args:
            task_type: Type of task to execute
            parameters: Task parameters
            session_id: Session ID for context
            preferred_agent: Preferred agent for execution
            
        Returns:
            TaskExecution result
        """
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Create task execution record
        task_exec = TaskExecution(
            task_id=task_id,
            task_type=task_type,
            parameters=parameters,
            assigned_agent="",
            status="pending",
            start_time=datetime.now()
        )
        
        try:
            # Find suitable agent
            agent = await self._find_suitable_agent(task_type, preferred_agent)
            if not agent:
                task_exec.status = "failed"
                task_exec.error = "No suitable agent found"
                task_exec.end_time = datetime.now()
                return task_exec
            
            task_exec.assigned_agent = agent.agent_id
            task_exec.status = "running"
            
            # Publish task started event
            await self.event_bus.publish(
                'task.started',
                {
                    'task_id': task_id,
                    'task_type': task_type,
                    'agent_id': agent.agent_id,
                    'session_id': session_id
                },
                'engine'
            )
            
            # Execute task
            context = {'session_id': session_id, 'engine': self}
            response = await agent.process_task(parameters, context)
            
            # Update task execution
            task_exec.result = response.result
            task_exec.status = "completed" if response.success else "failed"
            task_exec.error = response.error_message
            task_exec.end_time = datetime.now()
            
            # Update metrics
            self._system_metrics['tasks_executed'] += 1
            if not response.success:
                self._system_metrics['errors_count'] += 1
            
            # Publish task completed event
            await self.event_bus.publish(
                'task.completed',
                {
                    'task_id': task_id,
                    'success': response.success,
                    'agent_id': agent.agent_id,
                    'execution_time': (task_exec.end_time - task_exec.start_time).total_seconds()
                },
                'engine'
            )
            
        except Exception as e:
            task_exec.status = "error"
            task_exec.error = str(e)
            task_exec.end_time = datetime.now()
            self._system_metrics['errors_count'] += 1
            self._logger.error(f"Task execution error: {e}")
        
        return task_exec
    
    async def _find_suitable_agent(self, task_type: str, preferred_agent: Optional[str] = None) -> Optional[AgentInterface]:
        """Find a suitable agent for the given task type."""
        agents = self.registry.get_components_by_type('agents')
        
        # If preferred agent specified, try to use it
        if preferred_agent:
            agent = self.registry.get_component(preferred_agent)
            if agent and isinstance(agent, AgentInterface):
                capabilities = agent.get_capabilities()
                if task_type in capabilities:
                    return agent
        
        # Find any suitable agent
        for agent in agents:
            if isinstance(agent, AgentInterface):
                capabilities = agent.get_capabilities()
                if task_type in capabilities:
                    return agent
        
        return None
    
    async def _process_tasks(self):
        """Background task processor."""
        while self._is_running:
            try:
                # Process any queued tasks
                await asyncio.sleep(0.1)  # Prevent busy waiting
            except Exception as e:
                self._logger.error(f"Task processor error: {e}")
    
    def _setup_event_handlers(self):
        """Set up internal event handlers."""
        
        async def handle_component_error(event):
            self._logger.error(f"Component error: {event.data}")
            self._system_metrics['errors_count'] += 1
        
        async def handle_agent_communication(event):
            self._logger.debug(f"Agent communication: {event.data}")
        
        # Subscribe to important events
        self.event_bus.subscribe('component.error', handle_component_error)
        self.event_bus.subscribe('agent.communication', handle_agent_communication)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'is_running': self._is_running,
            'components': self.registry.get_registry_stats(),
            'metrics': self._system_metrics,
            'active_sessions': len(self._active_sessions),
            'event_types': self.event_bus.get_all_event_types()
        }
    
    def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get information about available agents."""
        agents = self.registry.get_components_by_type('agents')
        return [
            {
                'agent_id': agent.agent_id,
                'capabilities': agent.get_capabilities(),
                'status': agent.get_status()
            }
            for agent in agents if isinstance(agent, AgentInterface)
        ]
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get information about available tools."""
        tools = self.registry.get_components_by_type('tools')
        return [
            tool.get_tool_info()
            for tool in tools if isinstance(tool, ToolInterface)
        ]