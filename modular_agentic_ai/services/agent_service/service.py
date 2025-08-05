"""
Agent Service - Manages agent lifecycle, coordination, and communication.
Provides high-level interface for agent operations and multi-agent coordination.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ...core.interfaces import AgentInterface, AgentMessage, AgentResponse
from ...core.event_bus import EventBus, EventPriority


class AgentService:
    """
    Service for managing agents and their interactions.
    
    Responsibilities:
    - Agent lifecycle management
    - Inter-agent communication
    - Agent load balancing
    - Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self._logger = logging.getLogger(__name__)
        self._agents: Dict[str, AgentInterface] = {}
        self._agent_metrics: Dict[str, Dict[str, Any]] = {}
        self._communication_history: List[Dict[str, Any]] = []
        self._is_running = False
    
    async def initialize(self) -> bool:
        """Initialize the agent service."""
        try:
            self._logger.info("Initializing Agent Service...")
            
            # Subscribe to relevant events
            self.event_bus.subscribe('agent.register', self._handle_agent_registration)
            self.event_bus.subscribe('agent.communication', self._handle_agent_communication)
            
            self._is_running = True
            
            await self.event_bus.publish(
                'service.started',
                {'service': 'agent_service'},
                'agent_service',
                EventPriority.NORMAL
            )
            
            self._logger.info("Agent Service initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Agent Service initialization failed: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the agent service."""
        try:
            self._logger.info("Shutting down Agent Service...")
            
            # Shutdown all managed agents
            for agent_id, agent in self._agents.items():
                await self._shutdown_agent(agent_id)
            
            self._is_running = False
            
            await self.event_bus.publish(
                'service.stopped',
                {'service': 'agent_service'},
                'agent_service',
                EventPriority.NORMAL
            )
            
            self._logger.info("Agent Service shutdown completed")
            return True
            
        except Exception as e:
            self._logger.error(f"Error during Agent Service shutdown: {e}")
            return False
    
    async def register_agent(self, agent: AgentInterface) -> bool:
        """
        Register an agent with the service.
        
        Args:
            agent: Agent instance to register
            
        Returns:
            True if registration successful
        """
        try:
            if agent.agent_id in self._agents:
                self._logger.warning(f"Agent {agent.agent_id} already registered")
                return False
            
            # Initialize agent if needed
            if hasattr(agent, 'initialize'):
                success = await agent.initialize()
                if not success:
                    self._logger.error(f"Failed to initialize agent {agent.agent_id}")
                    return False
            
            self._agents[agent.agent_id] = agent
            self._agent_metrics[agent.agent_id] = {
                'registration_time': datetime.now(),
                'tasks_processed': 0,
                'success_rate': 0.0,
                'average_response_time': 0.0,
                'last_activity': datetime.now()
            }
            
            await self.event_bus.publish(
                'agent.registered',
                {
                    'agent_id': agent.agent_id,
                    'capabilities': agent.get_capabilities()
                },
                'agent_service',
                EventPriority.NORMAL
            )
            
            self._logger.info(f"Agent {agent.agent_id} registered successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Error registering agent {agent.agent_id}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the service.
        
        Args:
            agent_id: ID of agent to unregister
            
        Returns:
            True if unregistration successful
        """
        if agent_id not in self._agents:
            self._logger.warning(f"Agent {agent_id} not found for unregistration")
            return False
        
        await self._shutdown_agent(agent_id)
        del self._agents[agent_id]
        del self._agent_metrics[agent_id]
        
        await self.event_bus.publish(
            'agent.unregistered',
            {'agent_id': agent_id},
            'agent_service',
            EventPriority.NORMAL
        )
        
        self._logger.info(f"Agent {agent_id} unregistered")
        return True
    
    async def get_agent(self, agent_id: str) -> Optional[AgentInterface]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents with their info."""
        agents_info = []
        for agent_id, agent in self._agents.items():
            info = {
                'agent_id': agent_id,
                'capabilities': agent.get_capabilities(),
                'status': agent.get_status(),
                'metrics': self._agent_metrics.get(agent_id, {})
            }
            agents_info.append(info)
        return agents_info
    
    async def find_agents_by_capability(self, capability: str) -> List[AgentInterface]:
        """Find agents that have a specific capability."""
        suitable_agents = []
        for agent in self._agents.values():
            if capability in agent.get_capabilities():
                suitable_agents.append(agent)
        return suitable_agents
    
    async def execute_task_with_agent(self, agent_id: str, task: Dict[str, Any], 
                                    context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Execute a task with a specific agent.
        
        Args:
            agent_id: ID of the agent to use
            task: Task to execute
            context: Optional execution context
            
        Returns:
            Agent response
        """
        agent = self._agents.get(agent_id)
        if not agent:
            return AgentResponse(
                result=None,
                success=False,
                error_message=f"Agent {agent_id} not found"
            )
        
        try:
            start_time = datetime.now()
            
            # Execute task
            response = await agent.process_task(task, context)
            
            # Update metrics
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            metrics = self._agent_metrics[agent_id]
            metrics['tasks_processed'] += 1
            metrics['last_activity'] = end_time
            
            # Update success rate and average response time
            if response.success:
                current_avg = metrics['average_response_time']
                current_count = metrics['tasks_processed']
                metrics['average_response_time'] = (
                    (current_avg * (current_count - 1) + execution_time) / current_count
                )
            
            # Publish task execution event
            await self.event_bus.publish(
                'agent.task_executed',
                {
                    'agent_id': agent_id,
                    'task_type': task.get('type', 'unknown'),
                    'success': response.success,
                    'execution_time': execution_time
                },
                'agent_service'
            )
            
            return response
            
        except Exception as e:
            self._logger.error(f"Error executing task with agent {agent_id}: {e}")
            return AgentResponse(
                result=None,
                success=False,
                error_message=str(e)
            )
    
    async def facilitate_communication(self, sender_id: str, receiver_id: str, 
                                     message: AgentMessage) -> AgentResponse:
        """
        Facilitate communication between agents.
        
        Args:
            sender_id: ID of sending agent
            receiver_id: ID of receiving agent
            message: Message to send
            
        Returns:
            Response from receiving agent
        """
        receiver = self._agents.get(receiver_id)
        if not receiver:
            return AgentResponse(
                result=None,
                success=False,
                error_message=f"Receiver agent {receiver_id} not found"
            )
        
        try:
            # Log communication
            comm_record = {
                'timestamp': datetime.now(),
                'sender': sender_id,
                'receiver': receiver_id,
                'message_type': type(message.content).__name__,
                'success': None
            }
            
            # Send message to receiver
            response = await receiver.communicate(message)
            comm_record['success'] = response.success
            
            self._communication_history.append(comm_record)
            
            # Publish communication event
            await self.event_bus.publish(
                'agent.communication',
                {
                    'sender': sender_id,
                    'receiver': receiver_id,
                    'success': response.success
                },
                'agent_service'
            )
            
            return response
            
        except Exception as e:
            self._logger.error(f"Error in agent communication: {e}")
            return AgentResponse(
                result=None,
                success=False,
                error_message=str(e)
            )
    
    async def _shutdown_agent(self, agent_id: str):
        """Shutdown a specific agent."""
        agent = self._agents.get(agent_id)
        if agent and hasattr(agent, 'shutdown'):
            try:
                await agent.shutdown()
            except Exception as e:
                self._logger.error(f"Error shutting down agent {agent_id}: {e}")
    
    async def _handle_agent_registration(self, event):
        """Handle agent registration events."""
        self._logger.debug(f"Agent registration event: {event.data}")
    
    async def _handle_agent_communication(self, event):
        """Handle agent communication events."""
        self._logger.debug(f"Agent communication event: {event.data}")
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics."""
        return {
            'total_agents': len(self._agents),
            'active_agents': len([a for a in self._agents.values() if a.is_active]),
            'total_communications': len(self._communication_history),
            'agent_metrics': self._agent_metrics
        }