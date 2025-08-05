"""
Planning Service - Handles task planning, decomposition, and execution orchestration.
Provides intelligent task management and multi-agent coordination planning.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid

from ...core.event_bus import EventBus, EventPriority


class TaskStatus(Enum):
    """Status of task execution."""
    PENDING = "pending"
    PLANNING = "planning"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a task in the planning system."""
    task_id: str
    name: str
    description: str
    task_type: str
    parameters: Dict[str, Any]
    priority: TaskPriority
    status: TaskStatus
    dependencies: List[str]
    assigned_agent: Optional[str] = None
    estimated_duration: Optional[timedelta] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionPlan:
    """Represents an execution plan for a complex task."""
    plan_id: str
    name: str
    description: str
    tasks: List[Task]
    execution_order: List[str]  # Task IDs in execution order
    estimated_total_duration: timedelta
    created_at: datetime
    status: str = "draft"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PlanningService:
    """
    Service for intelligent task planning and execution orchestration.
    
    Responsibilities:
    - Task decomposition and planning
    - Dependency resolution
    - Resource allocation
    - Execution monitoring
    - Plan optimization
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self._logger = logging.getLogger(__name__)
        self._tasks: Dict[str, Task] = {}
        self._plans: Dict[str, ExecutionPlan] = {}
        self._active_executions: Dict[str, asyncio.Task] = {}
        self._task_queue = asyncio.PriorityQueue()
        self._is_running = False
        self._executor_task: Optional[asyncio.Task] = None
        self._planning_strategies: Dict[str, callable] = {}
        
        # Initialize built-in planning strategies
        self._register_built_in_strategies()
    
    async def initialize(self) -> bool:
        """Initialize the planning service."""
        try:
            self._logger.info("Initializing Planning Service...")
            
            # Subscribe to relevant events
            self.event_bus.subscribe('task.created', self._handle_task_created)
            self.event_bus.subscribe('task.completed', self._handle_task_completed)
            self.event_bus.subscribe('agent.available', self._handle_agent_available)
            
            # Start the task executor
            self._executor_task = asyncio.create_task(self._execute_tasks())
            
            self._is_running = True
            
            await self.event_bus.publish(
                'service.started',
                {'service': 'planning_service'},
                'planning_service',
                EventPriority.NORMAL
            )
            
            self._logger.info("Planning Service initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Planning Service initialization failed: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the planning service."""
        try:
            self._logger.info("Shutting down Planning Service...")
            
            self._is_running = False
            
            # Cancel executor task
            if self._executor_task and not self._executor_task.done():
                self._executor_task.cancel()
                try:
                    await self._executor_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel active executions
            for task_id, execution_task in self._active_executions.items():
                if not execution_task.done():
                    execution_task.cancel()
                    self._tasks[task_id].status = TaskStatus.CANCELLED
            
            await self.event_bus.publish(
                'service.stopped',
                {'service': 'planning_service'},
                'planning_service',
                EventPriority.NORMAL
            )
            
            self._logger.info("Planning Service shutdown completed")
            return True
            
        except Exception as e:
            self._logger.error(f"Error during Planning Service shutdown: {e}")
            return False
    
    async def create_task(self, name: str, description: str, task_type: str,
                         parameters: Dict[str, Any], priority: TaskPriority = TaskPriority.NORMAL,
                         dependencies: List[str] = None, 
                         estimated_duration: Optional[timedelta] = None) -> str:
        """
        Create a new task.
        
        Args:
            name: Task name
            description: Task description
            task_type: Type of task
            parameters: Task parameters
            priority: Task priority
            dependencies: List of task IDs this task depends on
            estimated_duration: Estimated task duration
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            name=name,
            description=description,
            task_type=task_type,
            parameters=parameters,
            priority=priority,
            status=TaskStatus.PENDING,
            dependencies=dependencies or [],
            estimated_duration=estimated_duration
        )
        
        self._tasks[task_id] = task
        
        await self.event_bus.publish(
            'task.created',
            {
                'task_id': task_id,
                'task_type': task_type,
                'priority': priority.name,
                'has_dependencies': len(task.dependencies) > 0
            },
            'planning_service'
        )
        
        # Add to task queue if ready
        if self._are_dependencies_met(task_id):
            await self._queue_task(task_id)
        
        self._logger.info(f"Created task {task_id}: {name}")
        return task_id
    
    async def create_execution_plan(self, name: str, description: str, 
                                  complex_task: Dict[str, Any],
                                  strategy: str = "sequential") -> str:
        """
        Create an execution plan for a complex task.
        
        Args:
            name: Plan name
            description: Plan description
            complex_task: Complex task to decompose
            strategy: Planning strategy to use
            
        Returns:
            Plan ID
        """
        plan_id = str(uuid.uuid4())
        
        try:
            # Use planning strategy to decompose the task
            strategy_func = self._planning_strategies.get(strategy, self._sequential_planning)
            tasks, execution_order, estimated_duration = await strategy_func(complex_task)
            
            plan = ExecutionPlan(
                plan_id=plan_id,
                name=name,
                description=description,
                tasks=tasks,
                execution_order=execution_order,
                estimated_total_duration=estimated_duration,
                created_at=datetime.now()
            )
            
            self._plans[plan_id] = plan
            
            # Add tasks to the task system
            for task in tasks:
                self._tasks[task.task_id] = task
            
            await self.event_bus.publish(
                'plan.created',
                {
                    'plan_id': plan_id,
                    'task_count': len(tasks),
                    'strategy': strategy,
                    'estimated_duration': estimated_duration.total_seconds()
                },
                'planning_service'
            )
            
            self._logger.info(f"Created execution plan {plan_id} with {len(tasks)} tasks")
            return plan_id
            
        except Exception as e:
            self._logger.error(f"Error creating execution plan: {e}")
            raise
    
    async def execute_plan(self, plan_id: str) -> bool:
        """
        Execute an execution plan.
        
        Args:
            plan_id: ID of the plan to execute
            
        Returns:
            True if execution started successfully
        """
        plan = self._plans.get(plan_id)
        if not plan:
            self._logger.error(f"Plan {plan_id} not found")
            return False
        
        plan.status = "executing"
        
        # Queue tasks according to execution order
        for task_id in plan.execution_order:
            if task_id in self._tasks and self._are_dependencies_met(task_id):
                await self._queue_task(task_id)
        
        await self.event_bus.publish(
            'plan.execution_started',
            {
                'plan_id': plan_id,
                'task_count': len(plan.tasks)
            },
            'planning_service'
        )
        
        self._logger.info(f"Started execution of plan {plan_id}")
        return True
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task."""
        task = self._tasks.get(task_id)
        if not task:
            return None
        
        return {
            'task_id': task.task_id,
            'name': task.name,
            'status': task.status.value,
            'priority': task.priority.name,
            'assigned_agent': task.assigned_agent,
            'created_at': task.created_at.isoformat() if task.created_at else None,
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'dependencies_met': self._are_dependencies_met(task_id),
            'result': task.result,
            'error_message': task.error_message
        }
    
    async def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an execution plan."""
        plan = self._plans.get(plan_id)
        if not plan:
            return None
        
        # Calculate progress
        total_tasks = len(plan.tasks)
        completed_tasks = sum(1 for task in plan.tasks if task.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for task in plan.tasks if task.status == TaskStatus.FAILED)
        
        return {
            'plan_id': plan.plan_id,
            'name': plan.name,
            'status': plan.status,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'progress_percentage': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            'estimated_duration': plan.estimated_total_duration.total_seconds(),
            'created_at': plan.created_at.isoformat()
        }
    
    def _are_dependencies_met(self, task_id: str) -> bool:
        """Check if all dependencies for a task are met."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        for dep_id in task.dependencies:
            dep_task = self._tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    async def _queue_task(self, task_id: str):
        """Add a task to the execution queue."""
        task = self._tasks[task_id]
        task.status = TaskStatus.READY
        
        # Priority queue uses negative values for higher priority
        priority_value = -task.priority.value
        await self._task_queue.put((priority_value, task_id))
    
    async def _execute_tasks(self):
        """Background task executor."""
        while self._is_running:
            try:
                # Get next task from queue
                priority, task_id = await asyncio.wait_for(
                    self._task_queue.get(), timeout=1.0
                )
                
                task = self._tasks.get(task_id)
                if not task or task.status != TaskStatus.READY:
                    continue
                
                # Start task execution
                execution_task = asyncio.create_task(self._execute_single_task(task_id))
                self._active_executions[task_id] = execution_task
                
            except asyncio.TimeoutError:
                # No tasks in queue, continue
                continue
            except Exception as e:
                self._logger.error(f"Error in task executor: {e}")
    
    async def _execute_single_task(self, task_id: str):
        """Execute a single task."""
        task = self._tasks[task_id]
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        try:
            # Publish task started event
            await self.event_bus.publish(
                'task.started',
                {
                    'task_id': task_id,
                    'task_type': task.task_type,
                    'assigned_agent': task.assigned_agent
                },
                'planning_service'
            )
            
            # Here would be the actual task execution logic
            # This would involve calling the appropriate agent or service
            # For now, we'll simulate task execution
            await asyncio.sleep(1)  # Simulate work
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = {"status": "success", "message": "Task completed successfully"}
            
            # Check if any dependent tasks can now be executed
            await self._check_dependent_tasks(task_id)
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error_message = str(e)
            self._logger.error(f"Task {task_id} failed: {e}")
        
        finally:
            # Remove from active executions
            if task_id in self._active_executions:
                del self._active_executions[task_id]
            
            # Publish task completed event
            await self.event_bus.publish(
                'task.completed',
                {
                    'task_id': task_id,
                    'status': task.status.value,
                    'success': task.status == TaskStatus.COMPLETED
                },
                'planning_service'
            )
    
    async def _check_dependent_tasks(self, completed_task_id: str):
        """Check and queue tasks that depend on the completed task."""
        for task_id, task in self._tasks.items():
            if (completed_task_id in task.dependencies and 
                task.status == TaskStatus.PENDING and 
                self._are_dependencies_met(task_id)):
                await self._queue_task(task_id)
    
    def _register_built_in_strategies(self):
        """Register built-in planning strategies."""
        self._planning_strategies['sequential'] = self._sequential_planning
        self._planning_strategies['parallel'] = self._parallel_planning
        self._planning_strategies['hybrid'] = self._hybrid_planning
    
    async def _sequential_planning(self, complex_task: Dict[str, Any]):
        """Sequential planning strategy."""
        # Simple sequential decomposition
        task_count = complex_task.get('subtask_count', 3)
        tasks = []
        execution_order = []
        
        for i in range(task_count):
            task_id = str(uuid.uuid4())
            dependencies = [execution_order[-1]] if execution_order else []
            
            task = Task(
                task_id=task_id,
                name=f"Subtask {i+1}",
                description=f"Sequential subtask {i+1}",
                task_type=complex_task.get('task_type', 'generic'),
                parameters=complex_task.get('parameters', {}),
                priority=TaskPriority.NORMAL,
                status=TaskStatus.PENDING,
                dependencies=dependencies,
                estimated_duration=timedelta(minutes=5)
            )
            
            tasks.append(task)
            execution_order.append(task_id)
        
        estimated_duration = timedelta(minutes=5 * task_count)
        return tasks, execution_order, estimated_duration
    
    async def _parallel_planning(self, complex_task: Dict[str, Any]):
        """Parallel planning strategy."""
        # Simple parallel decomposition
        task_count = complex_task.get('subtask_count', 3)
        tasks = []
        execution_order = []
        
        for i in range(task_count):
            task_id = str(uuid.uuid4())
            
            task = Task(
                task_id=task_id,
                name=f"Parallel Task {i+1}",
                description=f"Parallel subtask {i+1}",
                task_type=complex_task.get('task_type', 'generic'),
                parameters=complex_task.get('parameters', {}),
                priority=TaskPriority.NORMAL,
                status=TaskStatus.PENDING,
                dependencies=[],  # No dependencies for parallel execution
                estimated_duration=timedelta(minutes=5)
            )
            
            tasks.append(task)
            execution_order.append(task_id)
        
        estimated_duration = timedelta(minutes=5)  # All run in parallel
        return tasks, execution_order, estimated_duration
    
    async def _hybrid_planning(self, complex_task: Dict[str, Any]):
        """Hybrid planning strategy combining sequential and parallel elements."""
        # This is a simplified hybrid approach
        return await self._sequential_planning(complex_task)
    
    async def _handle_task_created(self, event):
        """Handle task creation events."""
        self._logger.debug(f"Task created event: {event.data}")
    
    async def _handle_task_completed(self, event):
        """Handle task completion events."""
        self._logger.debug(f"Task completed event: {event.data}")
    
    async def _handle_agent_available(self, event):
        """Handle agent availability events."""
        self._logger.debug(f"Agent available event: {event.data}")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get planning service statistics."""
        return {
            'total_tasks': len(self._tasks),
            'active_tasks': len(self._active_executions),
            'total_plans': len(self._plans),
            'task_status_counts': {
                status.value: sum(1 for task in self._tasks.values() if task.status == status)
                for status in TaskStatus
            },
            'available_strategies': list(self._planning_strategies.keys())
        }