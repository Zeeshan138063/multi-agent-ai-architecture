"""
Execution Agent - Specialized agent for task execution and action implementation.
Focuses on executing concrete actions and managing execution workflows.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid

from ...core.interfaces import AgentInterface, AgentMessage, AgentResponse


class ExecutionStatus(Enum):
    """Status of task execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ExecutionTask:
    """Individual execution task."""
    task_id: str
    name: str
    action_type: str
    parameters: Dict[str, Any]
    status: ExecutionStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class ExecutionAgent(AgentInterface):
    """
    Agent specialized in executing concrete actions and managing execution workflows.
    
    Capabilities:
    - Task execution
    - Action implementation
    - Workflow management
    - Resource utilization
    - Error handling and recovery
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self._logger = logging.getLogger(__name__)
        self._active_tasks: Dict[str, ExecutionTask] = {}
        self._execution_history: List[ExecutionTask] = []
        self._action_handlers: Dict[str, callable] = {}
        self._max_concurrent_tasks = config.get('max_concurrent_tasks', 5)
        self._default_timeout = config.get('default_timeout_seconds', 300)
        self._enable_recovery = config.get('enable_recovery', True)
        
        # Initialize action handlers
        self._initialize_action_handlers()
    
    async def process_task(self, task: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process an execution task by running the specified actions.
        
        Args:
            task: Task containing execution instructions
            context: Optional execution context
            
        Returns:
            AgentResponse with execution result
        """
        try:
            task_type = task.get('type', 'execute')
            
            self._logger.info(f"Processing execution task: {task_type}")
            
            if task_type == 'execute':
                result = await self._execute_action(task, context)
            elif task_type == 'execute_batch':
                result = await self._execute_batch(task, context)
            elif task_type == 'execute_workflow':
                result = await self._execute_workflow(task, context)
            elif task_type == 'cancel_execution':
                result = await self._cancel_execution(task, context)
            elif task_type == 'pause_execution':
                result = await self._pause_execution(task, context)
            elif task_type == 'resume_execution':
                result = await self._resume_execution(task, context)
            else:
                result = {
                    'error': f'Unknown execution task type: {task_type}',
                    'supported_types': ['execute', 'execute_batch', 'execute_workflow', 'cancel_execution', 'pause_execution', 'resume_execution']
                }
                return AgentResponse(result=result, success=False)
            
            return AgentResponse(
                result=result,
                success=result.get('success', True),
                metadata={
                    'task_type': task_type,
                    'execution_time': result.get('execution_time', 0.0),
                    'active_tasks': len(self._active_tasks)
                }
            )
            
        except Exception as e:
            self._logger.error(f"Error in execution task: {e}")
            return AgentResponse(
                result=None,
                success=False,
                error_message=str(e)
            )
    
    async def communicate(self, message: AgentMessage) -> AgentResponse:
        """
        Handle communication about execution matters.
        
        Args:
            message: Message from another agent or system
            
        Returns:
            Response with execution-related information
        """
        try:
            self._logger.debug(f"Received execution message from {message.sender}")
            
            content = message.content
            
            if isinstance(content, dict):
                message_type = content.get('type', 'general')
                
                if message_type == 'execution_status_request':
                    task_id = content.get('task_id')
                    response = await self._get_execution_status(task_id)
                elif message_type == 'execution_capability_request':
                    response = {
                        'capabilities': self.get_capabilities(),
                        'active_tasks': len(self._active_tasks),
                        'available_actions': list(self._action_handlers.keys()),
                        'max_concurrent_tasks': self._max_concurrent_tasks
                    }
                elif message_type == 'resource_status_request':
                    response = await self._get_resource_status()
                else:
                    response = {
                        'message': 'Execution agent ready to execute tasks and actions',
                        'supported_message_types': ['execution_status_request', 'execution_capability_request', 'resource_status_request']
                    }
            else:
                response = {
                    'message': 'Please specify the execution assistance you need',
                    'capabilities': self.get_capabilities()
                }
            
            return AgentResponse(
                result=response,
                success=True,
                metadata={'message_type': message_type if isinstance(content, dict) else 'general'}
            )
            
        except Exception as e:
            self._logger.error(f"Error in execution communication: {e}")
            return AgentResponse(
                result=None,
                success=False,
                error_message=str(e)
            )
    
    def get_capabilities(self) -> List[str]:
        """Return list of execution capabilities."""
        return [
            'task_execution',
            'batch_processing',
            'workflow_execution',
            'action_implementation',
            'resource_management',
            'error_recovery',
            'concurrent_execution',
            'execution_monitoring',
            'performance_optimization',
            'state_management'
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Return current agent status."""
        return {
            'agent_id': self.agent_id,
            'is_active': self.is_active,
            'active_tasks': len(self._active_tasks),
            'execution_history_count': len(self._execution_history),
            'available_actions': list(self._action_handlers.keys()),
            'capabilities': self.get_capabilities(),
            'config': {
                'max_concurrent_tasks': self._max_concurrent_tasks,
                'default_timeout': self._default_timeout,
                'enable_recovery': self._enable_recovery
            }
        }
    
    async def _execute_action(self, task: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a single action."""
        start_time = datetime.now()
        
        try:
            action_type = task.get('action_type', 'default')
            parameters = task.get('parameters', {})
            timeout = task.get('timeout', self._default_timeout)
            
            # Check if we can handle this action type
            if action_type not in self._action_handlers:
                return {
                    'success': False,
                    'error': f'Unknown action type: {action_type}',
                    'available_actions': list(self._action_handlers.keys())
                }
            
            # Check concurrent task limit
            if len(self._active_tasks) >= self._max_concurrent_tasks:
                return {
                    'success': False,
                    'error': 'Maximum concurrent tasks reached',
                    'max_concurrent': self._max_concurrent_tasks,
                    'active_count': len(self._active_tasks)
                }
            
            # Create execution task
            execution_task = ExecutionTask(
                task_id=str(uuid.uuid4()),
                name=task.get('name', f'Execute {action_type}'),
                action_type=action_type,
                parameters=parameters,
                status=ExecutionStatus.PENDING,
                created_at=datetime.now()
            )
            
            self._active_tasks[execution_task.task_id] = execution_task
            
            try:
                # Execute the action with timeout
                execution_task.status = ExecutionStatus.RUNNING
                execution_task.started_at = datetime.now()
                
                action_handler = self._action_handlers[action_type]
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    action_handler(parameters, context),
                    timeout=timeout
                )
                
                execution_task.status = ExecutionStatus.COMPLETED
                execution_task.completed_at = datetime.now()
                execution_task.result = result
                
                # Move to history
                self._execution_history.append(execution_task)
                del self._active_tasks[execution_task.task_id]
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                return {
                    'success': True,
                    'task_id': execution_task.task_id,
                    'result': result,
                    'execution_time': execution_time,
                    'action_type': action_type
                }
                
            except asyncio.TimeoutError:
                execution_task.status = ExecutionStatus.FAILED
                execution_task.error_message = 'Execution timeout'
                execution_task.completed_at = datetime.now()
                
                # Try recovery if enabled
                if self._enable_recovery and execution_task.retry_count < execution_task.max_retries:
                    recovery_result = await self._attempt_recovery(execution_task)
                    if recovery_result['success']:
                        return recovery_result
                
                del self._active_tasks[execution_task.task_id]
                self._execution_history.append(execution_task)
                
                return {
                    'success': False,
                    'error': 'Execution timeout',
                    'timeout': timeout,
                    'task_id': execution_task.task_id
                }
                
            except Exception as e:
                execution_task.status = ExecutionStatus.FAILED
                execution_task.error_message = str(e)
                execution_task.completed_at = datetime.now()
                
                # Try recovery if enabled
                if self._enable_recovery and execution_task.retry_count < execution_task.max_retries:
                    recovery_result = await self._attempt_recovery(execution_task)
                    if recovery_result['success']:
                        return recovery_result
                
                del self._active_tasks[execution_task.task_id]
                self._execution_history.append(execution_task)
                
                return {
                    'success': False,
                    'error': str(e),
                    'task_id': execution_task.task_id
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to create execution task: {str(e)}',
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
    
    async def _execute_batch(self, task: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute multiple actions in batch."""
        start_time = datetime.now()
        
        actions = task.get('actions', [])
        execution_mode = task.get('mode', 'sequential')  # sequential or parallel
        
        if not actions:
            return {
                'success': False,
                'error': 'No actions provided for batch execution'
            }
        
        results = []
        
        try:
            if execution_mode == 'parallel':
                # Execute actions in parallel
                tasks = []
                for action in actions:
                    action_task = self._execute_action(action, context)
                    tasks.append(action_task)
                
                parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(parallel_results):
                    if isinstance(result, Exception):
                        results.append({
                            'action_index': i,
                            'success': False,
                            'error': str(result)
                        })
                    else:
                        results.append({
                            'action_index': i,
                            **result
                        })
            
            else:  # sequential
                for i, action in enumerate(actions):
                    result = await self._execute_action(action, context)
                    results.append({
                        'action_index': i,
                        **result
                    })
                    
                    # If this action failed and we're in strict mode, stop
                    if not result.get('success', False) and task.get('strict_mode', False):
                        break
            
            # Calculate overall success
            successful_actions = sum(1 for r in results if r.get('success', False))
            total_actions = len(actions)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return {
                'success': successful_actions == total_actions,
                'batch_results': results,
                'successful_actions': successful_actions,
                'total_actions': total_actions,
                'execution_time': execution_time,
                'execution_mode': execution_mode
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Batch execution failed: {str(e)}',
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
    
    async def _execute_workflow(self, task: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a workflow with dependencies and conditions."""
        start_time = datetime.now()
        
        workflow_steps = task.get('workflow_steps', [])
        workflow_id = task.get('workflow_id', str(uuid.uuid4()))
        
        if not workflow_steps:
            return {
                'success': False,
                'error': 'No workflow steps provided'
            }
        
        workflow_results = {}
        step_results = []
        
        try:
            for step in workflow_steps:
                step_id = step.get('step_id', str(uuid.uuid4()))
                dependencies = step.get('dependencies', [])
                condition = step.get('condition', None)
                
                # Check if dependencies are met
                dependencies_met = all(
                    dep_id in workflow_results and workflow_results[dep_id].get('success', False)
                    for dep_id in dependencies
                )
                
                if not dependencies_met:
                    step_results.append({
                        'step_id': step_id,
                        'success': False,
                        'error': 'Dependencies not met',
                        'dependencies': dependencies
                    })
                    continue
                
                # Check condition if specified
                if condition and not self._evaluate_condition(condition, workflow_results):
                    step_results.append({
                        'step_id': step_id,
                        'success': False,
                        'skipped': True,
                        'reason': 'Condition not met'
                    })
                    continue
                
                # Execute the step
                step_context = {**(context or {}), 'workflow_results': workflow_results}
                step_result = await self._execute_action(step, step_context)
                
                workflow_results[step_id] = step_result
                step_results.append({
                    'step_id': step_id,
                    **step_result
                })
            
            # Calculate workflow success
            successful_steps = sum(1 for r in step_results if r.get('success', False))
            total_steps = len([s for s in step_results if not s.get('skipped', False)])
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return {
                'success': successful_steps == total_steps if total_steps > 0 else True,
                'workflow_id': workflow_id,
                'step_results': step_results,
                'successful_steps': successful_steps,
                'total_steps': total_steps,
                'execution_time': execution_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Workflow execution failed: {str(e)}',
                'workflow_id': workflow_id,
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _initialize_action_handlers(self):
        """Initialize available action handlers."""
        self._action_handlers = {
            'compute': self._handle_compute_action,
            'data_processing': self._handle_data_processing,
            'api_call': self._handle_api_call,
            'file_operation': self._handle_file_operation,
            'notification': self._handle_notification,
            'validation': self._handle_validation,
            'transformation': self._handle_transformation,
            'aggregation': self._handle_aggregation
        }
    
    async def _handle_compute_action(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle computational actions."""
        operation = parameters.get('operation', 'calculate')
        data = parameters.get('data', {})
        
        # Simulate computation
        await asyncio.sleep(0.1)  # Simulate processing time
        
        if operation == 'calculate':
            values = data.get('values', [1, 2, 3, 4, 5])
            result = sum(values)
            return {
                'operation': operation,
                'result': result,
                'input_count': len(values)
            }
        elif operation == 'analyze':
            items = data.get('items', [])
            return {
                'operation': operation,
                'analysis': {
                    'count': len(items),
                    'has_data': len(items) > 0
                }
            }
        else:
            return {
                'operation': operation,
                'result': 'Computation completed',
                'message': f'Executed {operation} operation'
            }
    
    async def _handle_data_processing(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle data processing actions."""
        processing_type = parameters.get('type', 'filter')
        data = parameters.get('data', [])
        
        # Simulate data processing
        await asyncio.sleep(0.2)
        
        if processing_type == 'filter':
            criteria = parameters.get('criteria', {})
            processed_data = [item for item in data if self._matches_criteria(item, criteria)]
        elif processing_type == 'transform':
            transformation = parameters.get('transformation', 'identity')
            processed_data = [self._apply_transformation(item, transformation) for item in data]
        else:
            processed_data = data
        
        return {
            'processing_type': processing_type,
            'input_count': len(data),
            'output_count': len(processed_data),
            'processed_data': processed_data
        }
    
    async def _handle_api_call(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle API call actions."""
        url = parameters.get('url', 'http://example.com/api')
        method = parameters.get('method', 'GET')
        
        # Simulate API call
        await asyncio.sleep(0.5)
        
        return {
            'url': url,
            'method': method,
            'status_code': 200,
            'response': {'message': 'API call successful', 'data': 'simulated response'}
        }
    
    async def _handle_file_operation(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle file operations."""
        operation = parameters.get('operation', 'read')
        filename = parameters.get('filename', 'example.txt')
        
        # Simulate file operation
        await asyncio.sleep(0.1)
        
        return {
            'operation': operation,
            'filename': filename,
            'success': True,
            'message': f'File {operation} operation completed for {filename}'
        }
    
    async def _handle_notification(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle notification actions."""
        message = parameters.get('message', 'Notification sent')
        recipients = parameters.get('recipients', ['system'])
        
        # Simulate notification
        await asyncio.sleep(0.1)
        
        return {
            'message': message,
            'recipients': recipients,
            'sent_count': len(recipients),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _handle_validation(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle validation actions."""
        data = parameters.get('data', {})
        rules = parameters.get('rules', [])
        
        # Simulate validation
        await asyncio.sleep(0.1)
        
        validation_results = []
        for rule in rules:
            # Simple validation simulation
            passed = True  # Assume validation passes
            validation_results.append({
                'rule': rule,
                'passed': passed
            })
        
        all_passed = all(result['passed'] for result in validation_results)
        
        return {
            'data_validated': True,
            'all_rules_passed': all_passed,
            'validation_results': validation_results,
            'total_rules': len(rules)
        }
    
    async def _handle_transformation(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle data transformation actions."""
        input_data = parameters.get('input_data', {})
        transformation_type = parameters.get('transformation_type', 'format')
        
        # Simulate transformation
        await asyncio.sleep(0.1)
        
        # Simple transformation simulation
        if transformation_type == 'format':
            output_data = {k: str(v).upper() if isinstance(v, str) else v for k, v in input_data.items()}
        elif transformation_type == 'normalize':
            output_data = {k: v / 100 if isinstance(v, (int, float)) else v for k, v in input_data.items()}
        else:
            output_data = input_data
        
        return {
            'transformation_type': transformation_type,
            'input_data': input_data,
            'output_data': output_data,
            'transformed': True
        }
    
    async def _handle_aggregation(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle data aggregation actions."""
        data = parameters.get('data', [])
        aggregation_type = parameters.get('aggregation_type', 'sum')
        
        # Simulate aggregation
        await asyncio.sleep(0.1)
        
        if aggregation_type == 'sum':
            numeric_values = [v for v in data if isinstance(v, (int, float))]
            result = sum(numeric_values)
        elif aggregation_type == 'count':
            result = len(data)
        elif aggregation_type == 'average':
            numeric_values = [v for v in data if isinstance(v, (int, float))]
            result = sum(numeric_values) / len(numeric_values) if numeric_values else 0
        else:
            result = len(data)
        
        return {
            'aggregation_type': aggregation_type,
            'input_count': len(data),
            'result': result
        }
    
    def _matches_criteria(self, item: Any, criteria: Dict[str, Any]) -> bool:
        """Check if item matches filtering criteria."""
        # Simple criteria matching simulation
        return True  # Assume all items match for simulation
    
    def _apply_transformation(self, item: Any, transformation: str) -> Any:
        """Apply transformation to an item."""
        # Simple transformation simulation
        if transformation == 'uppercase' and isinstance(item, str):
            return item.upper()
        elif transformation == 'double' and isinstance(item, (int, float)):
            return item * 2
        return item
    
    def _evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a workflow condition."""
        # Simple condition evaluation
        condition_type = condition.get('type', 'always')
        
        if condition_type == 'always':
            return True
        elif condition_type == 'never':
            return False
        elif condition_type == 'success_count':
            required_successes = condition.get('required_successes', 1)
            success_count = sum(1 for result in context.values() if result.get('success', False))
            return success_count >= required_successes
        
        return True
    
    async def _attempt_recovery(self, execution_task: ExecutionTask) -> Dict[str, Any]:
        """Attempt to recover from failed execution."""
        execution_task.retry_count += 1
        
        # Simple recovery strategy: retry with exponential backoff
        backoff_time = 2 ** execution_task.retry_count
        await asyncio.sleep(min(backoff_time, 10))  # Max 10 seconds backoff
        
        try:
            # Retry the action
            action_handler = self._action_handlers[execution_task.action_type]
            result = await action_handler(execution_task.parameters, None)
            
            execution_task.status = ExecutionStatus.COMPLETED
            execution_task.completed_at = datetime.now()
            execution_task.result = result
            
            return {
                'success': True,
                'recovered': True,
                'retry_count': execution_task.retry_count,
                'result': result
            }
            
        except Exception as e:
            return {
                'success': False,
                'recovery_failed': True,
                'retry_count': execution_task.retry_count,
                'error': str(e)
            }
    
    async def _cancel_execution(self, task: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Cancel a running execution task."""
        task_id = task.get('task_id')
        
        if task_id not in self._active_tasks:
            return {'success': False, 'error': f'Task {task_id} not found or not active'}
        
        execution_task = self._active_tasks[task_id]
        execution_task.status = ExecutionStatus.CANCELLED
        execution_task.completed_at = datetime.now()
        
        del self._active_tasks[task_id]
        self._execution_history.append(execution_task)
        
        return {
            'success': True,
            'task_id': task_id,
            'cancelled': True,
            'message': f'Task {task_id} cancelled successfully'
        }
    
    async def _pause_execution(self, task: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Pause a running execution task."""
        task_id = task.get('task_id')
        
        if task_id not in self._active_tasks:
            return {'success': False, 'error': f'Task {task_id} not found or not active'}
        
        execution_task = self._active_tasks[task_id]
        execution_task.status = ExecutionStatus.PAUSED
        
        return {
            'success': True,
            'task_id': task_id,
            'paused': True,
            'message': f'Task {task_id} paused successfully'
        }
    
    async def _resume_execution(self, task: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Resume a paused execution task."""
        task_id = task.get('task_id')
        
        if task_id not in self._active_tasks:
            return {'success': False, 'error': f'Task {task_id} not found'}
        
        execution_task = self._active_tasks[task_id]
        if execution_task.status != ExecutionStatus.PAUSED:
            return {'success': False, 'error': f'Task {task_id} is not paused'}
        
        execution_task.status = ExecutionStatus.RUNNING
        
        return {
            'success': True,
            'task_id': task_id,
            'resumed': True,
            'message': f'Task {task_id} resumed successfully'
        }
    
    async def _get_execution_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific execution task."""
        # Check active tasks
        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            return {
                'task_id': task_id,
                'status': task.status.value,
                'name': task.name,
                'action_type': task.action_type,
                'created_at': task.created_at.isoformat(),
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'retry_count': task.retry_count,
                'active': True
            }
        
        # Check execution history
        for task in self._execution_history:
            if task.task_id == task_id:
                return {
                    'task_id': task_id,
                    'status': task.status.value,
                    'name': task.name,
                    'action_type': task.action_type,
                    'created_at': task.created_at.isoformat(),
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'result': task.result,
                    'error_message': task.error_message,
                    'retry_count': task.retry_count,
                    'active': False
                }
        
        return {'error': f'Task {task_id} not found'}
    
    async def _get_resource_status(self) -> Dict[str, Any]:
        """Get current resource utilization status."""
        return {
            'active_tasks': len(self._active_tasks),
            'max_concurrent_tasks': self._max_concurrent_tasks,
            'available_slots': self._max_concurrent_tasks - len(self._active_tasks),
            'total_executions': len(self._execution_history),
            'available_actions': list(self._action_handlers.keys())
        }
    
    async def initialize(self) -> bool:
        """Initialize the execution agent."""
        try:
            self._logger.info(f"Initializing ExecutionAgent {self.agent_id}")
            
            # Validate configuration
            if self._max_concurrent_tasks <= 0:
                self._max_concurrent_tasks = 5
            
            if self._default_timeout <= 0:
                self._default_timeout = 300
            
            self.is_active = True
            self._logger.info(f"ExecutionAgent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize ExecutionAgent {self.agent_id}: {e}")
            return False