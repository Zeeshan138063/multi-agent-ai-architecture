"""
Planning Agent - Specialized agent for task planning and strategic thinking.
Demonstrates complex planning capabilities within the agent framework.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ...core.interfaces import AgentInterface, AgentMessage, AgentResponse


class PlanStatus(Enum):
    """Status of a plan."""
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PlanStep:
    """Individual step in a plan."""
    step_id: str
    name: str
    description: str
    estimated_duration: timedelta
    dependencies: List[str]
    resources_required: List[str]
    status: str = "pending"


@dataclass
class Plan:
    """Complete plan structure."""
    plan_id: str
    name: str
    description: str
    steps: List[PlanStep]
    total_estimated_duration: timedelta
    priority: int
    status: PlanStatus
    created_at: datetime
    metadata: Dict[str, Any]


class PlanningAgent(AgentInterface):
    """
    Agent specialized in strategic planning, task orchestration, and workflow design.
    
    Capabilities:
    - Strategic planning
    - Task decomposition
    - Resource allocation
    - Timeline optimization
    - Plan execution monitoring
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self._logger = logging.getLogger(__name__)
        self._active_plans: Dict[str, Plan] = {}
        self._plan_templates: Dict[str, Dict[str, Any]] = {}
        self._planning_strategies: Dict[str, callable] = {}
        self._max_plan_complexity = config.get('max_plan_complexity', 50)
        self._default_planning_horizon = config.get('default_planning_horizon_hours', 24)
        
        # Initialize planning strategies
        self._initialize_planning_strategies()
        
        # Load plan templates
        self._load_plan_templates()
    
    async def process_task(self, task: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process a planning task to create or manage plans.
        
        Args:
            task: Task containing planning requirements
            context: Optional execution context
            
        Returns:
            AgentResponse with planning result
        """
        try:
            task_type = task.get('type', 'create_plan')
            
            self._logger.info(f"Processing planning task: {task_type}")
            
            if task_type == 'create_plan':
                result = await self._create_plan(task, context)
            elif task_type == 'optimize_plan':
                result = await self._optimize_plan(task, context)
            elif task_type == 'execute_plan':
                result = await self._execute_plan(task, context)
            elif task_type == 'monitor_plan':
                result = await self._monitor_plan(task, context)
            elif task_type == 'adapt_plan':
                result = await self._adapt_plan(task, context)
            else:
                result = {
                    'error': f'Unknown planning task type: {task_type}',
                    'supported_types': ['create_plan', 'optimize_plan', 'execute_plan', 'monitor_plan', 'adapt_plan']
                }
                return AgentResponse(result=result, success=False)
            
            return AgentResponse(
                result=result,
                success=result.get('success', True),
                metadata={
                    'task_type': task_type,
                    'processing_time': result.get('processing_time', 0.0),
                    'plan_complexity': result.get('plan_complexity', 0)
                }
            )
            
        except Exception as e:
            self._logger.error(f"Error in planning task: {e}")
            return AgentResponse(
                result=None,
                success=False,
                error_message=str(e)
            )
    
    async def communicate(self, message: AgentMessage) -> AgentResponse:
        """
        Handle communication about planning matters.
        
        Args:
            message: Message from another agent or system
            
        Returns:
            Response with planning-related information
        """
        try:
            self._logger.debug(f"Received planning message from {message.sender}")
            
            content = message.content
            
            if isinstance(content, dict):
                message_type = content.get('type', 'general')
                
                if message_type == 'plan_status_request':
                    plan_id = content.get('plan_id')
                    response = await self._get_plan_status(plan_id)
                elif message_type == 'planning_capability_request':
                    response = {
                        'capabilities': self.get_capabilities(),
                        'active_plans': len(self._active_plans),
                        'available_strategies': list(self._planning_strategies.keys()),
                        'plan_templates': list(self._plan_templates.keys())
                    }
                elif message_type == 'collaboration_request':
                    response = await self._handle_collaboration_request(content)
                else:
                    response = {
                        'message': 'Planning agent ready to assist with strategic planning tasks',
                        'supported_message_types': ['plan_status_request', 'planning_capability_request', 'collaboration_request']
                    }
            else:
                response = {
                    'message': 'Please specify the planning assistance you need',
                    'capabilities': self.get_capabilities()
                }
            
            return AgentResponse(
                result=response,
                success=True,
                metadata={'message_type': message_type if isinstance(content, dict) else 'general'}
            )
            
        except Exception as e:
            self._logger.error(f"Error in planning communication: {e}")
            return AgentResponse(
                result=None,
                success=False,
                error_message=str(e)
            )
    
    def get_capabilities(self) -> List[str]:
        """Return list of planning capabilities."""
        return [
            'strategic_planning',
            'task_decomposition',
            'resource_allocation',
            'timeline_optimization',
            'risk_assessment',
            'contingency_planning',
            'plan_monitoring',
            'adaptive_planning',
            'workflow_design',
            'dependency_resolution'
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Return current agent status."""
        return {
            'agent_id': self.agent_id,
            'is_active': self.is_active,
            'active_plans': len(self._active_plans),
            'plan_templates': len(self._plan_templates),
            'planning_strategies': len(self._planning_strategies),
            'capabilities': self.get_capabilities(),
            'config': {
                'max_plan_complexity': self._max_plan_complexity,
                'default_planning_horizon': self._default_planning_horizon
            }
        }
    
    async def _create_plan(self, task: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a new plan based on task requirements."""
        start_time = datetime.now()
        
        try:
            # Extract planning parameters
            goal = task.get('goal', 'Unspecified goal')
            requirements = task.get('requirements', {})
            constraints = task.get('constraints', {})
            strategy = task.get('strategy', 'sequential')
            
            # Use appropriate planning strategy
            planning_func = self._planning_strategies.get(strategy, self._sequential_planning)
            
            # Create the plan
            plan = await planning_func(goal, requirements, constraints)
            
            if plan:
                self._active_plans[plan.plan_id] = plan
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                return {
                    'success': True,
                    'plan_id': plan.plan_id,
                    'plan': {
                        'name': plan.name,
                        'description': plan.description,
                        'steps_count': len(plan.steps),
                        'estimated_duration': plan.total_estimated_duration.total_seconds(),
                        'priority': plan.priority,
                        'status': plan.status.value
                    },
                    'processing_time': processing_time,
                    'plan_complexity': len(plan.steps)
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to create plan',
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    async def _optimize_plan(self, task: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize an existing plan for better efficiency."""
        plan_id = task.get('plan_id')
        optimization_criteria = task.get('criteria', ['time', 'resources'])
        
        if plan_id not in self._active_plans:
            return {'success': False, 'error': f'Plan {plan_id} not found'}
        
        plan = self._active_plans[plan_id]
        
        # Apply optimization algorithms
        optimizations_applied = []
        
        if 'time' in optimization_criteria:
            time_optimization = await self._optimize_timeline(plan)
            optimizations_applied.append(time_optimization)
        
        if 'resources' in optimization_criteria:
            resource_optimization = await self._optimize_resources(plan)
            optimizations_applied.append(resource_optimization)
        
        if 'dependencies' in optimization_criteria:
            dependency_optimization = await self._optimize_dependencies(plan)
            optimizations_applied.append(dependency_optimization)
        
        return {
            'success': True,
            'plan_id': plan_id,
            'optimizations_applied': optimizations_applied,
            'new_estimated_duration': plan.total_estimated_duration.total_seconds()
        }
    
    async def _execute_plan(self, task: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Begin execution of a plan."""
        plan_id = task.get('plan_id')
        
        if plan_id not in self._active_plans:
            return {'success': False, 'error': f'Plan {plan_id} not found'}
        
        plan = self._active_plans[plan_id]
        plan.status = PlanStatus.ACTIVE
        
        # In a real implementation, this would coordinate with other agents/services
        execution_summary = {
            'plan_started': True,
            'initial_steps_ready': len([s for s in plan.steps if not s.dependencies]),
            'total_steps': len(plan.steps),
            'estimated_completion': (datetime.now() + plan.total_estimated_duration).isoformat()
        }
        
        return {
            'success': True,
            'plan_id': plan_id,
            'execution_summary': execution_summary
        }
    
    async def _monitor_plan(self, task: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Monitor the progress of an active plan."""
        plan_id = task.get('plan_id')
        
        if plan_id not in self._active_plans:
            return {'success': False, 'error': f'Plan {plan_id} not found'}
        
        plan = self._active_plans[plan_id]
        
        # Calculate progress metrics
        total_steps = len(plan.steps)
        completed_steps = len([s for s in plan.steps if s.status == 'completed'])
        in_progress_steps = len([s for s in plan.steps if s.status == 'in_progress'])
        pending_steps = len([s for s in plan.steps if s.status == 'pending'])
        
        progress_percentage = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        
        return {
            'success': True,
            'plan_id': plan_id,
            'status': plan.status.value,
            'progress': {
                'percentage': progress_percentage,
                'completed_steps': completed_steps,
                'in_progress_steps': in_progress_steps,
                'pending_steps': pending_steps,
                'total_steps': total_steps
            },
            'estimated_completion': (datetime.now() + plan.total_estimated_duration).isoformat()
        }
    
    async def _adapt_plan(self, task: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Adapt a plan based on changing conditions."""
        plan_id = task.get('plan_id')
        changes = task.get('changes', {})
        
        if plan_id not in self._active_plans:
            return {'success': False, 'error': f'Plan {plan_id} not found'}
        
        plan = self._active_plans[plan_id]
        adaptations_made = []
        
        # Handle different types of adaptations
        if 'new_constraints' in changes:
            adaptation = await self._adapt_to_constraints(plan, changes['new_constraints'])
            adaptations_made.append(adaptation)
        
        if 'resource_changes' in changes:
            adaptation = await self._adapt_to_resource_changes(plan, changes['resource_changes'])
            adaptations_made.append(adaptation)
        
        if 'timeline_changes' in changes:
            adaptation = await self._adapt_to_timeline_changes(plan, changes['timeline_changes'])
            adaptations_made.append(adaptation)
        
        return {
            'success': True,
            'plan_id': plan_id,
            'adaptations_made': adaptations_made,
            'updated_plan': {
                'estimated_duration': plan.total_estimated_duration.total_seconds(),
                'steps_count': len(plan.steps)
            }
        }
    
    def _initialize_planning_strategies(self):
        """Initialize available planning strategies."""
        self._planning_strategies = {
            'sequential': self._sequential_planning,
            'parallel': self._parallel_planning,
            'hybrid': self._hybrid_planning,
            'critical_path': self._critical_path_planning,
            'agile': self._agile_planning
        }
    
    def _load_plan_templates(self):
        """Load predefined plan templates."""
        self._plan_templates = {
            'software_development': {
                'phases': ['analysis', 'design', 'implementation', 'testing', 'deployment'],
                'typical_duration_hours': 160,
                'required_roles': ['developer', 'tester', 'project_manager']
            },
            'research_project': {
                'phases': ['literature_review', 'hypothesis', 'methodology', 'data_collection', 'analysis', 'writing'],
                'typical_duration_hours': 320,
                'required_roles': ['researcher', 'analyst', 'writer']
            },
            'marketing_campaign': {
                'phases': ['research', 'strategy', 'creative', 'execution', 'monitoring'],
                'typical_duration_hours': 80,
                'required_roles': ['marketer', 'designer', 'analyst']
            }
        }
    
    async def _sequential_planning(self, goal: str, requirements: Dict[str, Any], 
                                 constraints: Dict[str, Any]) -> Optional[Plan]:
        """Create a sequential execution plan."""
        import uuid
        
        plan_id = str(uuid.uuid4())
        steps = []
        
        # Create basic sequential steps
        base_steps = [
            ('analysis', 'Analyze requirements and constraints', 2),
            ('design', 'Design solution approach', 4),
            ('implementation', 'Implement the solution', 8),
            ('testing', 'Test and validate results', 3),
            ('deployment', 'Deploy and finalize', 2)
        ]
        
        total_duration = timedelta(hours=0)
        previous_step_id = None
        
        for i, (name, description, hours) in enumerate(base_steps):
            step_id = f"{plan_id}_step_{i+1}"
            duration = timedelta(hours=hours)
            
            step = PlanStep(
                step_id=step_id,
                name=name,
                description=description,
                estimated_duration=duration,
                dependencies=[previous_step_id] if previous_step_id else [],
                resources_required=['agent', 'compute']
            )
            
            steps.append(step)
            total_duration += duration
            previous_step_id = step_id
        
        return Plan(
            plan_id=plan_id,
            name=f"Sequential Plan for: {goal}",
            description=f"Sequential execution plan to achieve: {goal}",
            steps=steps,
            total_estimated_duration=total_duration,
            priority=requirements.get('priority', 2),
            status=PlanStatus.DRAFT,
            created_at=datetime.now(),
            metadata={'strategy': 'sequential', 'goal': goal}
        )
    
    async def _parallel_planning(self, goal: str, requirements: Dict[str, Any], 
                               constraints: Dict[str, Any]) -> Optional[Plan]:
        """Create a parallel execution plan."""
        import uuid
        
        plan_id = str(uuid.uuid4())
        steps = []
        
        # Create parallel steps
        parallel_steps = [
            ('research_a', 'Research approach A', 4),
            ('research_b', 'Research approach B', 4),
            ('prototype_a', 'Create prototype A', 6),
            ('prototype_b', 'Create prototype B', 6),
        ]
        
        # Final integration step
        integration_step = ('integration', 'Integrate results', 3)
        
        max_duration = timedelta(hours=0)
        parallel_step_ids = []
        
        # Create parallel steps
        for i, (name, description, hours) in enumerate(parallel_steps):
            step_id = f"{plan_id}_parallel_{i+1}"
            duration = timedelta(hours=hours)
            max_duration = max(max_duration, duration)
            
            step = PlanStep(
                step_id=step_id,
                name=name,
                description=description,
                estimated_duration=duration,
                dependencies=[],
                resources_required=['agent', 'compute']
            )
            
            steps.append(step)
            parallel_step_ids.append(step_id)
        
        # Add integration step
        integration_step_id = f"{plan_id}_integration"
        integration_duration = timedelta(hours=integration_step[2])
        
        integration = PlanStep(
            step_id=integration_step_id,
            name=integration_step[0],
            description=integration_step[1],
            estimated_duration=integration_duration,
            dependencies=parallel_step_ids,
            resources_required=['agent', 'compute']
        )
        
        steps.append(integration)
        total_duration = max_duration + integration_duration
        
        return Plan(
            plan_id=plan_id,
            name=f"Parallel Plan for: {goal}",
            description=f"Parallel execution plan to achieve: {goal}",
            steps=steps,
            total_estimated_duration=total_duration,
            priority=requirements.get('priority', 2),
            status=PlanStatus.DRAFT,
            created_at=datetime.now(),
            metadata={'strategy': 'parallel', 'goal': goal}
        )
    
    async def _hybrid_planning(self, goal: str, requirements: Dict[str, Any], 
                             constraints: Dict[str, Any]) -> Optional[Plan]:
        """Create a hybrid (sequential + parallel) execution plan."""
        # Combination of sequential and parallel elements
        # This is a simplified version - real implementation would be more sophisticated
        sequential_plan = await self._sequential_planning(goal, requirements, constraints)
        return sequential_plan
    
    async def _critical_path_planning(self, goal: str, requirements: Dict[str, Any], 
                                    constraints: Dict[str, Any]) -> Optional[Plan]:
        """Create a plan optimized for critical path."""
        # Simplified critical path method
        return await self._sequential_planning(goal, requirements, constraints)
    
    async def _agile_planning(self, goal: str, requirements: Dict[str, Any], 
                            constraints: Dict[str, Any]) -> Optional[Plan]:
        """Create an agile-style iterative plan."""
        # Simplified agile planning
        return await self._sequential_planning(goal, requirements, constraints)
    
    async def _optimize_timeline(self, plan: Plan) -> Dict[str, Any]:
        """Optimize plan timeline."""
        # Simple optimization: reduce durations by 10%
        for step in plan.steps:
            step.estimated_duration = timedelta(
                seconds=step.estimated_duration.total_seconds() * 0.9
            )
        
        # Recalculate total duration
        plan.total_estimated_duration = sum(
            (step.estimated_duration for step in plan.steps),
            timedelta()
        )
        
        return {
            'type': 'timeline_optimization',
            'improvement': '10% duration reduction',
            'new_duration': plan.total_estimated_duration.total_seconds()
        }
    
    async def _optimize_resources(self, plan: Plan) -> Dict[str, Any]:
        """Optimize resource allocation."""
        # Simple resource optimization
        return {
            'type': 'resource_optimization',
            'improvement': 'Resource allocation optimized',
            'savings': '15% resource efficiency'
        }
    
    async def _optimize_dependencies(self, plan: Plan) -> Dict[str, Any]:
        """Optimize step dependencies."""
        # Simple dependency optimization
        return {
            'type': 'dependency_optimization',
            'improvement': 'Dependencies streamlined',
            'parallelization_opportunities': 2
        }
    
    async def _adapt_to_constraints(self, plan: Plan, new_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt plan to new constraints."""
        return {
            'type': 'constraint_adaptation',
            'changes_made': 'Plan adapted to new constraints',
            'constraint_count': len(new_constraints)
        }
    
    async def _adapt_to_resource_changes(self, plan: Plan, resource_changes: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt plan to resource changes."""
        return {
            'type': 'resource_adaptation',
            'changes_made': 'Plan adapted to resource changes',
            'affected_steps': len(plan.steps)
        }
    
    async def _adapt_to_timeline_changes(self, plan: Plan, timeline_changes: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt plan to timeline changes."""
        return {
            'type': 'timeline_adaptation',
            'changes_made': 'Plan adapted to timeline changes',
            'new_deadline': timeline_changes.get('new_deadline', 'unchanged')
        }
    
    async def _get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """Get status of a specific plan."""
        if plan_id not in self._active_plans:
            return {'error': f'Plan {plan_id} not found'}
        
        plan = self._active_plans[plan_id]
        return {
            'plan_id': plan_id,
            'name': plan.name,
            'status': plan.status.value,
            'steps_count': len(plan.steps),
            'estimated_duration': plan.total_estimated_duration.total_seconds(),
            'created_at': plan.created_at.isoformat()
        }
    
    async def _handle_collaboration_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Handle collaboration requests from other agents."""
        collaboration_type = content.get('collaboration_type', 'general')
        
        if collaboration_type == 'plan_review':
            return {
                'response': 'I can review and provide feedback on plans',
                'capabilities': ['plan_analysis', 'optimization_suggestions', 'risk_assessment']
            }
        elif collaboration_type == 'joint_planning':
            return {
                'response': 'I can collaborate on joint planning efforts',
                'capabilities': ['plan_integration', 'resource_coordination', 'timeline_synchronization']
            }
        else:
            return {
                'response': 'I can assist with various planning collaborations',
                'supported_types': ['plan_review', 'joint_planning']
            }
    
    async def initialize(self) -> bool:
        """Initialize the planning agent."""
        try:
            self._logger.info(f"Initializing PlanningAgent {self.agent_id}")
            
            # Validate configuration
            if self._max_plan_complexity <= 0:
                self._max_plan_complexity = 50
            
            if self._default_planning_horizon <= 0:
                self._default_planning_horizon = 24
            
            self.is_active = True
            self._logger.info(f"PlanningAgent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize PlanningAgent {self.agent_id}: {e}")
            return False