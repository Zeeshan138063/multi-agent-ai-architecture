"""
Reasoning Agent - An intelligent agent focused on logical reasoning and problem-solving.
Demonstrates the agent interface implementation with reasoning capabilities.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ...core.interfaces import AgentInterface, AgentMessage, AgentResponse


class ReasoningAgent(AgentInterface):
    """
    Agent specialized in logical reasoning, problem decomposition, and analytical thinking.
    
    Capabilities:
    - Logical reasoning
    - Problem analysis
    - Solution generation
    - Decision making
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self._logger = logging.getLogger(__name__)
        self._reasoning_history: List[Dict[str, Any]] = []
        self._knowledge_base: Dict[str, Any] = {}
        self._max_reasoning_steps = config.get('max_reasoning_steps', 10)
        self._confidence_threshold = config.get('confidence_threshold', 0.7)
    
    async def process_task(self, task: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process a reasoning task using structured analytical approach.
        
        Args:
            task: Task containing problem statement and parameters
            context: Optional execution context
            
        Returns:
            AgentResponse with reasoning result
        """
        try:
            task_type = task.get('type', 'reasoning')
            problem = task.get('problem', '')
            parameters = task.get('parameters', {})
            
            self._logger.info(f"Processing reasoning task: {task_type}")
            
            # Start reasoning process
            reasoning_result = await self._perform_reasoning(problem, parameters, context)
            
            # Log reasoning session
            reasoning_record = {
                'timestamp': datetime.now(),
                'task_type': task_type,
                'problem': problem,
                'reasoning_steps': reasoning_result.get('steps', []),
                'conclusion': reasoning_result.get('conclusion', ''),
                'confidence': reasoning_result.get('confidence', 0.0)
            }
            self._reasoning_history.append(reasoning_record)
            
            # Determine success based on confidence
            success = reasoning_result.get('confidence', 0.0) >= self._confidence_threshold
            
            return AgentResponse(
                result=reasoning_result,
                success=success,
                metadata={
                    'reasoning_steps': len(reasoning_result.get('steps', [])),
                    'confidence': reasoning_result.get('confidence', 0.0),
                    'processing_time': reasoning_result.get('processing_time', 0.0)
                }
            )
            
        except Exception as e:
            self._logger.error(f"Error in reasoning task: {e}")
            return AgentResponse(
                result=None,
                success=False,
                error_message=str(e)
            )
    
    async def communicate(self, message: AgentMessage) -> AgentResponse:
        """
        Handle communication from other agents with reasoning context.
        
        Args:
            message: Message from another agent
            
        Returns:
            Response based on reasoning about the message
        """
        try:
            self._logger.debug(f"Received message from {message.sender}")
            
            # Analyze the message content
            analysis = await self._analyze_message(message)
            
            # Generate appropriate response
            response_content = await self._generate_response(message, analysis)
            
            return AgentResponse(
                result=response_content,
                success=True,
                metadata={
                    'message_analysis': analysis,
                    'response_type': response_content.get('type', 'acknowledgment')
                }
            )
            
        except Exception as e:
            self._logger.error(f"Error in agent communication: {e}")
            return AgentResponse(
                result=None,
                success=False,
                error_message=str(e)
            )
    
    def get_capabilities(self) -> List[str]:
        """Return list of reasoning capabilities."""
        return [
            'logical_reasoning',
            'problem_analysis',
            'decision_making',
            'pattern_recognition',
            'hypothesis_testing',
            'causal_reasoning',
            'deductive_reasoning',
            'inductive_reasoning'
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Return current agent status."""
        return {
            'agent_id': self.agent_id,
            'is_active': self.is_active,
            'reasoning_sessions': len(self._reasoning_history),
            'knowledge_items': len(self._knowledge_base),
            'capabilities': self.get_capabilities(),
            'config': {
                'max_reasoning_steps': self._max_reasoning_steps,
                'confidence_threshold': self._confidence_threshold
            }
        }
    
    async def _perform_reasoning(self, problem: str, parameters: Dict[str, Any], 
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Core reasoning process implementation.
        
        Args:
            problem: Problem statement to reason about
            parameters: Additional parameters for reasoning
            context: Execution context
            
        Returns:
            Reasoning result with conclusion and confidence
        """
        start_time = datetime.now()
        reasoning_steps = []
        
        try:
            # Step 1: Problem decomposition
            decomposition = await self._decompose_problem(problem)
            reasoning_steps.append({
                'step': 1,
                'type': 'decomposition',
                'description': 'Break down the problem into components',
                'result': decomposition
            })
            
            # Step 2: Gather relevant knowledge
            relevant_knowledge = await self._gather_knowledge(problem, decomposition)
            reasoning_steps.append({
                'step': 2,
                'type': 'knowledge_gathering',
                'description': 'Identify relevant knowledge and facts',
                'result': relevant_knowledge
            })
            
            # Step 3: Apply reasoning strategies
            reasoning_result = await self._apply_reasoning_strategies(
                decomposition, relevant_knowledge, parameters
            )
            reasoning_steps.append({
                'step': 3,
                'type': 'reasoning_application',
                'description': 'Apply logical reasoning strategies',
                'result': reasoning_result
            })
            
            # Step 4: Validate conclusion
            validation = await self._validate_conclusion(reasoning_result)
            reasoning_steps.append({
                'step': 4,
                'type': 'validation',
                'description': 'Validate the reasoning conclusion',
                'result': validation
            })
            
            # Calculate confidence score
            confidence = self._calculate_confidence(reasoning_steps, validation)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                'steps': reasoning_steps,
                'conclusion': reasoning_result.get('conclusion', ''),
                'confidence': confidence,
                'processing_time': processing_time,
                'problem': problem,
                'metadata': {
                    'decomposition_count': len(decomposition),
                    'knowledge_items_used': len(relevant_knowledge),
                    'validation_passed': validation.get('passed', False)
                }
            }
            
        except Exception as e:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                'steps': reasoning_steps,
                'conclusion': f"Reasoning failed: {str(e)}",
                'confidence': 0.0,
                'processing_time': processing_time,
                'error': str(e)
            }
    
    async def _decompose_problem(self, problem: str) -> List[Dict[str, Any]]:
        """Decompose a problem into smaller components."""
        # Simulate problem decomposition
        components = []
        
        # Simple keyword-based decomposition (in real implementation, would use NLP)
        if 'analyze' in problem.lower():
            components.append({'type': 'analysis', 'description': 'Perform detailed analysis'})
        if 'compare' in problem.lower():
            components.append({'type': 'comparison', 'description': 'Compare alternatives'})
        if 'decide' in problem.lower():
            components.append({'type': 'decision', 'description': 'Make a decision'})
        if 'solve' in problem.lower():
            components.append({'type': 'solution', 'description': 'Generate solution'})
        
        # Default component if no specific patterns found
        if not components:
            components.append({'type': 'general', 'description': 'General problem solving'})
        
        return components
    
    async def _gather_knowledge(self, problem: str, decomposition: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Gather relevant knowledge for the problem."""
        knowledge_items = []
        
        # Simulate knowledge retrieval from knowledge base
        for component in decomposition:
            component_type = component.get('type', 'general')
            
            # Add relevant knowledge based on component type
            if component_type == 'analysis':
                knowledge_items.append({
                    'type': 'method',
                    'content': 'Analysis requires systematic examination of parts',
                    'relevance': 0.9
                })
            elif component_type == 'comparison':
                knowledge_items.append({
                    'type': 'method',
                    'content': 'Comparison requires identifying criteria and evaluating alternatives',
                    'relevance': 0.8
                })
            elif component_type == 'decision':
                knowledge_items.append({
                    'type': 'method',
                    'content': 'Decision making requires weighing options against criteria',
                    'relevance': 0.9
                })
        
        return knowledge_items
    
    async def _apply_reasoning_strategies(self, decomposition: List[Dict[str, Any]], 
                                        knowledge: List[Dict[str, Any]], 
                                        parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply reasoning strategies to solve the problem."""
        strategies_applied = []
        intermediate_results = []
        
        # Apply deductive reasoning
        if any(comp.get('type') == 'analysis' for comp in decomposition):
            deductive_result = await self._apply_deductive_reasoning(knowledge, parameters)
            strategies_applied.append('deductive')
            intermediate_results.append(deductive_result)
        
        # Apply inductive reasoning
        if any(comp.get('type') == 'comparison' for comp in decomposition):
            inductive_result = await self._apply_inductive_reasoning(knowledge, parameters)
            strategies_applied.append('inductive')
            intermediate_results.append(inductive_result)
        
        # Apply abductive reasoning (inference to best explanation)
        if any(comp.get('type') == 'solution' for comp in decomposition):
            abductive_result = await self._apply_abductive_reasoning(knowledge, parameters)
            strategies_applied.append('abductive')
            intermediate_results.append(abductive_result)
        
        # Synthesize results
        conclusion = self._synthesize_results(intermediate_results)
        
        return {
            'strategies_applied': strategies_applied,
            'intermediate_results': intermediate_results,
            'conclusion': conclusion
        }
    
    async def _apply_deductive_reasoning(self, knowledge: List[Dict[str, Any]], 
                                       parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply deductive reasoning strategy."""
        # Simulate deductive reasoning
        premises = [k for k in knowledge if k.get('type') == 'fact']
        
        return {
            'type': 'deductive',
            'premises': premises,
            'conclusion': 'Based on the given facts, the logical conclusion is...',
            'confidence': 0.8
        }
    
    async def _apply_inductive_reasoning(self, knowledge: List[Dict[str, Any]], 
                                       parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply inductive reasoning strategy."""
        # Simulate inductive reasoning
        patterns = [k for k in knowledge if k.get('type') == 'pattern']
        
        return {
            'type': 'inductive',
            'patterns': patterns,
            'conclusion': 'Based on observed patterns, the general rule is...',
            'confidence': 0.7
        }
    
    async def _apply_abductive_reasoning(self, knowledge: List[Dict[str, Any]], 
                                       parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply abductive reasoning strategy."""
        # Simulate abductive reasoning
        explanations = [k for k in knowledge if k.get('type') == 'explanation']
        
        return {
            'type': 'abductive',
            'explanations': explanations,
            'conclusion': 'The best explanation for the observed phenomena is...',
            'confidence': 0.6
        }
    
    def _synthesize_results(self, results: List[Dict[str, Any]]) -> str:
        """Synthesize multiple reasoning results into a final conclusion."""
        if not results:
            return "No reasoning results to synthesize"
        
        # Simple synthesis - in practice would be more sophisticated
        conclusions = [r.get('conclusion', '') for r in results]
        confidences = [r.get('confidence', 0.0) for r in results]
        
        # Weight conclusions by confidence
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            return f"Synthesized conclusion (confidence: {avg_confidence:.2f}): " + \
                   " ".join(conclusions)
        
        return "Unable to synthesize conclusions"
    
    async def _validate_conclusion(self, reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the reasoning conclusion."""
        conclusion = reasoning_result.get('conclusion', '')
        
        # Simple validation checks
        validation_checks = []
        
        # Check if conclusion is non-empty
        validation_checks.append({
            'check': 'non_empty',
            'passed': bool(conclusion.strip()),
            'description': 'Conclusion should not be empty'
        })
        
        # Check if conclusion relates to applied strategies
        strategies = reasoning_result.get('strategies_applied', [])
        validation_checks.append({
            'check': 'strategy_alignment',
            'passed': len(strategies) > 0,
            'description': 'Conclusion should be based on applied strategies'
        })
        
        # Check logical consistency (simplified)
        validation_checks.append({
            'check': 'logical_consistency',
            'passed': 'contradiction' not in conclusion.lower(),
            'description': 'Conclusion should be logically consistent'
        })
        
        overall_passed = all(check['passed'] for check in validation_checks)
        
        return {
            'passed': overall_passed,
            'checks': validation_checks,
            'validation_score': sum(1 for check in validation_checks if check['passed']) / len(validation_checks)
        }
    
    def _calculate_confidence(self, reasoning_steps: List[Dict[str, Any]], 
                            validation: Dict[str, Any]) -> float:
        """Calculate overall confidence in the reasoning result."""
        base_confidence = 0.5
        
        # Boost confidence based on number of reasoning steps
        step_bonus = min(len(reasoning_steps) * 0.1, 0.3)
        
        # Adjust based on validation
        validation_score = validation.get('validation_score', 0.0)
        validation_bonus = validation_score * 0.2
        
        # Calculate final confidence
        confidence = base_confidence + step_bonus + validation_bonus
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    async def _analyze_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Analyze incoming message for reasoning context."""
        content = message.content
        
        # Simple message analysis
        analysis = {
            'message_type': 'unknown',
            'requires_reasoning': False,
            'complexity': 'low',
            'keywords': []
        }
        
        # Determine message type based on content
        if isinstance(content, str):
            content_lower = content.lower()
            
            if any(word in content_lower for word in ['analyze', 'reason', 'think']):
                analysis['message_type'] = 'reasoning_request'
                analysis['requires_reasoning'] = True
                analysis['complexity'] = 'high'
            elif any(word in content_lower for word in ['question', 'what', 'how', 'why']):
                analysis['message_type'] = 'question'
                analysis['requires_reasoning'] = True
                analysis['complexity'] = 'medium'
            elif any(word in content_lower for word in ['hello', 'hi', 'status']):
                analysis['message_type'] = 'greeting'
                analysis['complexity'] = 'low'
        
        return analysis
    
    async def _generate_response(self, message: AgentMessage, 
                               analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate response based on message analysis."""
        message_type = analysis.get('message_type', 'unknown')
        
        if message_type == 'reasoning_request':
            return {
                'type': 'reasoning_offer',
                'content': f"I can help with reasoning tasks. I have capabilities in {', '.join(self.get_capabilities())}.",
                'agent_id': self.agent_id
            }
        elif message_type == 'question':
            return {
                'type': 'reasoning_response',
                'content': "I'd be happy to help analyze that question. Please provide more details for thorough reasoning.",
                'agent_id': self.agent_id
            }
        elif message_type == 'greeting':
            return {
                'type': 'greeting_response',
                'content': f"Hello! I'm {self.agent_id}, a reasoning agent ready to help with analytical tasks.",
                'agent_id': self.agent_id
            }
        else:
            return {
                'type': 'acknowledgment',
                'content': "Message received. How can I assist with reasoning or analysis?",
                'agent_id': self.agent_id
            }
    
    async def initialize(self) -> bool:
        """Initialize the reasoning agent."""
        try:
            self._logger.info(f"Initializing ReasoningAgent {self.agent_id}")
            
            # Initialize knowledge base with basic reasoning concepts
            self._knowledge_base = {
                'reasoning_types': ['deductive', 'inductive', 'abductive'],
                'logical_operators': ['and', 'or', 'not', 'implies'],
                'analysis_methods': ['decomposition', 'synthesis', 'comparison'],
            }
            
            self.is_active = True
            self._logger.info(f"ReasoningAgent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize ReasoningAgent {self.agent_id}: {e}")
            return False