#!/usr/bin/env python3
"""
Modular Agentic AI Architecture - Comprehensive Demo

This demo showcases all components of the modular agentic AI system:
- Core engine with service registry and event bus
- Multiple agents (reasoning, planning, execution)
- Various tools (web search, code execution, API client)
- Memory systems (vector and graph)
- Model adapters (simulated)
- Inter-component communication and coordination

Run this demo to see the full system in action.
"""

import asyncio
import logging
import sys
import os
import yaml
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add parent directory to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import core components
from modular_agentic_ai.core.engine import ExecutionEngine
from modular_agentic_ai.core.registry import ServiceRegistry
from modular_agentic_ai.core.event_bus import EventBus, EventPriority

# Import interfaces
from modular_agentic_ai.core.interfaces import MemoryEntry

# Import services
from modular_agentic_ai.services.agent_service.service import AgentService
from modular_agentic_ai.services.memory_service.service import MemoryService
from modular_agentic_ai.services.planning_service.service import PlanningService

# Import plugins
from modular_agentic_ai.plugins.agents.reasoning_agent import ReasoningAgent
from modular_agentic_ai.plugins.agents.planning_agent import PlanningAgent
from modular_agentic_ai.plugins.agents.execution_agent import ExecutionAgent

from modular_agentic_ai.plugins.tools.web_search_tool import WebSearchTool
from modular_agentic_ai.plugins.tools.code_exec_tool import CodeExecutionTool
from modular_agentic_ai.plugins.tools.api_client_tool import APIClientTool

from modular_agentic_ai.plugins.memory.vector_memory import VectorMemory
from modular_agentic_ai.plugins.memory.graph_memory import GraphMemory

# Import adapters
from modular_agentic_ai.adapters.openai_adapter import OpenAIAdapter
from modular_agentic_ai.adapters.anthropic_adapter import AnthropicAdapter
from modular_agentic_ai.adapters.local_model_adapter import LocalModelAdapter


class ModularAgenticAIDemo:
    """
    Comprehensive demo of the Modular Agentic AI Architecture.
    
    This demo initializes all components and runs various scenarios to showcase
    the system's capabilities and modular design.
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "../config/system_config.yaml"
        self.config = self._load_config()
        
        # Core components
        self.engine = None
        self.registry = None
        self.event_bus = None
        
        # Services
        self.agent_service = None
        self.memory_service = None
        self.planning_service = None
        
        # Agents
        self.reasoning_agent = None
        self.planning_agent = None
        self.execution_agent = None
        
        # Tools
        self.web_search_tool = None
        self.code_exec_tool = None
        self.api_client_tool = None
        
        # Memory systems
        self.vector_memory = None
        self.graph_memory = None
        
        # Adapters (simulated)
        self.openai_adapter = None
        self.anthropic_adapter = None
        self.local_adapter = None
        
        # Demo state
        self.demo_results = {}
        
        # Setup logging
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration."""
        try:
            config_file = os.path.join(os.path.dirname(__file__), self.config_path)
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {config_file}")
            # Return default config
            return {
                'system': {'name': 'Demo System', 'log_level': 'INFO'},
                'agents': {},
                'tools': {},
                'memory': {},
                'adapters': {}
            }
    
    def _setup_logging(self):
        """Setup logging for the demo."""
        log_level = self.config.get('system', {}).get('log_level', 'INFO')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('demo.log')
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize_system(self):
        """Initialize all system components."""
        self.logger.info("🚀 Initializing Modular Agentic AI System...")
        
        # Initialize core components
        await self._initialize_core()
        
        # Initialize services
        await self._initialize_services()
        
        # Initialize agents
        await self._initialize_agents()
        
        # Initialize tools
        await self._initialize_tools()
        
        # Initialize memory systems
        await self._initialize_memory()
        
        # Initialize adapters
        await self._initialize_adapters()
        
        # Register all components
        await self._register_components()
        
        # Initialize the engine
        await self.engine.initialize()
        
        self.logger.info("✅ System initialization completed!")
        
        # Display system status
        await self._display_system_status()
    
    async def _initialize_core(self):
        """Initialize core components."""
        self.logger.info("🔧 Initializing core components...")
        
        # Create event bus
        self.event_bus = EventBus()
        await self.event_bus.start()
        
        # Create registry
        self.registry = ServiceRegistry()
        
        # Create engine
        engine_config = self.config.get('engine', {})
        self.engine = ExecutionEngine(engine_config)
        self.engine.registry = self.registry
        self.engine.event_bus = self.event_bus
    
    async def _initialize_services(self):
        """Initialize services."""
        self.logger.info("🛠️  Initializing services...")
        
        # Agent service
        agent_config = self.config.get('services', {}).get('agent_service', {})
        self.agent_service = AgentService(agent_config, self.event_bus)
        await self.agent_service.initialize()
        
        # Memory service
        memory_config = self.config.get('services', {}).get('memory_service', {})
        self.memory_service = MemoryService(memory_config, self.event_bus)
        await self.memory_service.initialize()
        
        # Planning service
        planning_config = self.config.get('services', {}).get('planning_service', {})
        self.planning_service = PlanningService(planning_config, self.event_bus)
        await self.planning_service.initialize()
    
    async def _initialize_agents(self):
        """Initialize agents."""
        self.logger.info("🤖 Initializing agents...")
        
        # Reasoning agent
        reasoning_config = self.config.get('agents', {}).get('reasoning_agent', {}).get('config', {})
        self.reasoning_agent = ReasoningAgent('reasoning_agent_1', reasoning_config)
        await self.reasoning_agent.initialize()
        
        # Planning agent
        planning_config = self.config.get('agents', {}).get('planning_agent', {}).get('config', {})
        self.planning_agent = PlanningAgent('planning_agent_1', planning_config)
        await self.planning_agent.initialize()
        
        # Execution agent
        execution_config = self.config.get('agents', {}).get('execution_agent', {}).get('config', {})
        self.execution_agent = ExecutionAgent('execution_agent_1', execution_config)
        await self.execution_agent.initialize()
    
    async def _initialize_tools(self):
        """Initialize tools."""
        self.logger.info("🔨 Initializing tools...")
        
        # Web search tool
        web_search_config = self.config.get('tools', {}).get('web_search_tool', {}).get('config', {})
        self.web_search_tool = WebSearchTool('web_search_tool_1', web_search_config)
        await self.web_search_tool.initialize()
        
        # Code execution tool
        code_exec_config = self.config.get('tools', {}).get('code_execution_tool', {}).get('config', {})
        self.code_exec_tool = CodeExecutionTool('code_exec_tool_1', code_exec_config)
        await self.code_exec_tool.initialize()
        
        # API client tool
        api_client_config = self.config.get('tools', {}).get('api_client_tool', {}).get('config', {})
        self.api_client_tool = APIClientTool('api_client_tool_1', api_client_config)
        await self.api_client_tool.initialize()
    
    async def _initialize_memory(self):
        """Initialize memory systems."""
        self.logger.info("🧠 Initializing memory systems...")
        
        # Vector memory
        vector_config = self.config.get('memory', {}).get('vector_memory', {}).get('config', {})
        self.vector_memory = VectorMemory('vector_memory_1', vector_config)
        await self.vector_memory.connect()
        
        # Graph memory
        graph_config = self.config.get('memory', {}).get('graph_memory', {}).get('config', {})
        self.graph_memory = GraphMemory('graph_memory_1', graph_config)
        await self.graph_memory.connect()
    
    async def _initialize_adapters(self):
        """Initialize model adapters."""
        self.logger.info("🔌 Initializing model adapters...")
        
        # OpenAI adapter (simulated)
        openai_config = self.config.get('adapters', {}).get('openai', {}).get('config', {})
        self.openai_adapter = OpenAIAdapter(openai_config)
        
        # Anthropic adapter (simulated)
        anthropic_config = self.config.get('adapters', {}).get('anthropic', {}).get('config', {})
        self.anthropic_adapter = AnthropicAdapter(anthropic_config)
        
        # Local model adapter (simulated)
        local_config = self.config.get('adapters', {}).get('local_model', {}).get('config', {})
        self.local_adapter = LocalModelAdapter(local_config)
        await self.local_adapter.initialize()
    
    async def _register_components(self):
        """Register all components with the registry."""
        self.logger.info("📋 Registering components...")
        
        # Register services
        self.registry.register_component('agent_service', 'services', self.agent_service)
        self.registry.register_component('memory_service', 'services', self.memory_service)
        self.registry.register_component('planning_service', 'services', self.planning_service)
        
        # Register agents
        self.registry.register_component('reasoning_agent_1', 'agents', self.reasoning_agent)
        self.registry.register_component('planning_agent_1', 'agents', self.planning_agent)
        self.registry.register_component('execution_agent_1', 'agents', self.execution_agent)
        
        # Register tools
        self.registry.register_component('web_search_tool_1', 'tools', self.web_search_tool)
        self.registry.register_component('code_exec_tool_1', 'tools', self.code_exec_tool)
        self.registry.register_component('api_client_tool_1', 'tools', self.api_client_tool)
        
        # Register memory systems
        self.registry.register_component('vector_memory_1', 'memory', self.vector_memory)
        self.registry.register_component('graph_memory_1', 'memory', self.graph_memory)
        
        # Register adapters
        self.registry.register_component('openai_adapter', 'adapters', self.openai_adapter)
        self.registry.register_component('anthropic_adapter', 'adapters', self.anthropic_adapter)
        self.registry.register_component('local_adapter', 'adapters', self.local_adapter)
    
    async def _display_system_status(self):
        """Display comprehensive system status."""
        print("\n" + "="*80)
        print("🏛️  MODULAR AGENTIC AI SYSTEM STATUS")
        print("="*80)
        
        # System info
        system_status = self.engine.get_system_status()
        print(f"System Running: {system_status['is_running']}")
        print(f"Total Components: {system_status['components']['total_components']}")
        print(f"Active Components: {system_status['components']['active_components']}")
        
        # Component breakdown
        components = system_status['components']['components_by_type']
        print(f"\n📊 Component Breakdown:")
        for comp_type, count in components.items():
            print(f"  • {comp_type.title()}: {count}")
        
        # Available agents
        agents = self.engine.get_available_agents()
        print(f"\n🤖 Available Agents: {len(agents)}")
        for agent in agents:
            print(f"  • {agent['agent_id']}: {', '.join(agent['capabilities'])}")
        
        # Available tools
        tools = self.engine.get_available_tools()
        print(f"\n🔨 Available Tools: {len(tools)}")
        for tool in tools:
            print(f"  • {tool['tool_id']}: {tool.get('description', 'No description')[:50]}...")
        
        print("="*80 + "\n")
    
    async def run_demo_scenarios(self):
        """Run comprehensive demo scenarios."""
        self.logger.info("🎬 Starting demo scenarios...")
        
        print("🎭 RUNNING DEMO SCENARIOS")
        print("="*50)
        
        # Scenario 1: Agent reasoning and communication
        await self._demo_agent_reasoning()
        
        # Scenario 2: Tool usage and integration
        await self._demo_tool_usage()
        
        # Scenario 3: Memory storage and retrieval
        await self._demo_memory_systems()
        
        # Scenario 4: Planning and execution
        await self._demo_planning_execution()
        
        # Scenario 5: Multi-agent collaboration
        await self._demo_multi_agent_collaboration()
        
        # Scenario 6: Event-driven communication
        await self._demo_event_driven_communication()
        
        # Scenario 7: Model adapter integration
        await self._demo_model_adapters()
        
        # Display demo results
        await self._display_demo_results()
    
    async def _demo_agent_reasoning(self):
        """Demonstrate agent reasoning capabilities."""
        print("\n🧠 SCENARIO 1: Agent Reasoning")
        print("-" * 30)
        
        try:
            # Test reasoning agent
            reasoning_task = {
                'type': 'reasoning',
                'problem': 'Analyze the benefits and drawbacks of modular AI architecture',
                'parameters': {'depth': 'comprehensive'}
            }
            
            start_time = datetime.now()
            response = await self.reasoning_agent.process_task(reasoning_task)
            end_time = datetime.now()
            
            print(f"✅ Reasoning completed in {(end_time - start_time).total_seconds():.2f}s")
            print(f"Success: {response.success}")
            print(f"Confidence: {response.metadata.get('confidence', 'N/A')}")
            print(f"Reasoning Steps: {response.metadata.get('reasoning_steps', 'N/A')}")
            
            self.demo_results['agent_reasoning'] = {
                'success': response.success,
                'execution_time': (end_time - start_time).total_seconds(),
                'confidence': response.metadata.get('confidence', 0)
            }
            
        except Exception as e:
            print(f"❌ Agent reasoning failed: {e}")
            self.demo_results['agent_reasoning'] = {'success': False, 'error': str(e)}
    
    async def _demo_tool_usage(self):
        """Demonstrate tool usage capabilities."""
        print("\n🔨 SCENARIO 2: Tool Usage")
        print("-" * 25)
        
        # Web search tool demo
        try:
            search_params = {
                'query': 'artificial intelligence modular architecture',
                'max_results': 5
            }
            
            start_time = datetime.now()
            search_result = await self.web_search_tool.execute(search_params)
            end_time = datetime.now()
            
            print(f"🔍 Web Search: {search_result.status.value}")
            if search_result.data:
                results_count = len(search_result.data.get('results', []))
                print(f"   Found {results_count} results in {(end_time - start_time).total_seconds():.2f}s")
            
            self.demo_results['web_search'] = {
                'success': search_result.status.value == 'success',
                'results_count': len(search_result.data.get('results', [])) if search_result.data else 0
            }
            
        except Exception as e:
            print(f"❌ Web search failed: {e}")
            self.demo_results['web_search'] = {'success': False, 'error': str(e)}
        
        # Code execution tool demo
        try:
            code_params = {
                'code': 'print("Hello from modular AI system!")\nresult = 2 + 2\nprint(f"2 + 2 = {result}")',
                'language': 'python'
            }
            
            start_time = datetime.now()
            code_result = await self.code_exec_tool.execute(code_params)
            end_time = datetime.now()
            
            print(f"💻 Code Execution: {code_result.status.value}")
            if code_result.data and code_result.data.get('success'):
                print(f"   Executed in {(end_time - start_time).total_seconds():.2f}s")
            
            self.demo_results['code_execution'] = {
                'success': code_result.status.value == 'success',
                'execution_time': (end_time - start_time).total_seconds()
            }
            
        except Exception as e:
            print(f"❌ Code execution failed: {e}")
            self.demo_results['code_execution'] = {'success': False, 'error': str(e)}
    
    async def _demo_memory_systems(self):
        """Demonstrate memory storage and retrieval."""
        print("\n🧠 SCENARIO 3: Memory Systems")
        print("-" * 28)
        
        # Vector memory demo
        try:
            # Store memories
            memories = [
                MemoryEntry(
                    id="mem_1",
                    content="Modular architecture enables flexible AI system design",
                    timestamp=datetime.now(),
                    tags=["architecture", "AI", "modular"],
                    metadata={"importance": "high", "category": "design"},
                    importance_score=0.9
                ),
                MemoryEntry(
                    id="mem_2", 
                    content="Event-driven communication improves system responsiveness",
                    timestamp=datetime.now(),
                    tags=["events", "communication", "performance"],
                    metadata={"importance": "medium", "category": "implementation"},
                    importance_score=0.7
                )
            ]
            
            stored_count = 0
            for memory in memories:
                if await self.vector_memory.store(memory):
                    stored_count += 1
            
            print(f"📚 Vector Memory: Stored {stored_count}/{len(memories)} entries")
            
            # Search memories
            from modular_agentic_ai.core.interfaces import MemoryQuery
            query = MemoryQuery(
                query="modular AI architecture benefits",
                limit=5,
                similarity_threshold=0.3
            )
            
            search_results = await self.vector_memory.search(query)
            print(f"🔍 Search Results: {len(search_results.entries)} entries found")
            
            self.demo_results['vector_memory'] = {
                'stored': stored_count,
                'search_results': len(search_results.entries),
                'search_time': search_results.search_time
            }
            
        except Exception as e:
            print(f"❌ Vector memory failed: {e}")
            self.demo_results['vector_memory'] = {'success': False, 'error': str(e)}
        
        # Graph memory demo
        try:
            # Store in graph memory
            graph_stored = 0
            for memory in memories:
                if await self.graph_memory.store(memory):
                    graph_stored += 1
            
            print(f"🕸️  Graph Memory: Stored {graph_stored}/{len(memories)} entries")
            
            # Search graph memory
            graph_results = await self.graph_memory.search(query)
            print(f"🔍 Graph Search: {len(graph_results.entries)} entries found")
            
            self.demo_results['graph_memory'] = {
                'stored': graph_stored,
                'search_results': len(graph_results.entries),
                'search_time': graph_results.search_time
            }
            
        except Exception as e:
            print(f"❌ Graph memory failed: {e}")
            self.demo_results['graph_memory'] = {'success': False, 'error': str(e)}
    
    async def _demo_planning_execution(self):
        """Demonstrate planning and execution capabilities."""
        print("\n📋 SCENARIO 4: Planning & Execution")
        print("-" * 32)
        
        try:
            # Create a plan
            planning_task = {
                'type': 'create_plan',
                'goal': 'Implement a new AI feature',
                'requirements': {'priority': 2, 'timeline': '1 week'},
                'constraints': {'resources': 'limited', 'complexity': 'medium'},
                'strategy': 'sequential'
            }
            
            start_time = datetime.now()
            plan_response = await self.planning_agent.process_task(planning_task)
            end_time = datetime.now()
            
            if plan_response.success:
                plan_data = plan_response.result
                print(f"📝 Plan Created: '{plan_data.get('plan', {}).get('name', 'Unknown')}'")
                print(f"   Steps: {plan_data.get('plan', {}).get('steps_count', 0)}")
                print(f"   Duration: {plan_data.get('plan', {}).get('estimated_duration', 0):.0f}s")
                print(f"   Created in: {(end_time - start_time).total_seconds():.2f}s")
                
                # Execute some tasks
                execution_task = {
                    'type': 'execute',
                    'action_type': 'compute',
                    'parameters': {'operation': 'analyze', 'data': {'items': [1, 2, 3, 4, 5]}}
                }
                
                exec_response = await self.execution_agent.process_task(execution_task)
                print(f"⚡ Execution: {'Success' if exec_response.success else 'Failed'}")
                
                self.demo_results['planning_execution'] = {
                    'plan_created': True,
                    'steps_count': plan_data.get('plan', {}).get('steps_count', 0),
                    'execution_success': exec_response.success
                }
            else:
                print(f"❌ Planning failed: {plan_response.error_message}")
                self.demo_results['planning_execution'] = {'success': False}
                
        except Exception as e:
            print(f"❌ Planning & execution failed: {e}")
            self.demo_results['planning_execution'] = {'success': False, 'error': str(e)}
    
    async def _demo_multi_agent_collaboration(self):
        """Demonstrate multi-agent collaboration."""
        print("\n🤝 SCENARIO 5: Multi-Agent Collaboration")
        print("-" * 38)
        
        try:
            # Agent communication
            from modular_agentic_ai.core.interfaces import AgentMessage
            
            message = AgentMessage(
                content={'type': 'collaboration_request', 'task': 'joint analysis'},
                sender='reasoning_agent_1',
                timestamp=datetime.now().timestamp(),
                metadata={'priority': 'high'}
            )
            
            # Reasoning agent asks planning agent for collaboration
            start_time = datetime.now()
            collab_response = await self.planning_agent.communicate(message)
            end_time = datetime.now()
            
            print(f"💬 Agent Communication: {'Success' if collab_response.success else 'Failed'}")
            print(f"   Response time: {(end_time - start_time).total_seconds():.2f}s")
            
            if collab_response.success:
                response_data = collab_response.result
                print(f"   Response: {response_data.get('response', 'No response')[:50]}...")
            
            self.demo_results['multi_agent_collaboration'] = {
                'communication_success': collab_response.success,
                'response_time': (end_time - start_time).total_seconds()
            }
            
        except Exception as e:
            print(f"❌ Multi-agent collaboration failed: {e}")
            self.demo_results['multi_agent_collaboration'] = {'success': False, 'error': str(e)}
    
    async def _demo_event_driven_communication(self):
        """Demonstrate event-driven communication."""
        print("\n📡 SCENARIO 6: Event-Driven Communication")
        print("-" * 40)
        
        try:
            # Subscribe to events
            event_received = asyncio.Event()
            received_events = []
            
            async def event_handler(event):
                received_events.append(event)
                event_received.set()
            
            self.event_bus.subscribe('demo.test_event', event_handler)
            
            # Publish events
            event_data = {
                'message': 'Demo event from modular AI system',
                'timestamp': datetime.now().isoformat(),
                'demo_scenario': 'event_communication'
            }
            
            start_time = datetime.now()
            event_id = await self.event_bus.publish(
                'demo.test_event',
                event_data,
                'demo_system',
                EventPriority.NORMAL
            )
            
            # Wait for event processing
            await asyncio.wait_for(event_received.wait(), timeout=5.0)
            end_time = datetime.now()
            
            print(f"📨 Event Published: {event_id[:8]}...")
            print(f"📬 Event Received: {len(received_events)} events")
            print(f"   Processing time: {(end_time - start_time).total_seconds():.3f}s")
            
            self.demo_results['event_communication'] = {
                'event_published': True,
                'events_received': len(received_events),
                'processing_time': (end_time - start_time).total_seconds()
            }
            
        except Exception as e:
            print(f"❌ Event-driven communication failed: {e}")
            self.demo_results['event_communication'] = {'success': False, 'error': str(e)}
    
    async def _demo_model_adapters(self):
        """Demonstrate model adapter integration."""
        print("\n🔌 SCENARIO 7: Model Adapters")
        print("-" * 27)
        
        # OpenAI adapter demo
        try:
            messages = [{'role': 'user', 'content': 'Hello from modular AI system!'}]
            
            start_time = datetime.now()
            openai_response = await self.openai_adapter.chat_completion(
                messages=messages,
                max_tokens=50,
                temperature=0.7
            )
            end_time = datetime.now()
            
            print(f"🤖 OpenAI Adapter: {'Success' if openai_response else 'Failed'}")
            if openai_response:
                print(f"   Response time: {openai_response.get('response_time', 0):.2f}s")
                print(f"   Model: {openai_response.get('model', 'Unknown')}")
            
            self.demo_results['openai_adapter'] = {
                'success': bool(openai_response),
                'response_time': (end_time - start_time).total_seconds()
            }
            
        except Exception as e:
            print(f"❌ OpenAI adapter failed: {e}")
            self.demo_results['openai_adapter'] = {'success': False, 'error': str(e)}
        
        # Local model adapter demo
        try:
            start_time = datetime.now()
            local_response = await self.local_adapter.generate_text(
                prompt="Explain modular AI architecture",
                max_new_tokens=50,
                temperature=0.7
            )
            end_time = datetime.now()
            
            print(f"💻 Local Model: {'Success' if local_response else 'Failed'}")
            if local_response:
                print(f"   Generation time: {local_response.get('generation_time', 0):.2f}s")
                print(f"   Tokens: {local_response.get('tokens_generated', 0)}")
            
            self.demo_results['local_model'] = {
                'success': bool(local_response),
                'generation_time': (end_time - start_time).total_seconds(),
                'tokens_generated': local_response.get('tokens_generated', 0) if local_response else 0
            }
            
        except Exception as e:
            print(f"❌ Local model failed: {e}")
            self.demo_results['local_model'] = {'success': False, 'error': str(e)}
    
    async def _display_demo_results(self):
        """Display comprehensive demo results."""
        print("\n" + "="*80)
        print("📊 DEMO RESULTS SUMMARY")
        print("="*80)
        
        total_scenarios = len(self.demo_results)
        successful_scenarios = sum(1 for result in self.demo_results.values() 
                                 if result.get('success', True))
        
        print(f"Total Scenarios: {total_scenarios}")
        print(f"Successful: {successful_scenarios}")
        print(f"Success Rate: {(successful_scenarios/total_scenarios*100):.1f}%")
        
        print(f"\n📋 Detailed Results:")
        for scenario, result in self.demo_results.items():
            status = "✅" if result.get('success', True) else "❌"
            print(f"  {status} {scenario.replace('_', ' ').title()}")
            
            # Show key metrics
            if 'execution_time' in result:
                print(f"      ⏱️  Execution: {result['execution_time']:.3f}s")
            if 'response_time' in result:
                print(f"      ⏱️  Response: {result['response_time']:.3f}s")
            if 'results_count' in result:
                print(f"      📊 Results: {result['results_count']}")
            if 'stored' in result:
                print(f"      💾 Stored: {result['stored']} entries")
            if 'error' in result:
                print(f"      ❌ Error: {result['error']}")
        
        print("="*80)
    
    async def cleanup_system(self):
        """Cleanup all system components."""
        self.logger.info("🧹 Cleaning up system...")
        
        try:
            # Cleanup agents
            if self.reasoning_agent:
                await self.reasoning_agent.shutdown()
            if self.planning_agent:
                await self.planning_agent.shutdown()
            if self.execution_agent:
                await self.execution_agent.shutdown()
            
            # Cleanup tools
            if self.web_search_tool:
                await self.web_search_tool.cleanup()
            if self.code_exec_tool:
                await self.code_exec_tool.cleanup()
            if self.api_client_tool:
                await self.api_client_tool.cleanup()
            
            # Cleanup memory systems
            if self.vector_memory:
                await self.vector_memory.disconnect()
            if self.graph_memory:
                await self.graph_memory.disconnect()
            
            # Cleanup adapters
            if self.openai_adapter:
                await self.openai_adapter.cleanup()
            if self.anthropic_adapter:
                await self.anthropic_adapter.cleanup()
            if self.local_adapter:
                await self.local_adapter.cleanup()
            
            # Cleanup services
            if self.agent_service:
                await self.agent_service.shutdown()
            if self.memory_service:
                await self.memory_service.shutdown()
            if self.planning_service:
                await self.planning_service.shutdown()
            
            # Cleanup core
            if self.engine:
                await self.engine.shutdown()
            if self.event_bus:
                await self.event_bus.stop()
            
            self.logger.info("✅ System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")


async def main():
    """Main demo function."""
    print("🚀 MODULAR AGENTIC AI ARCHITECTURE DEMO")
    print("="*50)
    print("This demo showcases all components of the modular AI system")
    print("including agents, tools, memory, services, and adapters.")
    print("="*50 + "\n")
    
    demo = ModularAgenticAIDemo()
    
    try:
        # Initialize system
        await demo.initialize_system()
        
        # Wait a moment for system to stabilize
        await asyncio.sleep(1)
        
        # Run demo scenarios
        await demo.run_demo_scenarios()
        
        print("\n🎉 Demo completed successfully!")
        print("Check the logs for detailed information about each component.")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logging.exception("Demo error")
    finally:
        # Cleanup
        await demo.cleanup_system()
        print("\n👋 Demo finished. Thank you for exploring the Modular Agentic AI Architecture!")


if __name__ == "__main__":
    asyncio.run(main())