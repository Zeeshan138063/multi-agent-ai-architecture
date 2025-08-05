# 🚀 Running the Modular Agentic AI Demo

This guide explains how to set up and run the comprehensive demo of the Modular Agentic AI Architecture.

## Quick Start

### Option 1: Run Directly (Recommended)
```bash
# From the project root directory
python3 run_demo.py
```

### Option 2: Run as Module
```bash
# From the project root directory
python3 -m modular_agentic_ai.examples.modular_demo
```

### Option 3: Install and Run
```bash
# Install the package
pip install -e .

# Run the demo
modular-ai-demo
```

## Requirements

- Python 3.9 or higher
- All dependencies will be simulated (no external API keys needed for the demo)

## Demo Features

The demo showcases all components:

### 🤖 Agents
- **Reasoning Agent**: Logical reasoning and problem-solving
- **Planning Agent**: Task planning and strategic thinking  
- **Execution Agent**: Task execution and action implementation

### 🔨 Tools
- **Web Search Tool**: Information retrieval (simulated)
- **Code Execution Tool**: Safe code execution in multiple languages
- **API Client Tool**: HTTP requests and API integration

### 🧠 Memory Systems
- **Vector Memory**: Semantic search using embeddings
- **Graph Memory**: Relationship-based storage and traversal

### 🔌 Model Adapters
- **OpenAI Adapter**: GPT models integration (simulated)
- **Anthropic Adapter**: Claude models integration (simulated)
- **Local Model Adapter**: Local AI models support (simulated)

### ⚡ Core Features
- **Event Bus**: Async pub-sub communication
- **Service Registry**: Component discovery and management
- **Execution Engine**: Task orchestration and lifecycle management

## Demo Scenarios

The demo runs 7 comprehensive scenarios:

1. **Agent Reasoning**: Complex problem analysis
2. **Tool Usage**: Web search, code execution, API calls
3. **Memory Systems**: Storage and retrieval in vector/graph memory
4. **Planning & Execution**: Task decomposition and execution
5. **Multi-Agent Collaboration**: Inter-agent communication
6. **Event-Driven Communication**: Async messaging
7. **Model Adapters**: AI model integration

## Expected Output

The demo will:
- Initialize all system components
- Display system status and available components
- Run scenarios demonstrating different capabilities  
- Show performance metrics and results
- Clean up all resources

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the project root directory:
```bash
cd /path/to/multi-agent-ai-architecture
python3 run_demo.py
```

### Python Path Issues
The `run_demo.py` script automatically handles Python path setup. If you're still having issues, you can manually set the PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 modular_agentic_ai/examples/modular_demo.py
```

### Missing Dependencies
Install required dependencies:
```bash
pip install -r modular_agentic_ai/requirements.txt
```

## Configuration

The demo uses the configuration file at `modular_agentic_ai/config/system_config.yaml`. You can modify this file to:
- Enable/disable components
- Adjust performance parameters
- Configure logging levels
- Set custom parameters

## Next Steps

After running the demo:
1. Explore the codebase structure
2. Modify configuration in `system_config.yaml`
3. Add your own agents, tools, or memory systems
4. Integrate with real AI model APIs by providing API keys
5. Build your own applications using the modular architecture

## Support

If you encounter issues:
1. Check the logs in `demo.log`
2. Verify Python version (3.9+)
3. Ensure all files are present
4. Try running with `python3 -v` for verbose output

The demo is fully self-contained and doesn't require external services - all AI model calls are simulated for demonstration purposes.