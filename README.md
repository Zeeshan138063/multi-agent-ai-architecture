# 🤖 Modular Agentic AI Architecture

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Design](https://img.shields.io/badge/architecture-plugin--based-lightgrey.svg)]()
[![Async](https://img.shields.io/badge/messaging-event--driven-orange.svg)]()

> **Author**: Zeehsan (PS.Brij Kishore Pandey)  
> **Modular blueprint for agentic AI systems** with plugins, services, interfaces, and async messaging at its core.

---

## 🧩 Overview

This project defines a truly modular agentic AI architecture using:

- Plugin system
- Service-oriented architecture
- Interface-driven contracts
- Async messaging via an internal event bus

It’s designed to **scale** and evolve, not break when adding new models, tools, or agents.

---

## 🚀 Key Highlights

### 🔌 Plugin System

- **Agents**, **tools**, **memory**, and **adapters** are all hot-swappable
- Interface-driven contracts ensure clean separation
- Supports **OpenAI**, **Anthropic**, and **local models** via adapters

🛠️ Plugin System Workflow:
1. Define interfaces for all component types
2. Implement plugins following interface contracts
3. Register components with the service registry
4. Use dependency injection for loose coupling

---

### 🧱 Service-Oriented Design

- Independent `agent`, `memory`, and `planning` services
- Each service is:
  - Configurable
  - Loosely coupled
  - Replaceable
- Managed via:
  - Internal **event bus**
  - Centralized **service registry**

---

## 🧠 Core Components

| File / Module     | Description                                |
|-------------------|--------------------------------------------|
| `interfaces/`     | Abstract contracts for agents, tools, memory |
| `engine.py`       | Orchestrates execution and flow            |
| `registry.py`     | Handles plugin/component discovery         |
| `event_bus.py`    | Async communication bus for messaging      |
| `services/`       | Independent, modular service definitions   |

---

## 🗂 Project Structure

```bash
modular_agentic_ai/
├── core/
│   ├── interfaces/
│   │   ├── agent_interface.py
│   │   ├── tool_interface.py
│   │   └── memory_interface.py
│   ├── engine.py
│   ├── registry.py
│   └── event_bus.py
│
├── services/
│   ├── agent_service/
│   │   ├── service.py
│   │   └── config.yaml
│   ├── memory_service/
│   │   ├── service.py
│   │   └── config.yaml
│   └── planning_service/
│       ├── service.py
│       └── config.yaml
│
├── plugins/
│   ├── agents/
│   │   ├── reasoning_agent.py
│   │   ├── planning_agent.py
│   │   └── execution_agent.py
│   ├── tools/
│   │   ├── web_search_tool.py
│   │   ├── code_exec_tool.py
│   │   └── api_client_tool.py
│   └── memory/
│       ├── vector_memory.py
│       └── graph_memory.py
│
├── adapters/
│   ├── openai_adapter.py
│   ├── anthropic_adapter.py
│   └── local_model_adapter.py
│
├── config/
│   └── system_config.yaml
│
├── examples/
│   └── modular_demo.py
│
├── requirements.txt
└── README.md
```

## 🗂 Design Principles


| Principle                    | Description                               |
| ---------------------------- | ----------------------------------------- |
| 🔌 Plugin-based architecture | Load modules dynamically at runtime       |
| 🔄 Interface segregation     | Prevent tight coupling between components |
| 📦 Dependency injection      | Swappable and testable modules            |
| ⚡ Event-driven messaging     | Services interact via async events        |
| 🧊 Hot-swappable components  | Replace without restarting the system     |



## 🧩 Plugin System Explained
Define interfaces for all component types
```(e.g., agent, tool, memory)```

Implement plugins following the interface contracts
```(e.g., execution_agent.py implements agent_interface.py)```

Register components with the central registry.py
```(used for dynamic discovery and hot-swap)```

Use dependency injection
```(so components remain loosely coupled)```


##  🧪 Example Use Cases
- Autonomous multi-agent task planning
- Tool-using LLMs with memory and APIs
- Swappable adapters for model experimentation
- LLM orchestration pipelines
