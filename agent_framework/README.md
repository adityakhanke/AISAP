# Agent Framework

An extensible framework for building multi-agent systems with shared context and workflow capabilities. This framework provides a solid foundation for creating specialized AI agents that can work together while maintaining a consistent contextual understanding.

## Features

- **Universal Context Management**: Centralized context system for sharing knowledge between agents
- **Base Agent Framework**: Extensible foundation for specialized agents
- **Workflow Engine**: Orchestrates complex multi-agent processes
- **RAG Integration**: Context-aware retrieval from documentation and codebase
- **Tool System**: Flexible tools that agents can use to perform tasks
- **Event System**: Communication between agents and workflow components

## Architecture

The Agent Framework architecture consists of the following core components:

1. **Context Management Layer**: Manages shared knowledge across agents
2. **Base Agent Framework**: Provides core capabilities for all agents
3. **Workflow Engine**: Orchestrates multi-agent processes
4. **Contextual RAG**: Enhances RAG with context-awareness
5. **Tool Registry**: Manages available tools for agents

## Installation

### Prerequisites

- Python 3.8 or higher
- RAG system (existing implementation of RAG or EnhancedRAG)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agent-framework.git
cd agent-framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure your existing RAG system is properly set up and accessible.

## Quick Start

1. Create a configuration file (`config.yaml`) or use the example in `agent_framework/examples/config.yaml`.

2. Run the example script:
```bash
python -m agent_framework.examples.main --config path/to/config.yaml
```

3. Try the interaction demo:
```bash
python -m agent_framework.examples.main --demo-type interaction
```

4. Or run a workflow demo:
```bash
python -m agent_framework.examples.main --demo-type workflow --workflow feature_development
```

## Creating Custom Agents

To create a custom agent, extend the `BaseAgent` class and implement the required methods:

```python
from agent_framework.core.agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self, agent_id, name, context_manager, tool_registry=None):
        super().__init__(agent_id, name, context_manager, tool_registry)
        self._register_tools()
    
    def _register_tools(self):
        # Register agent-specific tools
        self.tool_registry.register_function(
            name="my_custom_tool",
            description="Description of the tool",
            func=self._my_custom_tool
        )
    
    def analyze_input(self, input_data):
        # Implement input analysis
        pass
    
    def determine_intent(self, analysis_result):
        # Implement intent determination
        pass
    
    def execute_plan(self, plan):
        # Implement plan execution
        pass
    
    def generate_response(self, execution_result):
        # Implement response generation
        pass
    
    def _my_custom_tool(self, param1, param2):
        # Implement the custom tool
        pass
```

## Creating Custom Workflows

To create a custom workflow, use the workflow definition API:

```python
from agent_framework.workflow.definition import Workflow

# Create a workflow
workflow = Workflow.create(
    name="My Custom Workflow",
    description="Description of the workflow"
)

# Add states
state1 = workflow.add_state(
    name="State 1",
    is_initial=True,
    agent_id="agent_id_here",
    instructions="Instructions for the agent",
    auto_transition=False
)

state2 = workflow.add_state(
    name="State 2",
    agent_id="agent_id_here",
    instructions="Instructions for the agent",
    auto_transition=False
)

# Add transitions
workflow.add_transition(
    from_state=state1.id,
    to_state=state2.id,
    name="Transition Name",
    condition="${variables.some_condition} == true"
)

# Register the workflow
workflow_engine.register_workflow(workflow)
```

## Configuration

The framework is configured using a YAML file with the following sections:

- **rag**: Configuration for the RAG system
- **context**: Settings for the context management system
- **workflow**: Configuration for the workflow engine
- **agents**: Settings for specific agent types
- **security**: Security-related settings
- **logging**: Logging configuration

See the example in `agent_framework/examples/config.yaml` for a complete configuration.

## Core Components

### Context Management

The context management system maintains a shared understanding of entities and their relationships.

```python
# Create an entity
entity = context_manager.create_entity(
    entity_type="requirement",
    data={
        "title": "Search Feature",
        "description": "Implement a search feature",
        "priority": "high"
    }
)

# Create a relationship
context_manager.create_relationship(
    from_entity_id=entity.id,
    relation_type="assigned_to",
    to_entity_id=developer_entity.id
)

# Query entities
results = context_manager.query_entities(
    entity_type="requirement",
    filters={"priority": "high"}
)
```

### Tool Registry

The tool registry manages tools that agents can use.

```python
# Register a function as a tool
tool_registry.register_function(
    name="generate_code",
    description="Generate code based on a specification",
    func=generate_code_function
)

# Use a tool
result = agent.use_tool("generate_code", {
    "specification": "A function that calculates factorial",
    "language": "python"
})
```

### Workflow Engine

The workflow engine orchestrates multi-agent processes.

```python
# Start a workflow
instance_id = workflow_engine.start_workflow(
    workflow_id="workflow_id",
    input_data={
        "title": "Project Title",
        "description": "Project Description"
    }
)

# Send an event to a workflow
workflow_engine.send_event(
    instance_id=instance_id,
    event_type="approval_received",
    event_data={"approver": "Manager"}
)

# Get workflow status
status = workflow_engine.get_instance_status(instance_id)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.