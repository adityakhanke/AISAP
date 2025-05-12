"""
Example script demonstrating how to initialize and use the agent framework.
"""

import os
import sys
import argparse
import logging
import yaml
import uuid
from typing import Dict, Any

# Add the parent directory to the path so we can import the agent_framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the agent framework

from agent_framework.rag.adapter import RAGAdapter
from agent_framework.rag.contextual_rag import ContextualRAG
from agent_framework.agents.pm_agent import PMAgent
from agent_framework.agents.dev_agent import DevAgent
from agent_framework.examples.workflows import get_workflow_creator, list_available_workflows
from agent_framework import initialize_framework
from rag_package.enhanced_rag import EnhancedRAG
from rag_package.rag import RAG

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(config: Dict[str, Any]):
    """
    Set up logging based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file')
    
    handlers = []
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )

def initialize_rag(config: Dict[str, Any]):
    """
    Initialize the RAG system.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RAG adapter
    """
    rag_config = config.get('rag', {})
    
    # Try to import the EnhancedRAG and RAG classes
    try:
        # Use EnhancedRAG if available
        rag_instance = EnhancedRAG(
            persist_directory=rag_config.get('persist_directory', './chroma_db'),
            embedding_model=rag_config.get('embedding_model', 'nomic-ai/nomic-embed-code'),
            cross_encoder_model=rag_config.get('cross_encoder_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
            chunking_strategy=rag_config.get('chunking_strategy', 'semantic'),
            chunk_size=rag_config.get('chunk_size', 500),
            chunk_overlap=rag_config.get('chunk_overlap', 50),
            hybrid_alpha=rag_config.get('hybrid_alpha', 0.7),
            enable_query_expansion=rag_config.get('enable_query_expansion', True),
            enable_hybrid_search=rag_config.get('enable_hybrid_search', True),
            enable_diversity=rag_config.get('enable_diversity', True),
            use_8bit=rag_config.get('use_8bit', False)
        )
        
        logging.info("Initialized EnhancedRAG instance")
    except ImportError:
        try:
            # Fall back to standard RAG
            rag_instance = RAG(
                persist_directory=rag_config.get('persist_directory', './chroma_db'),
                embedding_model=rag_config.get('embedding_model', 'nomic-ai/nomic-embed-code'),
                cross_encoder_model=rag_config.get('cross_encoder_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
                chunking_strategy=rag_config.get('chunking_strategy', 'semantic'),
                chunk_size=rag_config.get('chunk_size', 500),
                chunk_overlap=rag_config.get('chunk_overlap', 50),
                use_8bit=rag_config.get('use_8bit', False)
            )
            
            logging.info("Initialized standard RAG instance")
        except ImportError:
            logging.warning("Could not import RAG implementations. Using dummy RAG adapter.")
            rag_instance = None
    
    # Create RAG adapter
    return RAGAdapter(rag_instance=rag_instance)

def create_agents(framework_components, config: Dict[str, Any]):
    """
    Create agents based on configuration.
    
    Args:
        framework_components: Dictionary of framework components
        config: Configuration dictionary
        
    Returns:
        Dictionary of created agents
    """
    context_manager = framework_components['context_manager']
    agent_registry = framework_components['agent_registry']
    
    # Get agent configurations
    agent_configs = config.get('agents', {})
    
    # Create RAG adapter
    rag_adapter = initialize_rag(config)
    
    # Create contextual RAG
    contextual_rag = ContextualRAG(context_manager, rag_adapter)
    
    # Create PM agent
    pm_config = agent_configs.get('pm_agent', {})
    pm_agent = PMAgent(
        agent_id=str(uuid.uuid4()),
        name=pm_config.get('name', 'Product Manager'),
        context_manager=context_manager,
        tool_registry=framework_components['tool_registry'],
        rag=contextual_rag
    )
    
    # Create Dev agent
    dev_config = agent_configs.get('dev_agent', {})
    dev_agent = DevAgent(
        agent_id=str(uuid.uuid4()),
        name=dev_config.get('name', 'Developer'),
        context_manager=context_manager,
        tool_registry=framework_components['tool_registry'],
        rag=contextual_rag
    )
    
    # Register agents
    agent_registry.register_agent(pm_agent)
    agent_registry.register_agent(dev_agent)
    
    # Register agent types
    agent_registry.register_agent_type('pm_agent', PMAgent)
    agent_registry.register_agent_type('dev_agent', DevAgent)
    
    return {
        'pm_agent': pm_agent,
        'dev_agent': dev_agent
    }

def register_workflows(framework_components, workflow_names=None):
    """
    Register example workflows.
    
    Args:
        framework_components: Dictionary of framework components
        workflow_names: List of workflow names to register, or None for all
    """
    workflow_engine = framework_components['workflow_engine']
    
    if workflow_names is None:
        workflow_names = list_available_workflows()
    
    for workflow_name in workflow_names:
        workflow_creator = get_workflow_creator(workflow_name)
        if workflow_creator:
            workflow = workflow_creator()
            workflow_engine.register_workflow(workflow)
            logging.info(f"Registered workflow: {workflow.name}")
        else:
            logging.warning(f"Unknown workflow: {workflow_name}")

def initialize_project_context(context_manager):
    """
    Initialize the context with a project entity.
    
    Args:
        context_manager: Context manager
    """
    # Check if a project already exists
    existing_projects = context_manager.query_entities(
        entity_type="project",
        limit=1
    )
    
    if existing_projects:
        logging.info(f"Using existing project: {existing_projects[0].data.get('name')}")
        return existing_projects[0]
    
    # Create a new project
    project = context_manager.create_entity(
        entity_type="project",
        data={
            "name": "Agent Framework Demo",
            "description": "Demonstration of the agent framework capabilities",
            "goals": [
                "Showcase agent collaboration",
                "Demonstrate workflow orchestration",
                "Show context sharing between agents"
            ],
            "metadata": {
                "created_date": "2023-06-01",
                "version": "0.1.0"
            }
        }
    )
    
    logging.info(f"Created new project: {project.data.get('name')}")
    return project

def run_interaction_demo(agents, context_manager):
    """
    Run a simple interaction demo with the agents.
    
    Args:
        agents: Dictionary of agents
        context_manager: Context manager
    """
    print("\n=== Agent Framework Interaction Demo ===\n")
    
    # Get the PM and Dev agents
    pm_agent = agents['pm_agent']
    dev_agent = agents['dev_agent']
    
    # Start sessions for both agents
    pm_session = pm_agent.start_session()
    dev_session = dev_agent.start_session()
    
    # Step 1: PM Agent creates a requirement
    print("\n--- Step 1: PM Agent creates a requirement ---\n")
    pm_input = {
        "content": "Create a requirement for a new search feature that allows users to find documents using keywords and filters"
    }
    pm_response = pm_agent.process_input(pm_input)
    print(f"PM Agent: {pm_response.get('content')}")
    
    # Step A.2: Get the requirement from context
    requirements = context_manager.query_entities(
        entity_type="requirement",
        limit=1
    )
    
    if not requirements:
        print("No requirements created!")
        return
    
    requirement = requirements[0]
    requirement_id = requirement.id
    print(f"\nCreated requirement with ID: {requirement_id}")
    print(f"Title: {requirement.data.get('title')}")
    print(f"Description: {requirement.data.get('description')}")
    
    # Step 2: Dev Agent implements the requirement
    print("\n--- Step 2: Dev Agent implements the requirement ---\n")
    dev_input = {
        "content": f"Implement requirement with ID {requirement_id}"
    }
    dev_response = dev_agent.process_input(dev_input)
    print(f"Dev Agent: {dev_response.get('content')}")
    
    # Step 3: PM Agent creates a user story
    print("\n--- Step 3: PM Agent creates a user story ---\n")
    pm_input = {
        "content": "Create a user story for advanced search filters"
    }
    pm_response = pm_agent.process_input(pm_input)
    print(f"PM Agent: {pm_response.get('content')}")
    
    # Step 4: Dev Agent explains some code
    print("\n--- Step 4: Dev Agent explains some code ---\n")
    dev_input = {
        "content": """Explain this code:
```python
def search_documents(query, filters=None):
    results = []
    for doc in documents:
        if query.lower() in doc.content.lower():
            if filters is None or all(doc.metadata.get(k) == v for k, v in filters.items()):
                results.append(doc)
    return results
```"""
    }
    dev_response = dev_agent.process_input(dev_input)
    print(f"Dev Agent: {dev_response.get('content')}")
    
    # End sessions
    pm_agent.end_session()
    dev_agent.end_session()
    
    print("\n=== Demo Complete ===\n")

def run_workflow_demo(workflow_name, agents, framework_components):
    """
    Run a workflow demo.
    
    Args:
        workflow_name: Name of the workflow to run
        agents: Dictionary of agents
        framework_components: Dictionary of framework components
    """
    workflow_engine = framework_components['workflow_engine']
    
    # Get the workflow
    workflows = [w for w in workflow_engine.get_all_workflows() if w['name'].lower().replace(' ', '_') == workflow_name]
    
    if not workflows:
        print(f"Workflow '{workflow_name}' not found!")
        return
    
    workflow_id = workflows[0]['id']
    
    print(f"\n=== Running {workflows[0]['name']} Workflow ===\n")
    
    # Assign agents to workflow states
    workflow = workflow_engine.workflows[workflow_id]
    
    for state in workflow.states:
        if "requirement" in state.name.lower() or "review" in state.name.lower():
            state.agent_id = agents['pm_agent'].agent_id
        elif "implement" in state.name.lower() or "test" in state.name.lower() or "design" in state.name.lower():
            state.agent_id = agents['dev_agent'].agent_id
    
    # Start the workflow
    instance_id = workflow_engine.start_workflow(
        workflow_id=workflow_id,
        input_data={
            "title": "Search Feature",
            "description": "Implement a search feature with filters"
        }
    )
    
    print(f"Started workflow instance: {instance_id}")
    
    # Get current status
    status = workflow_engine.get_instance_status(instance_id)
    print(f"Current state: {status['current_state_name']}")
    
    # Manually advance the workflow for demo purposes
    print("\nAdvancing workflow...\n")
    
    # Set variables to trigger transitions
    if workflow_name == "feature_development":
        workflow_engine.active_instances[instance_id]["variables"]["requirements_complete"] = True
        workflow_engine.send_event(instance_id, "state_completed")
        print("Completed Requirements Gathering")
        
        status = workflow_engine.get_instance_status(instance_id)
        print(f"Current state: {status['current_state_name']}")
        
        workflow_engine.active_instances[instance_id]["variables"]["specification_complete"] = True
        workflow_engine.send_event(instance_id, "state_completed")
        print("Completed Specification")
        
        status = workflow_engine.get_instance_status(instance_id)
        print(f"Current state: {status['current_state_name']}")
    
    elif workflow_name == "requirement_processing":
        workflow_engine.active_instances[instance_id]["variables"]["draft_complete"] = True
        workflow_engine.send_event(instance_id, "state_completed")
        print("Completed Draft")
        
        status = workflow_engine.get_instance_status(instance_id)
        print(f"Current state: {status['current_state_name']}")
        
        workflow_engine.active_instances[instance_id]["variables"]["review_complete"] = True
        workflow_engine.send_event(instance_id, "state_completed")
        print("Completed Review")
        
        status = workflow_engine.get_instance_status(instance_id)
        print(f"Current state: {status['current_state_name']}")
    
    print("\n=== Workflow Demo Complete ===\n")

def main():
    """Main entry point for the example script."""
    parser = argparse.ArgumentParser(description="Initialize and demo the agent framework")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--demo-type", type=str, choices=["interaction", "workflow"], default="interaction", help="Type of demo to run")
    parser.add_argument("--workflow", type=str, help="Name of workflow to run (for workflow demo)")
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    if not os.path.isfile(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    
    config = load_config(config_path)
    
    # Set up logging
    setup_logging(config)
    
    # Initialize the framework
    framework_components = initialize_framework(config)

    # Create agents
    agents = create_agents(framework_components, config)
    
    # Register workflows
    # register_workflows(framework_components)
    
    # Initialize project context
    initialize_project_context(framework_components['context_manager'])
    
    # Run the requested demo
    if args.demo_type == "interaction":
        run_interaction_demo(agents, framework_components['context_manager'])
    elif args.demo_type == "workflow":
        workflow_name = args.workflow or "feature_development"
        run_workflow_demo(workflow_name, agents, framework_components)

if __name__ == "__main__":
    main()