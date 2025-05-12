#!/usr/bin/env python
"""
Interactive Agent Session
This script provides an interactive shell to communicate with different agent types.
Currently supports the PM Agent (Product Manager).
"""

import os
import sys
import argparse
import asyncio
import logging
import uuid
import yaml
import cmd
from typing import Dict, Any, Optional

# Add the parent directory to the path so we can import the agent_framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the agent framework components
from agent_framework import initialize_framework, setup_logging
from agent_framework.rag.adapter import RAGAdapter
from agent_framework.rag.contextual_rag import ContextualRAG
from agent_framework.agents.pm_agent import PMAgent


class AgentShell(cmd.Cmd):
    """Interactive shell for communicating with agents."""
    
    intro = "Welcome to the Agent Shell. Type 'help' for a list of commands."
    prompt = "(Agent) > "
    
    def __init__(self, framework_components, config):
        """Initialize the shell with framework components."""
        super().__init__()
        self.framework_components = framework_components
        self.config = config
        self.agents = {}
        self.current_agent = None
        self.current_agent_name = None
        self.rag_adapter = None
        self.contextual_rag = None
        
        # Initialize RAG
        self._initialize_rag()
        
        # Create available agents
        self._create_agents()
        
    def _initialize_rag(self):
        """Initialize the RAG system."""
        try:
            context_manager = self.framework_components['context_manager']
            self.rag_adapter = RAGAdapter()
            self.contextual_rag = ContextualRAG(context_manager, self.rag_adapter)
            logging.info("Initialized RAG adapter")
        except Exception as e:
            logging.error(f"Error initializing RAG: {e}")
            print(f"Warning: RAG initialization failed: {e}")
    
    def _create_agents(self):
        """Create available agents based on configuration."""
        context_manager = self.framework_components['context_manager']
        tool_registry = self.framework_components['tool_registry']
        agent_registry = self.framework_components['agent_registry']
        
        # Create PM Agent if configured
        agent_configs = self.config.get('agents', {})
        if 'pm_agent' in agent_configs:
            pm_config = agent_configs.get('pm_agent', {})
            
            try:
                pm_agent = PMAgent(
                    agent_id=str(uuid.uuid4()),
                    name=pm_config.get('name', 'Product Manager'),
                    context_manager=context_manager,
                    tool_registry=tool_registry,
                    rag=self.contextual_rag,
                    llm_api_key=pm_config.get('llm_api_key'),
                    llm_model_name=pm_config.get('llm_model_name', 'llama-70b-4096')
                )
                
                # Register agent
                agent_registry.register_agent(pm_agent)
                agent_registry.register_agent_type('pm_agent', PMAgent)
                
                # Add to our agents dictionary
                self.agents['pm'] = pm_agent
                
                # Initialize session
                pm_agent.start_session()
                
                print(f"Created PM Agent: {pm_agent.name}")
                
                # Set as default agent if none selected
                if self.current_agent is None:
                    self.current_agent = pm_agent
                    self.current_agent_name = 'pm'
                    self.prompt = f"({pm_agent.name}) > "
                    
            except Exception as e:
                logging.error(f"Error creating PM Agent: {e}")
                print(f"Error creating PM Agent: {e}")
    
    def initialize_project_context(self, project_name="Agent Framework Project"):
        """Initialize project context for the current agent if it's a PM Agent."""
        if not self.current_agent or not isinstance(self.current_agent, PMAgent):
            print("No PM Agent selected. Please select a PM Agent first.")
            return
        
        context_manager = self.framework_components['context_manager']
        
        # Check if a project already exists
        existing_projects = context_manager.query_entities(
            entity_type="project",
            limit=1
        )
        
        if existing_projects:
            project = existing_projects[0]
            print(f"Using existing project: {project.data.get('name')}")
        else:
            # Create a new project
            project = context_manager.create_entity(
                entity_type="project",
                data={
                    "name": project_name,
                    "description": "Project managed through the Agent Shell",
                    "goals": [
                        "Manage product requirements and specifications",
                        "Create and track user stories",
                        "Generate documentation and roadmaps"
                    ],
                    "metadata": {
                        "created_date": "2025-05-02",
                        "version": "1.0.0"
                    }
                }
            )
            print(f"Created new project: {project.data.get('name')}")
        
        # Set as current project for PM Agent
        self.current_agent.working_memory.store("current_project", project.id)
        
        return project
    
    def do_agents(self, arg):
        """List available agents."""
        if not self.agents:
            print("No agents available.")
            return
            
        print("Available agents:")
        for key, agent in self.agents.items():
            current = " (current)" if agent == self.current_agent else ""
            print(f"  {key}: {agent.name}{current}")
    
    def do_use(self, arg):
        """Select an agent to use. Usage: use <agent_key>"""
        if not arg:
            print("Please specify an agent key. Use 'agents' to see available agents.")
            return
            
        if arg not in self.agents:
            print(f"Agent '{arg}' not found. Use 'agents' to see available agents.")
            return
            
        self.current_agent = self.agents[arg]
        self.current_agent_name = arg
        self.prompt = f"({self.current_agent.name}) > "
        print(f"Now using agent: {self.current_agent.name}")
    
    def do_project(self, arg):
        """Initialize or set current project. Usage: project [name]"""
        if not self.current_agent:
            print("No agent selected. Please select an agent first.")
            return
            
        if not isinstance(self.current_agent, PMAgent):
            print("Current agent does not support projects.")
            return
            
        project_name = arg if arg else "Agent Framework Project"
        self.initialize_project_context(project_name)
    
    async def process_input(self, content):
        """Process input through the current agent."""
        if not self.current_agent:
            print("No agent selected. Please select an agent first.")
            return
        
        input_data = {"content": content}
        
        try:
            # Analyze the input
            analysis_result = await self.current_agent.analyze_input(input_data)
            
            # Determine intent and plan
            intent, plan = await self.current_agent.determine_intent(analysis_result)
            
            # Execute the plan
            execution_result = await self.current_agent.execute_plan(plan)
            
            # Generate response
            response = await self.current_agent.generate_response(execution_result)
            
            return response
        except Exception as e:
            logging.error(f"Error processing input: {e}")
            return {"content": f"Error processing your request: {e}"}
    
    def default(self, line):
        """Process input as a message to the current agent."""
        if not line:
            return
            
        if not self.current_agent:
            print("No agent selected. Please select an agent first.")
            return
        
        # Process the input asynchronously
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        response = loop.run_until_complete(self.process_input(line))
        
        if response and "content" in response:
            print(f"\n{response['content']}\n")
    
    def do_exit(self, arg):
        """Exit the shell."""
        print("Goodbye!")
        return True
    
    def do_quit(self, arg):
        """Exit the shell."""
        return self.do_exit(arg)
    
    def do_EOF(self, arg):
        """Exit on Ctrl-D."""
        print("Goodbye!")
        return True


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}


def main():
    """Main entry point for the interactive agent shell."""
    parser = argparse.ArgumentParser(description="Interactive Agent Shell")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
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
    
    # Start the interactive shell
    shell = AgentShell(framework_components, config)
    
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logging.error(f"Error in command loop: {e}")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()