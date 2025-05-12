"""
Test script for the ERPNext Developer (Dev) Agent.
Exercises core ERPNext customization and integration tasks.
"""

import os
import sys
import argparse
import logging
import yaml
import uuid
import asyncio
from typing import Dict, Any

# Add parent directory for agent framework imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_framework.rag.adapter import RAGAdapter
from agent_framework.rag.contextual_rag import ContextualRAG
from agent_framework.agents.dev_agent import DevAgent
from agent_framework.llm.agent_llm.dev_agent_llm import DevAgentLLMConfig
from agent_framework import initialize_framework, setup_logging


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration.
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}


def initialize_rag(config: Dict[str, Any]):
    """
    Initialize the RAG adapter for ERPNext knowledge base.
    """
    rag_adapter = RAGAdapter()
    logging.info("Initialized RAG adapter for ERPNext")
    return rag_adapter


def create_dev_agent(framework_components, config: Dict[str, Any]):
    """
    Instantiate the ERPNext DevAgent with LLM configuration.
    """
    cm = framework_components['context_manager']
    tr = framework_components['tool_registry']
    ar = framework_components['agent_registry']

    rag_adapter = initialize_rag(config)
    contextual_rag = ContextualRAG(cm, rag_adapter)

    dev_cfg = config.get('agents', {}).get('dev_agent', {})
    
    # Create DevAgentLLMConfig directly
    llm_config = DevAgentLLMConfig(
        model_name='codestral-2501',
        region='us-central1'
    )
    
    logging.info(f"Using Vertex AI LLM: {llm_config.model_name}")

    dev_agent = DevAgent(
        agent_id=str(uuid.uuid4()),
        name=dev_cfg.get('name', 'ERPNext Developer'),
        context_manager=cm,
        tool_registry=tr,
        rag=contextual_rag,
        llm_config=llm_config
    )
    ar.register_agent(dev_agent)
    ar.register_agent_type('dev_agent', DevAgent)
    return dev_agent


def initialize_project_context(cm, project_name="ERPNext Dev Project"):
    """
    Ensure a project entity exists for ERPNext tasks.
    """
    existing = cm.query_entities(entity_type="project", limit=1)
    if existing:
        logging.info("Using existing ERPNext project")
        return existing[0]

    project = cm.create_entity(
        entity_type="project",
        data={
            "name": project_name,
            "description": "ERPNext customization and integration tests",
            "goals": [
                "Create a new Doctype for Equipment Maintenance",
                "Write a server script for Sales Invoice due dates",
                "Write a client script for Purchase Order validation",
                "Consume ERPNext REST API in Python",
                "Generate a custom monthly sales report"
            ],
            "metadata": {"created_date": "2025-05-03", "version": "1.0.0"}
        }
    )
    logging.info(f"Created new project: {project.data.get('name')}")
    return project


async def test_doctype_generation(dev_agent):
    print("\n--- Testing Doctype Generation ---\n")
    prompt = (
        "Generate an ERPNext Doctype named 'Equipment Maintenance' with fields:\n"
        "- Equipment ID (Link to 'Equipment')\n"
        "- Maintenance Date (Date)\n"
        "- Technician (Link to 'Employee')\n"
        "- Notes (Text)\n"
        "Include the JSON meta to install via a Frappe app."
    )
    resp = await process_dev_input(dev_agent, prompt)
    print(resp.get('content'), "\n")


async def test_server_script(dev_agent):
    print("\n--- Testing Server Script Creation ---\n")
    prompt = (
        "Write a Frappe server script for 'Sales Invoice' that sets 'due_date' to 30 days after 'posting_date' on creation."
    )
    resp = await process_dev_input(dev_agent, prompt)
    print(resp.get('content'), "\n")


async def test_client_script(dev_agent):
    print("\n--- Testing Client Script Creation ---\n")
    prompt = (
        "Write a Frappe client script for 'Purchase Order' that alerts if 'order_date' < today when saving."
    )
    resp = await process_dev_input(dev_agent, prompt)
    print(resp.get('content'), "\n")


async def test_rest_api_call(dev_agent):
    print("\n--- Testing REST API Consumption ---\n")
    prompt = (
        "Provide a Python snippet using 'requests' to GET all 'Customer' records"
        " from ERPNext at 'https://erp.example.com' with API key auth, printing each name."
    )
    resp = await process_dev_input(dev_agent, prompt)
    print(resp.get('content'), "\n")


async def test_custom_report(dev_agent):
    print("\n--- Testing Custom Report Generation ---\n")
    prompt = (
        "Generate a Python script for an ERPNext custom report that shows total sales per month"
        " for the last 6 months, grouped by customer."
    )
    resp = await process_dev_input(dev_agent, prompt)
    print(resp.get('content'), "\n")


async def process_dev_input(dev_agent, content: str):
    data = {"content": content}
    analysis = await dev_agent.analyze_input(data)
    intent, plan = await dev_agent.determine_intent(analysis)
    exec_res = await dev_agent.execute_plan(plan)
    return await dev_agent.generate_response(exec_res)


async def run_dev_tests(dev_agent):
    print("\n=== Starting ERPNext Dev Agent Tests ===\n")
    session = dev_agent.start_session()
    print(f"Session: {session}\n")
    project = initialize_project_context(dev_agent.context_manager)
    dev_agent.working_memory.store("current_project", project.id)

    await test_doctype_generation(dev_agent)
    await test_server_script(dev_agent)
    await test_client_script(dev_agent)
    await test_rest_api_call(dev_agent)
    await test_custom_report(dev_agent)

    dev_agent.end_session()
    print("\n=== Completed ERPNext Dev Agent Tests ===\n")


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    fw = initialize_framework(cfg)
    dev = create_dev_agent(fw, cfg)
    await run_dev_tests(dev)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
