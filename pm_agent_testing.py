# pm_agent_testing.py

"""
Test script for the Product Manager (PM) Agent, focused on ERPNext product scenarios.
"""

import os
import sys
import argparse
import logging
import yaml
import uuid
import asyncio
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_framework.rag.adapter import RAGAdapter
from agent_framework.rag.contextual_rag import ContextualRAG
from agent_framework.agents.pm_agent import PMAgent
from agent_framework import initialize_framework, setup_logging

def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}

def initialize_rag(config: Dict[str, Any]):
    rag_adapter = RAGAdapter()
    logging.info("Initialized RAG adapter")
    return rag_adapter

def create_pm_agent(framework_components, config: Dict[str, Any]):
    cm = framework_components['context_manager']
    tr = framework_components['tool_registry']
    ar = framework_components['agent_registry']
    rag_adapter = initialize_rag(config)
    contextual_rag = ContextualRAG(cm, rag_adapter)
    pm = PMAgent(
        agent_id=str(uuid.uuid4()),
        name=config.get('agents', {}).get('pm_agent', {}).get('name', 'ERPNext PM'),
        context_manager=cm,
        tool_registry=tr,
        rag=contextual_rag,
        llm_api_key=config.get('agents', {}).get('pm_agent', {}).get('llm_api_key'),
        llm_model_name=config.get('agents', {}).get('pm_agent', {}).get('llm_model_name', 'llama-70b-4096')
    )
    ar.register_agent(pm)
    ar.register_agent_type('pm_agent', PMAgent)
    return pm

def initialize_project_context(cm, project_name="ERPNext PM Test Project"):
    existing = cm.query_entities(entity_type="project", limit=1)
    if existing:
        return existing[0]
    project = cm.create_entity(
        entity_type="project",
        data={
            "name": project_name,
            "description": "ERPNext feature planning and PRD tests",
            "goals": [
                "Create and manage ERPNext requirements",
                "Generate user stories from requirements",
                "Plan a 3-month roadmap for the ERPNext rollout",
                "Generate a PRD for a custom ERPNext module",
                "Analyze requirement clarity and metrics"
            ],
            "metadata": {"created_date": "2025-05-03", "version": "1.0.0"}
        }
    )
    return project

async def test_requirements_management(pm_agent):
    print("\n--- Testing Requirements Management ---\n")
    prompts = [
        "Create a requirement: 'Automated Stock Reorder' that triggers purchase orders when stock falls below threshold.",
        "List all current requirements.",
        "Create a high-priority requirement: 'QR Code scanning on delivery note'.",
        "Prioritize requirements using a value-effort matrix."
    ]
    for p in prompts:
        resp = await send(pm_agent, p)
        print(resp.get('content'), "\n")

async def test_user_stories(pm_agent):
    print("\n--- Testing User Story Management ---\n")
    resp = await send(pm_agent, "Create user stories from requirement 'Automated Stock Reorder'.")
    print(resp.get('content'), "\n")
    resp = await send(pm_agent, "List all user stories.")
    print(resp.get('content'), "\n")

async def test_product_planning(pm_agent):
    print("\n--- Testing Roadmap & Sprint Planning ---\n")
    resp = await send(pm_agent, 
        "Create a 3-month roadmap for ERPNext rollout, focusing on Stock, Sales, and Accounting modules."
    )
    print(resp.get('content'), "\n")
    resp = await send(pm_agent, 
        "Plan Sprint 1 with capacity 40 story points covering Stock Reorder and QR Code scanning features."
    )
    print(resp.get('content'), "\n")

async def test_documentation(pm_agent):
    print("\n--- Testing PRD Generation ---\n")
    resp = await send(pm_agent, 
        "Generate a PRD for the 'Custom Equipment Maintenance' module in ERPNext."
    )
    print(resp.get('content'), "\n")
    resp = await send(pm_agent, "Generate the Requirements section only.")
    print(resp.get('content'), "\n")

async def test_analysis(pm_agent):
    print("\n--- Testing Analysis Capabilities ---\n")
    resp = await send(pm_agent, "Analyze all ERPNext requirements for clarity, completeness, and missing fields.")
    print(resp.get('content'), "\n")
    resp = await send(pm_agent, "Calculate metrics: number of requirements per priority and percent completed.")
    print(resp.get('content'), "\n")

async def send(pm_agent, content: str):
    data = {"content": content}
    analysis = await pm_agent.analyze_input(data)
    intent, plan = await pm_agent.determine_intent(analysis)
    exec_res = await pm_agent.execute_plan(plan)
    return await pm_agent.generate_response(exec_res)

async def run_pm_tests(pm_agent):
    print("\n=== Starting ERPNext PM Agent Tests ===\n")
    session = pm_agent.start_session()
    print("Session:", session)
    project = initialize_project_context(pm_agent.context_manager)
    pm_agent.working_memory.store("current_project", project.id)
    await test_requirements_management(pm_agent)
    await test_user_stories(pm_agent)
    await test_product_planning(pm_agent)
    await test_documentation(pm_agent)
    await test_analysis(pm_agent)
    pm_agent.end_session()
    print("\n=== Completed ERPNext PM Agent Tests ===\n")

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    fw = initialize_framework(cfg)
    pm = create_pm_agent(fw, cfg)
    await run_pm_tests(pm)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
