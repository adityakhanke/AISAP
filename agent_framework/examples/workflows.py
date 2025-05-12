"""
Example workflow definitions for the agent framework.
"""

import logging
from typing import Dict, Any

from ..workflow.definitions import Workflow, WorkflowState, WorkflowTransition

logger = logging.getLogger(__name__)

def create_feature_development_workflow() -> Workflow:
    """
    Create a feature development workflow.
    
    This workflow orchestrates the process of developing a feature from requirements to implementation.
    
    Returns:
        Feature development workflow
    """
    workflow = Workflow.create(
        name="Feature Development",
        description="End-to-end workflow for developing a new feature"
    )
    
    # Create states
    requirements_state = workflow.add_state(
        name="Requirements Gathering",
        is_initial=True,
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Gather and define requirements for the feature",
        auto_transition=False
    )
    
    specification_state = workflow.add_state(
        name="Specification",
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Create detailed specifications based on requirements",
        auto_transition=False
    )
    
    design_state = workflow.add_state(
        name="Design",
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Create technical design based on specifications",
        auto_transition=False
    )
    
    implementation_state = workflow.add_state(
        name="Implementation",
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Implement the feature based on the design",
        auto_transition=False
    )
    
    testing_state = workflow.add_state(
        name="Testing",
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Test the implemented feature",
        auto_transition=False
    )
    
    review_state = workflow.add_state(
        name="Review",
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Review the feature implementation",
        auto_transition=False
    )
    
    completion_state = workflow.add_state(
        name="Completion",
        is_final=True,
        agent_id=None,  # No agent needed for completion state
        instructions="Feature development completed",
        auto_transition=True
    )
    
    # Create transitions
    workflow.add_transition(
        from_state=requirements_state.id,
        to_state=specification_state.id,
        name="Requirements Complete",
        condition="${variables.requirements_complete} == true"
    )
    
    workflow.add_transition(
        from_state=specification_state.id,
        to_state=design_state.id,
        name="Specification Complete",
        condition="${variables.specification_complete} == true"
    )
    
    workflow.add_transition(
        from_state=design_state.id,
        to_state=implementation_state.id,
        name="Design Complete",
        condition="${variables.design_complete} == true"
    )
    
    workflow.add_transition(
        from_state=implementation_state.id,
        to_state=testing_state.id,
        name="Implementation Complete",
        condition="${variables.implementation_complete} == true"
    )
    
    workflow.add_transition(
        from_state=testing_state.id,
        to_state=review_state.id,
        name="Testing Complete",
        condition="${variables.testing_complete} == true"
    )
    
    workflow.add_transition(
        from_state=review_state.id,
        to_state=implementation_state.id,
        name="Changes Requested",
        condition="${variables.changes_requested} == true",
        actions=[
            {
                "type": "set_variable",
                "variable": "implementation_complete",
                "value": False
            },
            {
                "type": "set_variable",
                "variable": "testing_complete",
                "value": False
            },
            {
                "type": "set_variable",
                "variable": "changes_requested",
                "value": False
            }
        ]
    )
    
    workflow.add_transition(
        from_state=review_state.id,
        to_state=completion_state.id,
        name="Review Complete",
        condition="${variables.review_complete} == true"
    )
    
    return workflow


def create_requirement_workflow() -> Workflow:
    """
    Create a workflow for processing a single requirement.
    
    This workflow handles the process of refining, approving, and implementing a requirement.
    
    Returns:
        Requirement workflow
    """
    workflow = Workflow.create(
        name="Requirement Processing",
        description="Process for refining, approving, and implementing a requirement"
    )
    
    # Create states
    draft_state = workflow.add_state(
        name="Draft",
        is_initial=True,
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Refine the requirement draft",
        auto_transition=False
    )
    
    review_state = workflow.add_state(
        name="Review",
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Review the requirement for clarity and completeness",
        auto_transition=False
    )
    
    estimation_state = workflow.add_state(
        name="Estimation",
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Estimate effort and complexity",
        auto_transition=False
    )
    
    approval_state = workflow.add_state(
        name="Approval",
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Approve or reject the requirement",
        auto_transition=False
    )
    
    implementation_state = workflow.add_state(
        name="Implementation",
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Implement the requirement",
        auto_transition=False
    )
    
    validation_state = workflow.add_state(
        name="Validation",
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Validate the implementation against the requirement",
        auto_transition=False
    )
    
    completion_state = workflow.add_state(
        name="Completion",
        is_final=True,
        agent_id=None,  # No agent needed for completion state
        instructions="Requirement processing completed",
        auto_transition=True
    )
    
    # Create transitions
    workflow.add_transition(
        from_state=draft_state.id,
        to_state=review_state.id,
        name="Draft Complete",
        condition="${variables.draft_complete} == true"
    )
    
    workflow.add_transition(
        from_state=review_state.id,
        to_state=draft_state.id,
        name="Revisions Needed",
        condition="${variables.revisions_needed} == true",
        actions=[
            {
                "type": "set_variable",
                "variable": "draft_complete",
                "value": False
            },
            {
                "type": "set_variable",
                "variable": "revisions_needed",
                "value": False
            }
        ]
    )
    
    workflow.add_transition(
        from_state=review_state.id,
        to_state=estimation_state.id,
        name="Review Complete",
        condition="${variables.review_complete} == true"
    )
    
    workflow.add_transition(
        from_state=estimation_state.id,
        to_state=approval_state.id,
        name="Estimation Complete",
        condition="${variables.estimation_complete} == true"
    )
    
    workflow.add_transition(
        from_state=approval_state.id,
        to_state=draft_state.id,
        name="Rejected",
        condition="${variables.requirement_rejected} == true",
        actions=[
            {
                "type": "set_variable",
                "variable": "draft_complete",
                "value": False
            },
            {
                "type": "set_variable",
                "variable": "review_complete",
                "value": False
            },
            {
                "type": "set_variable",
                "variable": "estimation_complete",
                "value": False
            },
            {
                "type": "set_variable",
                "variable": "requirement_rejected",
                "value": False
            }
        ]
    )
    
    workflow.add_transition(
        from_state=approval_state.id,
        to_state=implementation_state.id,
        name="Approved",
        condition="${variables.requirement_approved} == true"
    )
    
    workflow.add_transition(
        from_state=implementation_state.id,
        to_state=validation_state.id,
        name="Implementation Complete",
        condition="${variables.implementation_complete} == true"
    )
    
    workflow.add_transition(
        from_state=validation_state.id,
        to_state=implementation_state.id,
        name="Validation Failed",
        condition="${variables.validation_failed} == true",
        actions=[
            {
                "type": "set_variable",
                "variable": "implementation_complete",
                "value": False
            },
            {
                "type": "set_variable",
                "variable": "validation_failed",
                "value": False
            }
        ]
    )
    
    workflow.add_transition(
        from_state=validation_state.id,
        to_state=completion_state.id,
        name="Validation Complete",
        condition="${variables.validation_complete} == true"
    )
    
    return workflow


def create_code_review_workflow() -> Workflow:
    """
    Create a code review workflow.
    
    This workflow orchestrates the process of reviewing and approving code.
    
    Returns:
        Code review workflow
    """
    workflow = Workflow.create(
        name="Code Review",
        description="Process for reviewing, providing feedback, and approving code changes"
    )
    
    # Create states
    submission_state = workflow.add_state(
        name="Submission",
        is_initial=True,
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Submit code for review",
        auto_transition=True
    )
    
    initial_review_state = workflow.add_state(
        name="Initial Review",
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Perform initial review of the code",
        auto_transition=False
    )
    
    automated_checks_state = workflow.add_state(
        name="Automated Checks",
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Run automated checks on the code",
        auto_transition=True
    )
    
    detailed_review_state = workflow.add_state(
        name="Detailed Review",
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Perform detailed review of the code",
        auto_transition=False
    )
    
    revisions_state = workflow.add_state(
        name="Revisions",
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Make revisions to the code based on feedback",
        auto_transition=False
    )
    
    approval_state = workflow.add_state(
        name="Approval",
        agent_id=None,  # Will be set when instantiating the workflow
        instructions="Approve or reject the code changes",
        auto_transition=False
    )
    
    completion_state = workflow.add_state(
        name="Completion",
        is_final=True,
        agent_id=None,  # No agent needed for completion state
        instructions="Code review completed",
        auto_transition=True
    )
    
    # Create transitions
    workflow.add_transition(
        from_state=submission_state.id,
        to_state=initial_review_state.id,
        name="Code Submitted",
        condition=None  # Automatic transition
    )
    
    workflow.add_transition(
        from_state=initial_review_state.id,
        to_state=automated_checks_state.id,
        name="Initial Review Complete",
        condition="${variables.initial_review_complete} == true"
    )
    
    workflow.add_transition(
        from_state=automated_checks_state.id,
        to_state=detailed_review_state.id,
        name="Checks Passed",
        condition="${variables.checks_passed} == true"
    )
    
    workflow.add_transition(
        from_state=automated_checks_state.id,
        to_state=revisions_state.id,
        name="Checks Failed",
        condition="${variables.checks_passed} == false",
        actions=[
            {
                "type": "set_variable",
                "variable": "revision_reason",
                "value": "Failed automated checks"
            }
        ]
    )
    
    workflow.add_transition(
        from_state=detailed_review_state.id,
        to_state=revisions_state.id,
        name="Changes Requested",
        condition="${variables.changes_requested} == true",
        actions=[
            {
                "type": "set_variable",
                "variable": "revision_reason",
                "value": "Changes requested in detailed review"
            }
        ]
    )
    
    workflow.add_transition(
        from_state=detailed_review_state.id,
        to_state=approval_state.id,
        name="Review Complete",
        condition="${variables.review_complete} == true"
    )
    
    workflow.add_transition(
        from_state=revisions_state.id,
        to_state=initial_review_state.id,
        name="Revisions Complete",
        condition="${variables.revisions_complete} == true",
        actions=[
            {
                "type": "set_variable",
                "variable": "initial_review_complete",
                "value": False
            },
            {
                "type": "set_variable",
                "variable": "checks_passed",
                "value": False
            },
            {
                "type": "set_variable",
                "variable": "review_complete",
                "value": False
            },
            {
                "type": "set_variable",
                "variable": "changes_requested",
                "value": False
            }
        ]
    )
    
    workflow.add_transition(
        from_state=approval_state.id,
        to_state=completion_state.id,
        name="Code Approved",
        condition="${variables.code_approved} == true"
    )
    
    workflow.add_transition(
        from_state=approval_state.id,
        to_state=revisions_state.id,
        name="Approval Denied",
        condition="${variables.code_approved} == false",
        actions=[
            {
                "type": "set_variable",
                "variable": "revision_reason",
                "value": "Approval denied"
            }
        ]
    )
    
    return workflow


# Dictionary mapping workflow names to creation functions
WORKFLOW_CREATORS = {
    "feature_development": create_feature_development_workflow,
    "requirement_processing": create_requirement_workflow,
    "code_review": create_code_review_workflow
}


def get_workflow_creator(workflow_name: str):
    """
    Get a workflow creator function by name.
    
    Args:
        workflow_name: Name of the workflow creator
        
    Returns:
        Workflow creator function
    """
    return WORKFLOW_CREATORS.get(workflow_name)


def list_available_workflows():
    """
    List all available workflow creators.
    
    Returns:
        List of workflow creator names
    """
    return list(WORKFLOW_CREATORS.keys())