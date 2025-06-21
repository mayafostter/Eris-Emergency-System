"""
ERIS Base Agent - Factory for creating ERIS agents with Google ADK integration
UPDATED VERSION - Matches current agent naming and structure
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

# Google ADK imports
from google.adk.agents import Agent as LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# ERIS imports
from services import CloudServices
from utils.time_utils import SimulationPhase

logger = logging.getLogger(__name__)

class ERISAgentFactory:
    """
    Factory for creating ERIS agents using Google ADK.
    Provides standardized ERIS functionality through ADK agents.
    """
    
    @staticmethod
    def create_eris_agent(
        agent_id: str,
        agent_type: str,
        model: str = "gemini-2.0-flash",
        custom_tools: List = None
    ) -> LlmAgent:
        """
        Create an ERIS agent using ADK patterns.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type/category of the agent
            model: LLM model to use
            custom_tools: Additional tools for the agent
            
        Returns:
            Configured ADK LlmAgent
        """
        
        # Create ERIS-specific tools with FIXED signatures
        eris_tools = ERISAgentFactory._create_eris_tools()
        
        # Combine with custom tools if provided
        all_tools = eris_tools + (custom_tools or [])
        
        # Create agent instruction based on type
        instruction = ERISAgentFactory._get_agent_instruction(agent_type)
        
        # Create the ADK agent
        agent = LlmAgent(
            name=agent_id,
            model=model,
            description=f"ERIS {agent_type} agent for disaster simulation",
            instruction=instruction,
            tools=all_tools
        )
        
        logger.info(f"Created ERIS agent: {agent_id} ({agent_type})")
        return agent
    
    @staticmethod
    def _create_eris_tools():
        """Create standard ERIS tools for all agents - FIXED SIGNATURES"""
        
        def log_agent_action(action: str, details: str = "") -> Dict[str, str]:
            """
            Log an agent action during simulation.
            
            Args:
                action: Description of the action taken
                details: Additional details about the action (optional)
                
            Returns:
                Confirmation of logged action
            """
            logger.info(f"Agent action: {action} | Details: {details}")
            return {
                "status": "logged",
                "action": action,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        def request_coordination(agent_id: str, request_type: str, data: str = "{}") -> Dict[str, str]:
            """
            Request coordination with other agents or orchestrator.
            
            Args:
                agent_id: ID of the requesting agent
                request_type: Type of coordination needed
                data: Coordination request data as JSON string (optional)
                
            Returns:
                Coordination response
            """
            logger.info(f"Coordination request from {agent_id}: {request_type}")
            return {
                "status": "coordination_requested",
                "request_id": f"coord_{agent_id}_{datetime.utcnow().timestamp()}",
                "message": "Coordination request logged"
            }
        
        def get_simulation_context() -> Dict[str, Any]:
            """
            Get current simulation context information.
            
            Returns:
                Current simulation context
            """
            return {
                "current_phase": "impact",
                "simulation_time": "0.5",
                "disaster_status": "active"
            }
        
        def update_agent_metrics(metrics_json: str = "{}") -> Dict[str, str]:
            """
            Update agent-specific metrics.
            
            Args:
                metrics_json: JSON string of metrics to update (optional)
                
            Returns:
                Confirmation of metrics update
            """
            logger.info(f"Agent metrics updated: {metrics_json}")
            return {
                "status": "metrics_updated",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return [log_agent_action, request_coordination, get_simulation_context, update_agent_metrics]
    
    @staticmethod
    def _get_agent_instruction(agent_type: str) -> str:
        """Get type-specific instructions for agents"""
        
        base_instruction = """
        You are an ERIS disaster simulation agent. Your role is to model realistic responses during disaster scenarios.
        
        Use the provided tools to:
        - Log all significant actions you take
        - Request coordination when needed
        - Update your metrics regularly
        - Get simulation context when making decisions
        
        Always respond in character for your specific role and provide realistic, actionable responses.
        """
        
        type_instructions = {
            "emergency_response": base_instruction + """
            You are an Emergency Response Coordinator. Focus on:
            - Coordinating first responders
            - Managing emergency resources
            - Communicating with other agencies
            - Prioritizing life-saving operations
            """,
            
            "public_health": base_instruction + """
            You are a Public Health Official. Focus on:
            - Monitoring health impacts
            - Coordinating medical resources
            - Managing health communications
            - Tracking disease outbreak risks
            """,
            
            "infrastructure": base_instruction + """
            You are an Infrastructure Manager. Focus on:
            - Assessing infrastructure damage
            - Prioritizing repairs
            - Managing utility services
            - Coordinating with utilities
            """,
            
            "logistics": base_instruction + """
            You are a Logistics Coordinator. Focus on:
            - Managing supply chains
            - Coordinating resource distribution
            - Managing transportation
            - Tracking resource needs
            """,
            
            "communications": base_instruction + """
            You are a Communications Director. Focus on:
            - Public information management
            - Media relations
            - Community outreach
            - Emergency alerts
            """,
            
            "recovery": base_instruction + """
            You are a Recovery Coordinator. Focus on:
            - Long-term recovery planning
            - Community rebuilding
            - Economic recovery
            - Lessons learned documentation
            """
        }
        
        return type_instructions.get(agent_type, base_instruction)


# UPDATED: Factory functions for current agent implementations
def create_emergency_response_agent(cloud_services: CloudServices):
    """Create an Emergency Response agent - Updated import"""
    from agents.emergency_response_agent import create_emergency_response_agent
    return create_emergency_response_agent(cloud_services)

def create_public_health_agent(cloud_services: CloudServices):
    """Create a Public Health agent - Updated import"""
    from agents.public_health_agent import create_public_health_agent
    return create_public_health_agent(cloud_services)

def create_infrastructure_manager_agent(cloud_services: CloudServices):
    """Create an Infrastructure Manager agent - Updated import"""
    from agents.infrastructure_manager_agent import create_infrastructure_manager_agent
    return create_infrastructure_manager_agent(cloud_services)

def create_logistics_coordinator_agent(cloud_services: CloudServices):
    """Create a Logistics Coordinator agent - Updated import"""
    from agents.logistics_coordinator_agent import create_logistics_coordinator_agent
    return create_logistics_coordinator_agent(cloud_services)

def create_communications_director_agent(cloud_services: CloudServices):
    """Create a Communications Director agent - Updated import"""
    from agents.communications_director_agent import create_communications_director_agent
    return create_communications_director_agent(cloud_services)

def create_recovery_coordinator_agent(cloud_services: CloudServices):
    """Create a Recovery Coordinator agent - Updated import"""
    from agents.recovery_coordinator_agent import create_recovery_coordinator_agent
    return create_recovery_coordinator_agent(cloud_services)

# Legacy compatibility (for old orchestrator code)
def create_infrastructure_agent(cloud_services: CloudServices):
    """Legacy compatibility - redirects to infrastructure_manager_agent"""
    return create_infrastructure_manager_agent(cloud_services)

def create_logistics_agent(cloud_services: CloudServices):
    """Legacy compatibility - redirects to logistics_coordinator_agent"""
    return create_logistics_coordinator_agent(cloud_services)

def create_communications_agent(cloud_services: CloudServices):
    """Legacy compatibility - redirects to communications_director_agent"""
    return create_communications_director_agent(cloud_services)

def create_recovery_agent(cloud_services: CloudServices):
    """Legacy compatibility - redirects to recovery_coordinator_agent"""
    return create_recovery_coordinator_agent(cloud_services)
