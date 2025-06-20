"""
ERIS Base Agent - Factory for creating ERIS agents with Google ADK integration
FIXED VERSION - Corrects tool signatures and ADK Runner API usage
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


# ERIS enhanced agent factory functions
def create_emergency_response_agent() -> LlmAgent:
    """Create an Emergency Response agent"""
    return ERISAgentFactory.create_eris_agent(
        agent_id="emergency_response_coordinator",
        agent_type="emergency_response",
        model="gemini-2.0-flash"
    )

def create_public_health_agent() -> LlmAgent:
    """Create a Public Health agent"""
    return ERISAgentFactory.create_eris_agent(
        agent_id="public_health_official",
        agent_type="public_health",
        model="gemini-2.0-flash"
    )

def create_infrastructure_agent() -> LlmAgent:
    """Create an Infrastructure agent"""
    return ERISAgentFactory.create_eris_agent(
        agent_id="infrastructure_manager",
        agent_type="infrastructure",
        model="gemini-2.0-flash"
    )

def create_logistics_agent() -> LlmAgent:
    """Create a Logistics agent"""
    return ERISAgentFactory.create_eris_agent(
        agent_id="logistics_coordinator",
        agent_type="logistics",
        model="gemini-2.0-flash"
    )

def create_communications_agent() -> LlmAgent:
    """Create a Communications agent"""
    return ERISAgentFactory.create_eris_agent(
        agent_id="communications_director",
        agent_type="communications",
        model="gemini-2.0-flash"
    )

def create_recovery_agent() -> LlmAgent:
    """Create a Recovery agent"""
    return ERISAgentFactory.create_eris_agent(
        agent_id="recovery_coordinator",
        agent_type="recovery",
        model="gemini-2.0-flash"
    )
