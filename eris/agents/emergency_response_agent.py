"""
ERIS Emergency Response Agent - Compact version
Central command and control coordination for disaster response
"""

import logging
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass

from google.adk.agents import Agent as LlmAgent
from agents.base_agent import ERISAgentFactory
from services import CloudServices
from utils.time_utils import SimulationPhase

logger = logging.getLogger(__name__)

@dataclass
class EmergencyResource:
    resource_type: str
    total: int
    deployed: int
    response_time: float

class EmergencyResponseAgent:
    def __init__(self, cloud_services: CloudServices):
        self.cloud = cloud_services
        self.agent_id = "emergency_response_coordinator"
        self.agent_type = "emergency_response"
        
        # Core resources
        self.resources = {
            "fire_rescue": EmergencyResource("fire_rescue", 25, 0, 8.5),
            "police_units": EmergencyResource("police_units", 40, 0, 5.2),
            "ambulances": EmergencyResource("ambulances", 18, 0, 7.1),
            "search_rescue": EmergencyResource("search_rescue", 12, 0, 15.3)
        }
        
        # Metrics
        self.response_effectiveness = 0.0
        self.incidents_resolved = 0
        
        # Simulation state
        self.simulation_id = None
        self.current_phase = SimulationPhase.IMPACT
        self.disaster_type = None
        self.disaster_severity = 5
        
        self.adk_agent = self._create_agent()
    
    def _create_agent(self) -> LlmAgent:
        def deploy_resources(location: str, resource_types: str, priority: str = "medium") -> Dict[str, Any]:
            """Deploy emergency resources to incident location"""
            requested = [r.strip() for r in resource_types.split(',')]
            deployed = {}
            
            for resource_type in requested:
                if resource_type in self.resources:
                    resource = self.resources[resource_type]
                    available = resource.total - resource.deployed
                    
                    if available > 0:
                        units = min(available, 3 if priority == "critical" else 1)
                        resource.deployed += units
                        deployed[resource_type] = {
                            "units": units,
                            "response_time": resource.response_time
                        }
            
            return {
                "location": location,
                "deployed_resources": deployed,
                "status": "success" if deployed else "failed"
            }
        
        def coordinate_agencies(agencies: str, coordination_type: str = "standard") -> Dict[str, Any]:
            """Coordinate response across multiple agencies"""
            agency_list = [a.strip() for a in agencies.split(',')]
            
            self.response_effectiveness = min(1.0, 0.7 + (len(agency_list) * 0.05))
            
            return {
                "agencies": agency_list,
                "coordination_type": coordination_type,
                "effectiveness": round(self.response_effectiveness, 2),
                "status": "coordinated"
            }
        
        def assess_response_status() -> Dict[str, Any]:
            """Assess current emergency response status"""
            total_resources = sum(r.total for r in self.resources.values())
            deployed_resources = sum(r.deployed for r in self.resources.values())
            utilization = (deployed_resources / total_resources) * 100
            
            return {
                "resource_utilization": round(utilization, 1),
                "response_effectiveness": round(self.response_effectiveness * 100, 1),
                "incidents_resolved": self.incidents_resolved,
                "status": "operational" if utilization < 90 else "strained"
            }
        
        return ERISAgentFactory.create_eris_agent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            custom_tools=[deploy_resources, coordinate_agencies, assess_response_status]
        )
    
    async def initialize_for_simulation(self, simulation_id: str, disaster_type: str, severity: int, location: str):
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.disaster_severity = severity
        
        # Reset resources
        for resource in self.resources.values():
            resource.deployed = 0
        
        self.response_effectiveness = 0.0
        self.incidents_resolved = 0
        
        logger.info(f"Emergency Response Agent initialized for {disaster_type}")
    
    async def process_phase(self, phase: SimulationPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        self.current_phase = phase
        
        # Simple phase processing
        if phase == SimulationPhase.IMPACT:
            # Deploy initial resources
            for resource in self.resources.values():
                resource.deployed = min(resource.total, int(resource.total * 0.6))  # 60% deployment
            result = {"action": "initial_deployment", "resources_deployed": True}
            
        elif phase == SimulationPhase.RESPONSE:
            # Coordinate agencies
            self.response_effectiveness = 0.8
            self.incidents_resolved = 5
            result = {"action": "agency_coordination", "effectiveness": self.response_effectiveness}
            
        else:  # RECOVERY
            # Scale down operations
            for resource in self.resources.values():
                resource.deployed = max(0, resource.deployed - 2)
            result = {"action": "scale_down", "transition": "recovery"}
        
        # Generate metrics
        metrics = await self._generate_metrics()
        
        # Save state
        await self._save_state(metrics)
        
        return {
            "agent_id": self.agent_id,
            "phase": phase.value,
            "metrics": metrics,
            "actions": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _generate_metrics(self) -> Dict[str, Any]:
        total_resources = sum(r.total for r in self.resources.values())
        deployed_resources = sum(r.deployed for r in self.resources.values())
        
        return {
            "resource_utilization": round((deployed_resources / total_resources) * 100, 1),
            "response_effectiveness": round(self.response_effectiveness * 100, 1),
            "incidents_resolved": self.incidents_resolved,
            "active_resources": deployed_resources,
            "status": "operational"
        }
    
    async def _save_state(self, metrics: Dict[str, Any]):
        try:
            await self.cloud.firestore.save_agent_state(self.agent_id, self.simulation_id, {
                "metrics": metrics,
                "phase": self.current_phase.value,
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")


def create_emergency_response_agent(cloud_services: CloudServices) -> EmergencyResponseAgent:
    return EmergencyResponseAgent(cloud_services)
