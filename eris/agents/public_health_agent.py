"""
ERIS Public Health Agent - Compact version
Health crisis management and medical resource coordination
"""

import logging
from datetime import datetime
from typing import Dict, Any, List

from google.adk.agents import Agent as LlmAgent
from agents.base_agent import ERISAgentFactory
from services import CloudServices
from utils.time_utils import SimulationPhase

logger = logging.getLogger(__name__)

class PublicHealthAgent:
    def __init__(self, cloud_services: CloudServices):
        self.cloud = cloud_services
        self.agent_id = "public_health_official"
        self.agent_type = "public_health"
        
        # Medical resources
        self.medical_supplies = {"vaccines": 5000, "antibiotics": 2500, "ppe": 15000}
        self.allocated = {resource: 0 for resource in self.medical_supplies}
        
        # Health metrics
        self.disease_cases = 0
        self.vaccination_coverage = 0.0
        self.health_advisories_issued = 0
        self.water_safety_status = "safe"
        
        # Simulation state
        self.simulation_id = None
        self.current_phase = SimulationPhase.IMPACT
        self.disaster_type = None
        self.disaster_severity = 5
        
        self.adk_agent = self._create_agent()
    
    def _create_agent(self) -> LlmAgent:
        def assess_health_risks(disaster_type: str, severity: int, population: int) -> Dict[str, Any]:
            """Assess health risks from disaster"""
            # Simple risk calculation
            risk_multipliers = {
                'earthquake': 0.7, 'tsunami': 0.9, 'flood': 0.8, 'wildfire': 0.6
            }
            
            base_risk = risk_multipliers.get(disaster_type, 0.5) * (severity / 10)
            estimated_cases = int(population * base_risk * 0.01)  # 1% max affected
            
            return {
                "risk_level": "high" if base_risk > 0.7 else "medium" if base_risk > 0.4 else "low",
                "estimated_cases": estimated_cases,
                "population_at_risk": population,
                "interventions_needed": ["medical_supplies", "health_monitoring"]
            }
        
        def distribute_medical_resources(resource_type: str, quantity: int, locations: str) -> Dict[str, Any]:
            """Distribute medical resources"""
            if resource_type not in self.medical_supplies:
                return {"status": "failed", "error": "Unknown resource type"}
            
            available = self.medical_supplies[resource_type] - self.allocated[resource_type]
            allocated = min(quantity, available)
            self.allocated[resource_type] += allocated
            
            location_list = [loc.strip() for loc in locations.split(',')]
            
            return {
                "resource_type": resource_type,
                "allocated": allocated,
                "locations": location_list,
                "status": "complete" if allocated == quantity else "partial"
            }
        
        def issue_health_advisory(advisory_type: str, severity: str, message: str) -> Dict[str, Any]:
            """Issue public health advisory"""
            self.health_advisories_issued += 1
            
            # Update water safety if relevant
            if advisory_type == "water_safety":
                self.water_safety_status = "unsafe" if severity == "high" else "caution"
            
            return {
                "advisory_type": advisory_type,
                "severity": severity,
                "advisory_id": f"health_alert_{self.health_advisories_issued}",
                "estimated_reach": 150000,
                "status": "issued"
            }
        
        def monitor_health_status() -> Dict[str, Any]:
            """Monitor current health status"""
            # Simple disease surveillance
            if self.disaster_severity > 7:
                self.disease_cases += 20
            elif self.disaster_severity > 5:
                self.disease_cases += 10
            
            return {
                "total_disease_cases": self.disease_cases,
                "vaccination_coverage": round(self.vaccination_coverage * 100, 1),
                "water_safety_status": self.water_safety_status,
                "advisories_issued": self.health_advisories_issued,
                "health_system_status": "operational"
            }
        
        return ERISAgentFactory.create_eris_agent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            custom_tools=[assess_health_risks, distribute_medical_resources, 
                         issue_health_advisory, monitor_health_status]
        )
    
    async def initialize_for_simulation(self, simulation_id: str, disaster_type: str, severity: int, location: str):
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.disaster_severity = severity
        
        # Reset state
        for resource in self.allocated:
            self.allocated[resource] = 0
        
        self.disease_cases = 0
        self.vaccination_coverage = 0.0
        self.health_advisories_issued = 0
        self.water_safety_status = "safe"
        
        logger.info(f"Public Health Agent initialized for {disaster_type}")
    
    async def process_phase(self, phase: SimulationPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        self.current_phase = phase
        
        # Simple phase processing
        if phase == SimulationPhase.IMPACT:
            # Assess initial health impact
            self.disease_cases = self.disaster_severity * 5
            self.health_advisories_issued = 1
            result = {"action": "health_impact_assessment", "cases": self.disease_cases}
            
        elif phase == SimulationPhase.RESPONSE:
            # Distribute resources and issue advisories
            self.allocated["vaccines"] = 1000
            self.allocated["ppe"] = 3000
            self.vaccination_coverage = 0.6
            self.health_advisories_issued = 3
            result = {"action": "resource_distribution", "coverage": self.vaccination_coverage}
            
        else:  # RECOVERY
            # Normalize health status
            self.vaccination_coverage = 0.8
            self.water_safety_status = "safe"
            result = {"action": "health_normalization", "status": "improving"}
        
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
        total_supplies = sum(self.medical_supplies.values())
        total_allocated = sum(self.allocated.values())
        
        return {
            "resource_utilization": round((total_allocated / total_supplies) * 100, 1),
            "disease_cases": self.disease_cases,
            "vaccination_coverage": round(self.vaccination_coverage * 100, 1),
            "health_advisories": self.health_advisories_issued,
            "water_safety_status": self.water_safety_status,
            "health_system_status": "operational"
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


def create_public_health_agent(cloud_services: CloudServices) -> PublicHealthAgent:
    return PublicHealthAgent(cloud_services)
