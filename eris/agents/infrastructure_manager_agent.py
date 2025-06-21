"""
ERIS Infrastructure Manager Agent - Compact version
Critical systems oversight and restoration coordination
"""

import logging
from datetime import datetime
from typing import Dict, Any

from google.adk.agents import Agent as LlmAgent
from agents.base_agent import ERISAgentFactory
from services import CloudServices
from utils.time_utils import SimulationPhase

logger = logging.getLogger(__name__)

class InfrastructureManagerAgent:
    def __init__(self, cloud_services: CloudServices):
        self.cloud = cloud_services
        self.agent_id = "infrastructure_manager"
        self.agent_type = "infrastructure"
        
        # Infrastructure systems
        self.systems = {
            "power": {"capacity": 100, "status": "operational"},
            "water": {"capacity": 100, "status": "operational"},
            "communications": {"capacity": 100, "status": "operational"},
            "transportation": {"capacity": 100, "status": "operational"}
        }
        
        # Repair crews
        self.repair_crews = {"available": 6, "deployed": 0}
        self.active_repairs = 0
        
        # Simulation state
        self.simulation_id = None
        self.current_phase = SimulationPhase.IMPACT
        self.disaster_type = None
        self.disaster_severity = 5
        
        self.adk_agent = self._create_agent()
    
    def _create_agent(self) -> LlmAgent:
        def assess_infrastructure_damage(disaster_type: str, severity: int, areas: str) -> Dict[str, Any]:
            """Assess infrastructure damage from disaster"""
            # Damage patterns by disaster type
            damage_patterns = {
                'earthquake': {'power': 0.7, 'transportation': 0.8, 'communications': 0.5},
                'tsunami': {'power': 0.9, 'transportation': 0.9, 'water': 0.8},
                'flood': {'power': 0.6, 'transportation': 0.7, 'water': 0.8},
                'wildfire': {'power': 0.6, 'communications': 0.7}
            }
            
            patterns = damage_patterns.get(disaster_type, {'power': 0.5})
            severity_factor = severity / 10.0
            
            damage_assessment = {}
            critical_failures = []
            
            for system, system_data in self.systems.items():
                damage_prob = patterns.get(system, 0.3) * severity_factor
                
                if damage_prob > 0.8:
                    system_data["capacity"] = 0
                    system_data["status"] = "offline"
                    critical_failures.append(system)
                elif damage_prob > 0.6:
                    system_data["capacity"] = 25
                    system_data["status"] = "critical"
                elif damage_prob > 0.4:
                    system_data["capacity"] = 60
                    system_data["status"] = "degraded"
                
                damage_assessment[system] = {
                    "capacity": system_data["capacity"],
                    "status": system_data["status"]
                }
            
            return {
                "damage_assessment": damage_assessment,
                "critical_failures": critical_failures,
                "systems_affected": len([s for s in self.systems.values() if s["status"] != "operational"]),
                "overall_health": round(sum(s["capacity"] for s in self.systems.values()) / len(self.systems), 1)
            }
        
        def prioritize_repairs(repair_strategy: str = "critical_first") -> Dict[str, Any]:
            """Prioritize infrastructure repair operations"""
            # Find damaged systems
            damaged_systems = [
                system for system, data in self.systems.items()
                if data["status"] != "operational"
            ]
            
            if not damaged_systems:
                return {"repairs_needed": False, "message": "No repairs required"}
            
            # Prioritize repairs
            repair_order = []
            if repair_strategy == "critical_first":
                # Critical systems first
                critical = [s for s in damaged_systems if self.systems[s]["status"] == "offline"]
                degraded = [s for s in damaged_systems if self.systems[s]["status"] in ["critical", "degraded"]]
                repair_order = critical + degraded
            
            # Deploy crews
            repairs_scheduled = min(len(repair_order), self.repair_crews["available"])
            self.repair_crews["deployed"] = repairs_scheduled
            self.active_repairs = repairs_scheduled
            
            return {
                "repair_strategy": repair_strategy,
                "repairs_scheduled": repairs_scheduled,
                "repair_order": repair_order[:repairs_scheduled],
                "crews_deployed": repairs_scheduled,
                "estimated_completion": "24_hours"
            }
        
        def coordinate_utilities(utility_type: str, restoration_priority: str = "essential") -> Dict[str, Any]:
            """Coordinate utility restoration efforts"""
            if utility_type not in self.systems:
                return {"status": "failed", "error": "Unknown utility type"}
            
            system = self.systems[utility_type]
            
            # Simulate restoration progress
            if system["status"] != "operational":
                system["capacity"] = min(100, system["capacity"] + 30)
                if system["capacity"] >= 80:
                    system["status"] = "operational"
                elif system["capacity"] >= 50:
                    system["status"] = "degraded"
            
            return {
                "utility_type": utility_type,
                "current_capacity": system["capacity"],
                "status": system["status"],
                "restoration_priority": restoration_priority,
                "coordination_status": "active"
            }
        
        def monitor_system_health() -> Dict[str, Any]:
            """Monitor overall infrastructure system health"""
            overall_capacity = sum(s["capacity"] for s in self.systems.values()) / len(self.systems)
            
            operational_systems = len([s for s in self.systems.values() if s["status"] == "operational"])
            
            return {
                "overall_capacity": round(overall_capacity, 1),
                "operational_systems": operational_systems,
                "total_systems": len(self.systems),
                "active_repairs": self.active_repairs,
                "system_status": "healthy" if overall_capacity > 80 else "degraded" if overall_capacity > 50 else "critical"
            }
        
        return ERISAgentFactory.create_eris_agent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            custom_tools=[assess_infrastructure_damage, prioritize_repairs, 
                         coordinate_utilities, monitor_system_health]
        )
    
    async def initialize_for_simulation(self, simulation_id: str, disaster_type: str, severity: int, location: str):
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.disaster_severity = severity
        
        # Reset all systems to operational
        for system in self.systems.values():
            system["capacity"] = 100
            system["status"] = "operational"
        
        self.repair_crews["deployed"] = 0
        self.active_repairs = 0
        
        logger.info(f"Infrastructure Manager Agent initialized for {disaster_type}")
    
    async def process_phase(self, phase: SimulationPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        self.current_phase = phase
        
        # Simple phase processing
        if phase == SimulationPhase.IMPACT:
            # Apply damage based on disaster severity
            damage_factor = self.disaster_severity / 10
            for system in self.systems.values():
                system["capacity"] = max(0, 100 - (damage_factor * 60))
                if system["capacity"] < 50:
                    system["status"] = "critical"
                elif system["capacity"] < 80:
                    system["status"] = "degraded"
            
            self.active_repairs = 3
            result = {"action": "damage_assessment", "systems_affected": len(self.systems)}
            
        elif phase == SimulationPhase.RESPONSE:
            # Begin repairs
            for system in self.systems.values():
                if system["status"] != "operational":
                    system["capacity"] = min(100, system["capacity"] + 25)
                    if system["capacity"] >= 80:
                        system["status"] = "operational"
            
            result = {"action": "repair_operations", "repairs_active": self.active_repairs}
            
        else:  # RECOVERY
            # Complete restoration
            for system in self.systems.values():
                system["capacity"] = 100
                system["status"] = "operational"
            
            self.active_repairs = 0
            result = {"action": "restoration_complete", "status": "operational"}
        
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
        overall_capacity = sum(s["capacity"] for s in self.systems.values()) / len(self.systems)
        operational_count = len([s for s in self.systems.values() if s["status"] == "operational"])
        
        return {
            "overall_capacity": round(overall_capacity, 1),
            "operational_systems": operational_count,
            "total_systems": len(self.systems),
            "active_repairs": self.active_repairs,
            "crews_deployed": self.repair_crews["deployed"],
            "system_health": "healthy" if overall_capacity > 80 else "degraded"
        }
    
    async def _save_state(self, metrics: Dict[str, Any]):
        try:
            await self.cloud.firestore.save_agent_state(self.agent_id, self.simulation_id, {
                "metrics": metrics,
                "systems": self.systems,
                "phase": self.current_phase.value,
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")


def create_infrastructure_manager_agent(cloud_services: CloudServices) -> InfrastructureManagerAgent:
    return InfrastructureManagerAgent(cloud_services)
