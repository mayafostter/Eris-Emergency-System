"""
ERIS Logistics Coordinator Agent - Compact version
Supply chain and resource distribution management
"""

import logging
from datetime import datetime
from typing import Dict, Any

from google.adk.agents import Agent as LlmAgent
from agents.base_agent import ERISAgentFactory
from services import CloudServices
from utils.time_utils import SimulationPhase

logger = logging.getLogger(__name__)

class LogisticsCoordinatorAgent:
    def __init__(self, cloud_services: CloudServices):
        self.cloud = cloud_services
        self.agent_id = "logistics_coordinator"
        self.agent_type = "logistics"
        
        # Resource inventory
        self.resources = {
            "food": {"total": 15000, "allocated": 0},
            "water": {"total": 25000, "allocated": 0},
            "medical_supplies": {"total": 5000, "allocated": 0},
            "shelter_materials": {"total": 2000, "allocated": 0}
        }
        
        # Transportation
        self.transport = {"trucks": 15, "helicopters": 4, "boats": 6}
        self.transport_utilization = 0.0
        
        # Distribution points
        self.distribution_points = ["central_shelter", "evacuation_center", "community_center"]
        
        # Simulation state
        self.simulation_id = None
        self.current_phase = SimulationPhase.IMPACT
        self.disaster_type = None
        self.disaster_severity = 5
        
        self.adk_agent = self._create_agent()
    
    def _create_agent(self) -> LlmAgent:
        def distribute_resources(resource_type: str, quantity: int, locations: str, priority: str = "medium") -> Dict[str, Any]:
            """Distribute resources to specified locations"""
            if resource_type not in self.resources:
                return {"status": "failed", "error": "Unknown resource type"}
            
            resource = self.resources[resource_type]
            available = resource["total"] - resource["allocated"]
            allocated = min(quantity, available)
            resource["allocated"] += allocated
            
            location_list = [loc.strip() for loc in locations.split(',')]
            quantity_per_location = allocated // len(location_list) if location_list else 0
            
            return {
                "resource_type": resource_type,
                "requested": quantity,
                "allocated": allocated,
                "locations": location_list,
                "quantity_per_location": quantity_per_location,
                "status": "complete" if allocated == quantity else "partial"
            }
        
        def coordinate_supply_chain(urgency: str = "medium", scope: str = "local") -> Dict[str, Any]:
            """Coordinate supply chain operations"""
            # Calculate supply needs based on disaster severity
            supply_multiplier = self.disaster_severity / 5.0
            
            supply_needs = {
                "food": int(5000 * supply_multiplier),
                "water": int(8000 * supply_multiplier),
                "medical_supplies": int(1000 * supply_multiplier)
            }
            
            # Check for shortages
            shortages = {}
            for resource, needed in supply_needs.items():
                available = self.resources[resource]["total"] - self.resources[resource]["allocated"]
                if needed > available:
                    shortages[resource] = needed - available
            
            return {
                "urgency": urgency,
                "scope": scope,
                "supply_needs": supply_needs,
                "shortages": shortages,
                "procurement_needed": len(shortages) > 0,
                "status": "coordinated"
            }
        
        def optimize_transportation(destinations: str, cargo_type: str = "mixed") -> Dict[str, Any]:
            """Optimize transportation routes"""
            dest_list = [d.strip() for d in destinations.split(',')]
            
            # Simple transport allocation
            if len(dest_list) <= 5:
                transport_mode = "trucks"
                vehicles_needed = len(dest_list)
            elif "coastal" in destinations.lower():
                transport_mode = "boats"
                vehicles_needed = min(3, len(dest_list))
            else:
                transport_mode = "helicopters"
                vehicles_needed = min(2, len(dest_list))
            
            # Update utilization
            total_vehicles = sum(self.transport.values())
            self.transport_utilization = (vehicles_needed / total_vehicles) * 100
            
            return {
                "destinations": dest_list,
                "transport_mode": transport_mode,
                "vehicles_needed": vehicles_needed,
                "estimated_duration": "4_hours",
                "utilization": round(self.transport_utilization, 1),
                "status": "optimized"
            }
        
        def track_resource_status() -> Dict[str, Any]:
            """Track current resource status"""
            total_resources = sum(r["total"] for r in self.resources.values())
            total_allocated = sum(r["allocated"] for r in self.resources.values())
            utilization = (total_allocated / total_resources) * 100
            
            low_resources = [
                name for name, resource in self.resources.items()
                if (resource["total"] - resource["allocated"]) < (resource["total"] * 0.2)
            ]
            
            return {
                "resource_utilization": round(utilization, 1),
                "total_allocated": total_allocated,
                "low_resources": low_resources,
                "distribution_points_active": len(self.distribution_points),
                "transport_utilization": round(self.transport_utilization, 1),
                "status": "operational"
            }
        
        return ERISAgentFactory.create_eris_agent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            custom_tools=[distribute_resources, coordinate_supply_chain, 
                         optimize_transportation, track_resource_status]
        )
    
    async def initialize_for_simulation(self, simulation_id: str, disaster_type: str, severity: int, location: str):
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.disaster_severity = severity
        
        # Reset resource allocations
        for resource in self.resources.values():
            resource["allocated"] = 0
        
        self.transport_utilization = 0.0
        
        logger.info(f"Logistics Coordinator Agent initialized for {disaster_type}")
    
    async def process_phase(self, phase: SimulationPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        self.current_phase = phase
        
        # Simple phase processing
        if phase == SimulationPhase.IMPACT:
            # Initial resource distribution
            self.resources["food"]["allocated"] = 3000
            self.resources["water"]["allocated"] = 5000
            self.resources["medical_supplies"]["allocated"] = 1000
            self.transport_utilization = 60.0
            result = {"action": "emergency_distribution", "resources_deployed": True}
            
        elif phase == SimulationPhase.RESPONSE:
            # Sustained operations
            self.resources["food"]["allocated"] = 8000
            self.resources["water"]["allocated"] = 12000
            self.resources["shelter_materials"]["allocated"] = 800
            self.transport_utilization = 80.0
            result = {"action": "sustained_operations", "supply_chain_active": True}
            
        else:  # RECOVERY
            # Scale down operations
            for resource in self.resources.values():
                resource["allocated"] = max(0, resource["allocated"] - 2000)
            self.transport_utilization = 30.0
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
        total_resources = sum(r["total"] for r in self.resources.values())
        total_allocated = sum(r["allocated"] for r in self.resources.values())
        utilization = (total_allocated / total_resources) * 100
        
        return {
            "resource_utilization": round(utilization, 1),
            "total_allocated": total_allocated,
            "transport_utilization": round(self.transport_utilization, 1),
            "distribution_points": len(self.distribution_points),
            "supply_chain_status": "operational",
            "logistics_efficiency": 85.0
        }
    
    async def _save_state(self, metrics: Dict[str, Any]):
        try:
            await self.cloud.firestore.save_agent_state(self.agent_id, self.simulation_id, {
                "metrics": metrics,
                "resources": self.resources,
                "phase": self.current_phase.value,
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")


def create_logistics_coordinator_agent(cloud_services: CloudServices) -> LogisticsCoordinatorAgent:
    return LogisticsCoordinatorAgent(cloud_services)
