"""
ERIS Infrastructure Manager Agent - Critical systems oversight and restoration
Manages power grids, transportation, communications infrastructure, and utilities
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

# Google ADK imports
from google.adk.agents import Agent as LlmAgent

# ERIS imports
from agents.base_agent import ERISAgentFactory
from services import CloudServices
from utils.time_utils import SimulationPhase

logger = logging.getLogger(__name__)

class InfrastructureStatus(Enum):
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"

class InfrastructureType(Enum):
    POWER_GRID = "power_grid"
    TELECOMMUNICATIONS = "telecommunications"
    TRANSPORTATION = "transportation"
    WATER_SYSTEM = "water_system"
    SEWAGE_SYSTEM = "sewage_system"
    FUEL_DISTRIBUTION = "fuel_distribution"
    EMERGENCY_SERVICES = "emergency_services"
    HEALTHCARE_FACILITIES = "healthcare_facilities"

@dataclass
class InfrastructureAsset:
    asset_id: str
    asset_type: InfrastructureType
    location: str
    criticality: str  # critical, high, medium, low
    capacity_percentage: float
    status: InfrastructureStatus
    estimated_repair_time: int  # hours
    dependencies: List[str]

@dataclass
class RepairOperation:
    operation_id: str
    asset_id: str
    priority: str
    crew_assigned: str
    estimated_completion: datetime
    resources_required: List[str]

class InfrastructureManagerAgent:
    """
    Infrastructure Manager Agent for ERIS disaster simulation.
    Oversees critical infrastructure systems and coordinates restoration efforts.
    """
    
    def __init__(self, cloud_services: CloudServices):
        self.cloud = cloud_services
        self.agent_id = "infrastructure_manager"
        self.agent_type = "infrastructure"
        
        # Infrastructure inventory
        self.infrastructure_assets = {
            "power_plant_main": InfrastructureAsset(
                "power_plant_main", InfrastructureType.POWER_GRID, "central_district",
                "critical", 100.0, InfrastructureStatus.OPERATIONAL, 0, []
            ),
            "power_substation_north": InfrastructureAsset(
                "power_substation_north", InfrastructureType.POWER_GRID, "northern_zone",
                "high", 100.0, InfrastructureStatus.OPERATIONAL, 0, ["power_plant_main"]
            ),
            "power_substation_south": InfrastructureAsset(
                "power_substation_south", InfrastructureType.POWER_GRID, "southern_zone",
                "high", 100.0, InfrastructureStatus.OPERATIONAL, 0, ["power_plant_main"]
            ),
            "telecom_tower_central": InfrastructureAsset(
                "telecom_tower_central", InfrastructureType.TELECOMMUNICATIONS, "central_district",
                "critical", 100.0, InfrastructureStatus.OPERATIONAL, 0, ["power_plant_main"]
            ),
            "telecom_tower_coastal": InfrastructureAsset(
                "telecom_tower_coastal", InfrastructureType.TELECOMMUNICATIONS, "coastal_area",
                "high", 100.0, InfrastructureStatus.OPERATIONAL, 0, ["power_substation_south"]
            ),
            "airport_main": InfrastructureAsset(
                "airport_main", InfrastructureType.TRANSPORTATION, "airport_zone",
                "critical", 100.0, InfrastructureStatus.OPERATIONAL, 0, ["power_substation_north"]
            ),
            "port_facility": InfrastructureAsset(
                "port_facility", InfrastructureType.TRANSPORTATION, "port_district",
                "critical", 100.0, InfrastructureStatus.OPERATIONAL, 0, ["power_substation_south"]
            ),
            "highway_main": InfrastructureAsset(
                "highway_main", InfrastructureType.TRANSPORTATION, "city_wide",
                "high", 100.0, InfrastructureStatus.OPERATIONAL, 0, []
            ),
            "water_treatment_plant": InfrastructureAsset(
                "water_treatment_plant", InfrastructureType.WATER_SYSTEM, "utility_district",
                "critical", 100.0, InfrastructureStatus.OPERATIONAL, 0, ["power_plant_main"]
            ),
            "sewage_treatment_plant": InfrastructureAsset(
                "sewage_treatment_plant", InfrastructureType.SEWAGE_SYSTEM, "utility_district",
                "high", 100.0, InfrastructureStatus.OPERATIONAL, 0, ["power_plant_main"]
            ),
            "fuel_depot": InfrastructureAsset(
                "fuel_depot", InfrastructureType.FUEL_DISTRIBUTION, "industrial_zone",
                "high", 100.0, InfrastructureStatus.OPERATIONAL, 0, []
            ),
            "hospital_power_backup": InfrastructureAsset(
                "hospital_power_backup", InfrastructureType.HEALTHCARE_FACILITIES, "medical_district",
                "critical", 100.0, InfrastructureStatus.OPERATIONAL, 0, []
            )
        }
        
        # Repair operations and crews
        self.active_repairs = []
        self.available_crews = {
            "power_crew_1": {"specialty": "electrical", "available": True, "current_job": None},
            "power_crew_2": {"specialty": "electrical", "available": True, "current_job": None},
            "telecom_crew_1": {"specialty": "telecommunications", "available": True, "current_job": None},
            "civil_crew_1": {"specialty": "civil_engineering", "available": True, "current_job": None},
            "civil_crew_2": {"specialty": "civil_engineering", "available": True, "current_job": None},
            "emergency_crew": {"specialty": "general", "available": True, "current_job": None}
        }
        
        # Infrastructure metrics
        self.overall_infrastructure_health = 100.0
        self.power_grid_availability = 100.0
        self.communications_availability = 100.0
        self.transportation_availability = 100.0
        self.utilities_availability = 100.0
        
        # Simulation state
        self.simulation_id = None
        self.current_phase = SimulationPhase.IMPACT
        self.disaster_type = None
        self.disaster_severity = 5
        
        # Create the ADK agent
        self.adk_agent = self._create_infrastructure_agent()
        
    def _create_infrastructure_agent(self) -> LlmAgent:
        """Create the ADK agent for infrastructure management"""
        
        def assess_infrastructure_damage(disaster_type: str, severity: int, affected_areas: str) -> Dict[str, Any]:
            """
            Assess infrastructure damage from disaster impact.
            
            Args:
                disaster_type: Type of disaster affecting infrastructure
                severity: Disaster severity (1-10)
                affected_areas: Comma-separated list of affected geographic areas
                
            Returns:
                Infrastructure damage assessment and priority repair list
            """
            areas = [area.strip() for area in affected_areas.split(',')]
            
            # Disaster-specific damage patterns
            damage_patterns = {
                'earthquake': {
                    InfrastructureType.POWER_GRID: 0.7,
                    InfrastructureType.TELECOMMUNICATIONS: 0.5,
                    InfrastructureType.TRANSPORTATION: 0.8,
                    InfrastructureType.WATER_SYSTEM: 0.6,
                    InfrastructureType.SEWAGE_SYSTEM: 0.6
                },
                'tsunami': {
                    InfrastructureType.POWER_GRID: 0.9,
                    InfrastructureType.TELECOMMUNICATIONS: 0.8,
                    InfrastructureType.TRANSPORTATION: 0.9,
                    InfrastructureType.WATER_SYSTEM: 0.8,
                    InfrastructureType.FUEL_DISTRIBUTION: 0.7
                },
                'flood': {
                    InfrastructureType.POWER_GRID: 0.6,
                    InfrastructureType.TELECOMMUNICATIONS: 0.4,
                    InfrastructureType.TRANSPORTATION: 0.7,
                    InfrastructureType.WATER_SYSTEM: 0.8,
                    InfrastructureType.SEWAGE_SYSTEM: 0.9
                },
                'hurricane': {
                    InfrastructureType.POWER_GRID: 0.8,
                    InfrastructureType.TELECOMMUNICATIONS: 0.7,
                    InfrastructureType.TRANSPORTATION: 0.5,
                    InfrastructureType.FUEL_DISTRIBUTION: 0.4
                },
                'wildfire': {
                    InfrastructureType.POWER_GRID: 0.6,
                    InfrastructureType.TELECOMMUNICATIONS: 0.7,
                    InfrastructureType.TRANSPORTATION: 0.3,
                    InfrastructureType.FUEL_DISTRIBUTION: 0.5
                }
            }
            
            patterns = damage_patterns.get(disaster_type, damage_patterns['earthquake'])
            severity_factor = severity / 10.0
            
            damage_assessment = {}
            critical_failures = []
            repair_priorities = []
            
            for asset_id, asset in self.infrastructure_assets.items():
                # Check if asset is in affected area
                asset_affected = any(area.lower() in asset.location.lower() for area in areas) or \
                               asset.location.lower() in [area.lower() for area in areas]
                
                if asset_affected:
                    # Calculate damage based on disaster type and severity
                    damage_probability = patterns.get(asset.asset_type, 0.3) * severity_factor
                    
                    # Apply criticality modifier
                    if asset.criticality == "critical":
                        damage_probability *= 1.2
                    elif asset.criticality == "low":
                        damage_probability *= 0.7
                    
                    # Determine damage level
                    if damage_probability > 0.8:
                        asset.status = InfrastructureStatus.OFFLINE
                        asset.capacity_percentage = 0.0
                        asset.estimated_repair_time = 24 + (severity * 2)
                        critical_failures.append(asset_id)
                    elif damage_probability > 0.6:
                        asset.status = InfrastructureStatus.CRITICAL
                        asset.capacity_percentage = 25.0
                        asset.estimated_repair_time = 12 + severity
                    elif damage_probability > 0.4:
                        asset.status = InfrastructureStatus.DEGRADED
                        asset.capacity_percentage = 60.0
                        asset.estimated_repair_time = 6 + (severity // 2)
                    
                    damage_assessment[asset_id] = {
                        "asset_type": asset.asset_type.value,
                        "status": asset.status.value,
                        "capacity_percentage": asset.capacity_percentage,
                        "estimated_repair_time_hours": asset.estimated_repair_time,
                        "criticality": asset.criticality,
                        "location": asset.location
                    }
                    
                    # Add to repair priorities if damaged
                    if asset.status != InfrastructureStatus.OPERATIONAL:
                        priority_score = (
                            (4 if asset.criticality == "critical" else 
                             3 if asset.criticality == "high" else 
                             2 if asset.criticality == "medium" else 1) * 
                            (4 if asset.status == InfrastructureStatus.OFFLINE else
                             3 if asset.status == InfrastructureStatus.CRITICAL else 2)
                        )
                        
                        repair_priorities.append({
                            "asset_id": asset_id,
                            "priority_score": priority_score,
                            "asset_type": asset.asset_type.value,
                            "estimated_repair_hours": asset.estimated_repair_time
                        })
            
            # Sort repair priorities
            repair_priorities.sort(key=lambda x: x["priority_score"], reverse=True)
            
            # Calculate overall system health
            self._update_system_availability()
            
            result = {
                "total_assets_assessed": len(self.infrastructure_assets),
                "damaged_assets": len(damage_assessment),
                "critical_failures": critical_failures,
                "damage_assessment": damage_assessment,
                "repair_priorities": repair_priorities[:10],  # Top 10 priorities
                "overall_infrastructure_health": round(self.overall_infrastructure_health, 1),
                "system_availability": {
                    "power_grid": round(self.power_grid_availability, 1),
                    "communications": round(self.communications_availability, 1),
                    "transportation": round(self.transportation_availability, 1),
                    "utilities": round(self.utilities_availability, 1)
                },
                "immediate_action_required": len(critical_failures) > 0
            }
            
            logger.info(f"Infrastructure damage assessed: {len(damage_assessment)} assets damaged, "
                       f"{len(critical_failures)} critical failures")
            return result
        
        def prioritize_infrastructure_repairs(repair_strategy: str = "critical_first", available_resources: str = "all") -> Dict[str, Any]:
            """
            Prioritize and schedule infrastructure repair operations.
            
            Args:
                repair_strategy: Strategy for prioritization (critical_first, cascading_impact, economic_priority)
                available_resources: Available repair resources (all, limited, emergency_only)
                
            Returns:
                Repair schedule and resource allocation plan
            """
            # Get damaged assets
            damaged_assets = [
                (asset_id, asset) for asset_id, asset in self.infrastructure_assets.items()
                if asset.status != InfrastructureStatus.OPERATIONAL
            ]
            
            if not damaged_assets:
                return {
                    "repair_schedule": [],
                    "message": "No infrastructure repairs required",
                    "total_repair_time": 0
                }
            
            # Prioritization strategies
            if repair_strategy == "critical_first":
                # Prioritize by criticality and status
                damaged_assets.sort(key=lambda x: (
                    4 if x[1].criticality == "critical" else 3 if x[1].criticality == "high" else 2,
                    4 if x[1].status == InfrastructureStatus.OFFLINE else 3 if x[1].status == InfrastructureStatus.CRITICAL else 2
                ), reverse=True)
            
            elif repair_strategy == "cascading_impact":
                # Prioritize by dependency impact
                dependency_scores = {}
                for asset_id, asset in damaged_assets:
                    # Count how many other assets depend on this one
                    dependents = sum(1 for other_id, other_asset in self.infrastructure_assets.items()
                                   if asset_id in other_asset.dependencies)
                    dependency_scores[asset_id] = dependents
                
                damaged_assets.sort(key=lambda x: dependency_scores.get(x[0], 0), reverse=True)
            
            elif repair_strategy == "economic_priority":
                # Prioritize by economic impact (simplified)
                economic_weights = {
                    InfrastructureType.POWER_GRID: 10,
                    InfrastructureType.TRANSPORTATION: 8,
                    InfrastructureType.TELECOMMUNICATIONS: 7,
                    InfrastructureType.WATER_SYSTEM: 9,
                    InfrastructureType.FUEL_DISTRIBUTION: 6
                }
                
                damaged_assets.sort(key=lambda x: economic_weights.get(x[1].asset_type, 5), reverse=True)
            
            # Filter available crews based on resources
            available_crew_list = list(self.available_crews.keys())
            if available_resources == "limited":
                available_crew_list = available_crew_list[:3]  # Only first 3 crews
            elif available_resources == "emergency_only":
                available_crew_list = [crew_id for crew_id, crew in self.available_crews.items()
                                     if crew["specialty"] == "general"]
            
            # Schedule repairs
            repair_schedule = []
            current_time = datetime.utcnow()
            crew_schedules = {crew_id: current_time for crew_id in available_crew_list}
            
            for asset_id, asset in damaged_assets:
                # Find best crew for this asset
                suitable_crews = [
                    crew_id for crew_id in available_crew_list
                    if self._is_crew_suitable(self.available_crews[crew_id], asset.asset_type)
                ]
                
                if not suitable_crews:
                    continue  # No suitable crew available
                
                # Assign to crew with earliest availability
                best_crew = min(suitable_crews, key=lambda c: crew_schedules[c])
                start_time = crew_schedules[best_crew]
                completion_time = start_time + timedelta(hours=asset.estimated_repair_time)
                
                # Create repair operation
                operation = RepairOperation(
                    operation_id=f"repair_{len(repair_schedule)+1}_{asset_id}",
                    asset_id=asset_id,
                    priority=asset.criticality,
                    crew_assigned=best_crew,
                    estimated_completion=completion_time,
                    resources_required=self._get_required_resources(asset.asset_type)
                )
                
                repair_schedule.append({
                    "operation_id": operation.operation_id,
                    "asset_id": asset_id,
                    "asset_type": asset.asset_type.value,
                    "priority": asset.criticality,
                    "crew_assigned": best_crew,
                    "start_time": start_time.isoformat(),
                    "estimated_completion": completion_time.isoformat(),
                    "duration_hours": asset.estimated_repair_time,
                    "resources_required": operation.resources_required
                })
                
                self.active_repairs.append(operation)
                
                # Update crew schedule
                crew_schedules[best_crew] = completion_time
                
                # Mark crew as busy
                self.available_crews[best_crew]["available"] = False
                self.available_crews[best_crew]["current_job"] = operation.operation_id
            
            # Calculate total repair time
            total_repair_time = max([
                (datetime.fromisoformat(repair["estimated_completion"].replace('Z', '+00:00')) - current_time).total_seconds() / 3600
                for repair in repair_schedule
            ]) if repair_schedule else 0
            
            result = {
                "repair_strategy": repair_strategy,
                "total_repairs_scheduled": len(repair_schedule),
                "repair_schedule": repair_schedule,
                "total_estimated_hours": round(total_repair_time, 1),
                "crews_deployed": len(set(repair["crew_assigned"] for repair in repair_schedule)),
                "resource_allocation": self._calculate_resource_needs(repair_schedule),
                "critical_path_assets": [repair["asset_id"] for repair in repair_schedule[:3]]
            }
            
            logger.info(f"Infrastructure repairs prioritized: {len(repair_schedule)} operations scheduled, "
                       f"{total_repair_time:.1f} hours total duration")
            return result
        
        def coordinate_utility_restoration(utility_type: str, restoration_priority: str = "essential_services", coordination_level: str = "full") -> Dict[str, Any]:
            """
            Coordinate restoration of utility services.
            
            Args:
                utility_type: Type of utility to restore (power, water, telecommunications, all)
                restoration_priority: Priority focus (essential_services, residential, commercial, industrial)
                coordination_level: Level of coordination (minimal, standard, full)
                
            Returns:
                Utility restoration plan and coordination status
            """
            utility_mapping = {
                "power": InfrastructureType.POWER_GRID,
                "water": InfrastructureType.WATER_SYSTEM,
                "telecommunications": InfrastructureType.TELECOMMUNICATIONS,
                "sewage": InfrastructureType.SEWAGE_SYSTEM,
                "fuel": InfrastructureType.FUEL_DISTRIBUTION
            }
            
            if utility_type == "all":
                target_utilities = list(utility_mapping.values())
            else:
                target_utilities = [utility_mapping.get(utility_type, InfrastructureType.POWER_GRID)]
            
            restoration_plan = {}
            coordination_actions = []
            
            for utility in target_utilities:
                utility_assets = [
                    (asset_id, asset) for asset_id, asset in self.infrastructure_assets.items()
                    if asset.asset_type == utility
                ]
                
                # Categorize assets by restoration priority
                priority_categories = {
                    "essential_services": [],
                    "residential": [],
                    "commercial": [],
                    "industrial": []
                }
                
                for asset_id, asset in utility_assets:
                    if asset.criticality == "critical" or "hospital" in asset_id or "emergency" in asset_id:
                        priority_categories["essential_services"].append((asset_id, asset))
                    elif "residential" in asset.location or "housing" in asset.location:
                        priority_categories["residential"].append((asset_id, asset))
                    elif "commercial" in asset.location or "business" in asset.location:
                        priority_categories["commercial"].append((asset_id, asset))
                    else:
                        priority_categories["industrial"].append((asset_id, asset))
                
                # Create restoration sequence
                restoration_sequence = priority_categories[restoration_priority]
                if restoration_priority == "essential_services":
                    # Include all critical assets regardless of location
                    restoration_sequence.extend([
                        (aid, asset) for aid, asset in utility_assets
                        if asset.criticality == "critical" and (aid, asset) not in restoration_sequence
                    ])
                
                # Calculate restoration timeline
                restoration_timeline = []
                current_capacity = sum(asset.capacity_percentage for _, asset in restoration_sequence) / max(1, len(restoration_sequence))
                
                for asset_id, asset in restoration_sequence:
                    if asset.status != InfrastructureStatus.OPERATIONAL:
                        restoration_timeline.append({
                            "asset_id": asset_id,
                            "current_capacity": asset.capacity_percentage,
                            "target_capacity": 100.0,
                            "estimated_restoration_hours": asset.estimated_repair_time,
                            "dependencies": asset.dependencies
                        })
                
                restoration_plan[utility.value] = {
                    "current_availability": round(current_capacity, 1),
                    "target_availability": 100.0,
                    "restoration_sequence": restoration_timeline,
                    "estimated_full_restoration_hours": max([item["estimated_restoration_hours"] for item in restoration_timeline], default=0),
                    "coordination_requirements": self._get_coordination_requirements(utility, coordination_level)
                }
                
                # Add coordination actions
                if coordination_level in ["standard", "full"]:
                    coordination_actions.extend([
                        f"Coordinate with {utility.value} utility company",
                        f"Establish communication protocols for {utility.value} restoration",
                        f"Deploy liaison personnel to {utility.value} control center"
                    ])
                
                if coordination_level == "full":
                    coordination_actions.extend([
                        f"Activate mutual aid agreements for {utility.value}",
                        f"Coordinate with regional {utility.value} authorities",
                        f"Implement public communication plan for {utility.value} restoration"
                    ])
            
            # Calculate overall restoration metrics
            total_assets_affected = sum(len(plan["restoration_sequence"]) for plan in restoration_plan.values())
            average_restoration_time = sum(plan["estimated_full_restoration_hours"] for plan in restoration_plan.values()) / max(1, len(restoration_plan))
            
            result = {
                "utility_type": utility_type,
                "restoration_priority": restoration_priority,
                "coordination_level": coordination_level,
                "restoration_plan": restoration_plan,
                "coordination_actions": list(set(coordination_actions)),  # Remove duplicates
                "overall_metrics": {
                    "total_assets_in_plan": total_assets_affected,
                    "average_restoration_time_hours": round(average_restoration_time, 1),
                    "coordination_complexity": "high" if coordination_level == "full" else "medium" if coordination_level == "standard" else "low",
                    "success_probability": self._calculate_restoration_success_probability(restoration_plan, coordination_level)
                }
            }
            
            logger.info(f"Utility restoration coordinated for {utility_type}: {total_assets_affected} assets in plan, "
                       f"{average_restoration_time:.1f} hours average restoration time")
            return result
        
        def monitor_infrastructure_recovery(monitoring_period_hours: int = 24) -> Dict[str, Any]:
            """
            Monitor infrastructure recovery progress and system health.
            
            Args:
                monitoring_period_hours: Duration of monitoring period
                
            Returns:
                Infrastructure recovery status and performance metrics
            """
            monitoring_start = datetime.utcnow()
            monitoring_end = monitoring_start + timedelta(hours=monitoring_period_hours)
            
            # Simulate recovery progress for active repairs
            recovery_progress = {}
            completed_repairs = []
            ongoing_repairs = []
            
            for repair in self.active_repairs:
                time_elapsed = (monitoring_start - repair.estimated_completion + timedelta(hours=self.infrastructure_assets[repair.asset_id].estimated_repair_time)).total_seconds() / 3600
                
                if time_elapsed >= self.infrastructure_assets[repair.asset_id].estimated_repair_time:
                    # Repair completed
                    asset = self.infrastructure_assets[repair.asset_id]
                    asset.status = InfrastructureStatus.OPERATIONAL
                    asset.capacity_percentage = 100.0
                    asset.estimated_repair_time = 0
                    
                    completed_repairs.append({
                        "operation_id": repair.operation_id,
                        "asset_id": repair.asset_id,
                        "completion_status": "completed",
                        "actual_duration_hours": round(time_elapsed, 1)
                    })
                    
                    # Free up crew
                    self.available_crews[repair.crew_assigned]["available"] = True
                    self.available_crews[repair.crew_assigned]["current_job"] = None
                    
                else:
                    # Repair ongoing - calculate progress
                    progress_percentage = (time_elapsed / self.infrastructure_assets[repair.asset_id].estimated_repair_time) * 100
                    
                    ongoing_repairs.append({
                        "operation_id": repair.operation_id,
                        "asset_id": repair.asset_id,
                        "progress_percentage": round(min(100, progress_percentage), 1),
                        "estimated_completion": repair.estimated_completion.isoformat(),
                        "crew_assigned": repair.crew_assigned
                    })
                    
                    # Update asset capacity based on progress
                    if progress_percentage > 50:  # Partial restoration after 50% completion
                        asset = self.infrastructure_assets[repair.asset_id]
                        asset.capacity_percentage = min(100, 20 + (progress_percentage * 0.8))
                        if asset.capacity_percentage > 80:
                            asset.status = InfrastructureStatus.DEGRADED
            
            # Remove completed repairs from active list
            self.active_repairs = [repair for repair in self.active_repairs 
                                 if repair.operation_id not in [cr["operation_id"] for cr in completed_repairs]]
            
            # Update system availability metrics
            self._update_system_availability()
            
            # Calculate recovery metrics
            total_assets = len(self.infrastructure_assets)
            operational_assets = len([asset for asset in self.infrastructure_assets.values() 
                                    if asset.status == InfrastructureStatus.OPERATIONAL])
            
            recovery_rate = (operational_assets / total_assets) * 100
            
            # Identify bottlenecks
            bottlenecks = []
            for asset_id, asset in self.infrastructure_assets.items():
                if asset.status in [InfrastructureStatus.OFFLINE, InfrastructureStatus.CRITICAL]:
                    # Check if this asset is blocking others
                    blocked_assets = [
                        other_id for other_id, other_asset in self.infrastructure_assets.items()
                        if asset_id in other_asset.dependencies and other_asset.status != InfrastructureStatus.OPERATIONAL
                    ]
                    
                    if blocked_assets:
                        bottlenecks.append({
                            "bottleneck_asset": asset_id,
                            "blocked_assets": blocked_assets,
                            "impact_level": "high" if asset.criticality == "critical" else "medium"
                        })
            
            result = {
                "monitoring_period_hours": monitoring_period_hours,
                "monitoring_start": monitoring_start.isoformat(),
                "monitoring_end": monitoring_end.isoformat(),
                "recovery_progress": {
                    "completed_repairs": completed_repairs,
                    "ongoing_repairs": ongoing_repairs,
                    "repairs_remaining": len(self.active_repairs)
                },
                "system_health": {
                    "overall_recovery_percentage": round(recovery_rate, 1),
                    "operational_assets": operational_assets,
                    "total_assets": total_assets,
                    "overall_infrastructure_health": round(self.overall_infrastructure_health, 1)
                },
                "system_availability": {
                    "power_grid": round(self.power_grid_availability, 1),
                    "communications": round(self.communications_availability, 1),
                    "transportation": round(self.transportation_availability, 1),
                    "utilities": round(self.utilities_availability, 1)
                },
                "bottlenecks": bottlenecks,
                "crew_utilization": {
                    crew_id: "busy" if not crew["available"] else "available"
                    for crew_id, crew in self.available_crews.items()
                },
                "next_monitoring_cycle": monitoring_end.isoformat(),
                "recommendations": self._generate_recovery_recommendations(recovery_rate, bottlenecks, ongoing_repairs)
            }
            
            logger.info(f"Infrastructure recovery monitored: {recovery_rate:.1f}% recovery rate, "
                       f"{len(completed_repairs)} repairs completed, {len(bottlenecks)} bottlenecks identified")
            return result
        
        # Create infrastructure management ADK agent
        return ERISAgentFactory.create_eris_agent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            model="gemini-2.0-flash",
            custom_tools=[assess_infrastructure_damage, prioritize_infrastructure_repairs, 
                         coordinate_utility_restoration, monitor_infrastructure_recovery]
        )
    
    def _update_system_availability(self):
        """Update system availability metrics based on current asset status"""
        
        system_types = {
            InfrastructureType.POWER_GRID: [],
            InfrastructureType.TELECOMMUNICATIONS: [],
            InfrastructureType.TRANSPORTATION: [],
            InfrastructureType.WATER_SYSTEM: [],
            InfrastructureType.SEWAGE_SYSTEM: []
        }
        
        # Group assets by type
        for asset in self.infrastructure_assets.values():
            if asset.asset_type in system_types:
                system_types[asset.asset_type].append(asset)
        
        # Calculate availability for each system
        self.power_grid_availability = self._calculate_system_availability(system_types[InfrastructureType.POWER_GRID])
        self.communications_availability = self._calculate_system_availability(system_types[InfrastructureType.TELECOMMUNICATIONS])
        self.transportation_availability = self._calculate_system_availability(system_types[InfrastructureType.TRANSPORTATION])
        
        # Utilities combines water and sewage
        utility_assets = system_types[InfrastructureType.WATER_SYSTEM] + system_types[InfrastructureType.SEWAGE_SYSTEM]
        self.utilities_availability = self._calculate_system_availability(utility_assets)
        
        # Overall infrastructure health
        self.overall_infrastructure_health = (
            self.power_grid_availability * 0.3 +
            self.communications_availability * 0.2 +
            self.transportation_availability * 0.25 +
            self.utilities_availability * 0.25
        )
    
    def _calculate_system_availability(self, assets: List[InfrastructureAsset]) -> float:
        """Calculate availability percentage for a system type"""
        if not assets:
            return 100.0
        
        # Weight by criticality
        total_weight = 0
        weighted_capacity = 0
        
        for asset in assets:
            weight = 4 if asset.criticality == "critical" else 3 if asset.criticality == "high" else 2 if asset.criticality == "medium" else 1
            total_weight += weight
            weighted_capacity += asset.capacity_percentage * weight
        
        return weighted_capacity / total_weight if total_weight > 0 else 100.0
    
    def _is_crew_suitable(self, crew: Dict[str, Any], asset_type: InfrastructureType) -> bool:
        """Check if crew is suitable for asset type"""
        specialty_mapping = {
            InfrastructureType.POWER_GRID: ["electrical", "general"],
            InfrastructureType.TELECOMMUNICATIONS: ["telecommunications", "electrical", "general"],
            InfrastructureType.TRANSPORTATION: ["civil_engineering", "general"],
            InfrastructureType.WATER_SYSTEM: ["civil_engineering", "general"],
            InfrastructureType.SEWAGE_SYSTEM: ["civil_engineering", "general"],
            InfrastructureType.FUEL_DISTRIBUTION: ["general"],
            InfrastructureType.HEALTHCARE_FACILITIES: ["electrical", "general"]
        }
        
        suitable_specialties = specialty_mapping.get(asset_type, ["general"])
        return crew["specialty"] in suitable_specialties
    
    def _get_required_resources(self, asset_type: InfrastructureType) -> List[str]:
        """Get required resources for asset repair"""
        resource_mapping = {
            InfrastructureType.POWER_GRID: ["electrical_equipment", "generators", "cables", "transformers"],
            InfrastructureType.TELECOMMUNICATIONS: ["communication_equipment", "fiber_cables", "radio_equipment"],
            InfrastructureType.TRANSPORTATION: ["construction_materials", "heavy_machinery", "asphalt", "concrete"],
            InfrastructureType.WATER_SYSTEM: ["pipes", "pumps", "treatment_chemicals", "filtration_equipment"],
            InfrastructureType.SEWAGE_SYSTEM: ["pipes", "pumps", "treatment_equipment"],
            InfrastructureType.FUEL_DISTRIBUTION: ["storage_tanks", "pumps", "safety_equipment"],
            InfrastructureType.HEALTHCARE_FACILITIES: ["medical_equipment", "backup_power", "HVAC_systems"]
        }
        
        return resource_mapping.get(asset_type, ["general_supplies"])
    
    def _calculate_resource_needs(self, repair_schedule: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate total resource needs for repair schedule"""
        resource_totals = {}
        
        for repair in repair_schedule:
            for resource in repair["resources_required"]:
                resource_totals[resource] = resource_totals.get(resource, 0) + 1
        
        return resource_totals
    
    def _get_coordination_requirements(self, utility: InfrastructureType, coordination_level: str) -> List[str]:
        """Get coordination requirements for utility restoration"""
        base_requirements = [
            f"Contact {utility.value} utility company",
            f"Establish repair timeline for {utility.value}",
            f"Coordinate resource allocation for {utility.value}"
        ]
        
        if coordination_level in ["standard", "full"]:
            base_requirements.extend([
                f"Deploy liaison to {utility.value} control center",
                f"Implement communication protocols",
                f"Coordinate with emergency services"
            ])
        
        if coordination_level == "full":
            base_requirements.extend([
                f"Activate mutual aid for {utility.value}",
                f"Coordinate with state/regional authorities",
                f"Implement public information campaign"
            ])
        
        return base_requirements
    
    def _calculate_restoration_success_probability(self, restoration_plan: Dict[str, Any], coordination_level: str) -> float:
        """Calculate probability of successful restoration"""
        base_probability = 0.7  # 70% base success rate
        
        # Coordination level bonus
        coordination_bonus = {
            "minimal": 0.0,
            "standard": 0.1,
            "full": 0.2
        }.get(coordination_level, 0.1)
        
        # Complexity penalty
        total_assets = sum(len(plan["restoration_sequence"]) for plan in restoration_plan.values())
        complexity_penalty = min(0.2, total_assets * 0.01)  # Max 20% penalty
        
        success_probability = base_probability + coordination_bonus - complexity_penalty
        return round(min(0.95, max(0.4, success_probability)), 2)  # Clamp between 40% and 95%
    
    def _generate_recovery_recommendations(self, recovery_rate: float, bottlenecks: List[Dict], ongoing_repairs: List[Dict]) -> List[str]:
        """Generate recommendations based on recovery status"""
        recommendations = []
        
        if recovery_rate < 50:
            recommendations.append("Consider requesting additional repair crews and resources")
            recommendations.append("Activate mutual aid agreements with neighboring jurisdictions")
        
        if recovery_rate < 75:
            recommendations.append("Prioritize critical infrastructure repairs")
            recommendations.append("Implement rolling restoration schedule")
        
        if bottlenecks:
            recommendations.append("Focus resources on bottleneck assets to unblock dependent systems")
            recommendations.append("Consider temporary workarounds for blocked dependencies")
        
        if len(ongoing_repairs) > 5:
            recommendations.append("Monitor crew fatigue and consider shift rotations")
            recommendations.append("Ensure adequate supply chain for repair materials")
        
        if recovery_rate > 85:
            recommendations.append("Begin transition from emergency to normal operations")
            recommendations.append("Conduct lessons learned assessment for future improvements")
        
        return recommendations
    
    async def initialize_for_simulation(self, simulation_id: str, disaster_type: str, severity: int, location: str):
        """Initialize agent for a specific simulation"""
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.disaster_severity = severity
        
        # Reset all infrastructure to operational status
        for asset in self.infrastructure_assets.values():
            asset.status = InfrastructureStatus.OPERATIONAL
            asset.capacity_percentage = 100.0
            asset.estimated_repair_time = 0
        
        # Reset crews to available
        for crew in self.available_crews.values():
            crew["available"] = True
            crew["current_job"] = None
        
        # Reset metrics
        self.overall_infrastructure_health = 100.0
        self.power_grid_availability = 100.0
        self.communications_availability = 100.0
        self.transportation_availability = 100.0
        self.utilities_availability = 100.0
        self.active_repairs = []
        
        logger.info(f"Infrastructure Manager Agent initialized for {disaster_type} severity {severity}")
    
    async def process_phase(self, phase: SimulationPhase, simulation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process infrastructure management for a specific simulation phase"""
        self.current_phase = phase
        
        # Execute phase-specific infrastructure logic
        phase_results = await self._process_phase_specific_logic(phase, simulation_context)
        
        # Generate comprehensive metrics
        metrics = await self._generate_infrastructure_metrics()
        
        # Save state to cloud services
        await self._save_infrastructure_state(metrics)
        
        # Log to BigQuery for analytics
        await self._log_infrastructure_event(phase, metrics)
        
        return {
            "agent_id": self.agent_id,
            "phase": phase.value,
            "infrastructure_metrics": metrics,
            "phase_actions": phase_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _process_phase_specific_logic(self, phase: SimulationPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute phase-specific infrastructure logic"""
        
        if phase == SimulationPhase.IMPACT:
            return await self._process_impact_phase(context)
        elif phase == SimulationPhase.RESPONSE:
            return await self._process_response_phase(context)
        elif phase == SimulationPhase.RECOVERY:
            return await self._process_recovery_phase(context)
        
        return {}
    
    async def _process_impact_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process immediate infrastructure impact assessment"""
        
        # Determine affected areas based on disaster type and context
        affected_areas = ["central_district", "coastal_area", "northern_zone"]
        if self.disaster_type in ['tsunami', 'flood']:
            affected_areas.extend(["port_district", "southern_zone"])
        elif self.disaster_type in ['earthquake']:
            affected_areas.extend(["utility_district", "airport_zone"])
        
        try:
            # Assess infrastructure damage using internal implementation
            damage_patterns = {
                'earthquake': {
                    InfrastructureType.POWER_GRID: 0.7,
                    InfrastructureType.TELECOMMUNICATIONS: 0.5,
                    InfrastructureType.TRANSPORTATION: 0.8,
                    InfrastructureType.WATER_SYSTEM: 0.6
                },
                'tsunami': {
                    InfrastructureType.POWER_GRID: 0.9,
                    InfrastructureType.TELECOMMUNICATIONS: 0.8,
                    InfrastructureType.TRANSPORTATION: 0.9,
                    InfrastructureType.WATER_SYSTEM: 0.8
                },
                'flood': {
                    InfrastructureType.POWER_GRID: 0.6,
                    InfrastructureType.TELECOMMUNICATIONS: 0.4,
                    InfrastructureType.TRANSPORTATION: 0.7,
                    InfrastructureType.SEWAGE_SYSTEM: 0.9
                }
            }.get(self.disaster_type, {
                InfrastructureType.POWER_GRID: 0.5,
                InfrastructureType.TRANSPORTATION: 0.6
            })
            
            severity_factor = self.disaster_severity / 10.0
            critical_failures = []
            damaged_assets = {}
            
            for asset_id, asset in self.infrastructure_assets.items():
                # Check if asset is in affected area
                asset_affected = any(area in asset.location for area in affected_areas)
                
                if asset_affected:
                    damage_probability = damage_patterns.get(asset.asset_type, 0.3) * severity_factor
                    
                    if asset.criticality == "critical":
                        damage_probability *= 1.2
                    
                    # Apply damage
                    if damage_probability > 0.8:
                        asset.status = InfrastructureStatus.OFFLINE
                        asset.capacity_percentage = 0.0
                        asset.estimated_repair_time = 24 + (self.disaster_severity * 2)
                        critical_failures.append(asset_id)
                    elif damage_probability > 0.6:
                        asset.status = InfrastructureStatus.CRITICAL
                        asset.capacity_percentage = 25.0
                        asset.estimated_repair_time = 12 + self.disaster_severity
                    elif damage_probability > 0.4:
                        asset.status = InfrastructureStatus.DEGRADED
                        asset.capacity_percentage = 60.0
                        asset.estimated_repair_time = 6 + (self.disaster_severity // 2)
                    
                    if asset.status != InfrastructureStatus.OPERATIONAL:
                        damaged_assets[asset_id] = {
                            "asset_type": asset.asset_type.value,
                            "status": asset.status.value,
                            "capacity_percentage": asset.capacity_percentage
                        }
            
            self._update_system_availability()
            
            damage_assessment = {
                "total_assets_assessed": len(self.infrastructure_assets),
                "damaged_assets": len(damaged_assets),
                "critical_failures": critical_failures,
                "overall_infrastructure_health": round(self.overall_infrastructure_health, 1)
            }
            
            # Initial repair prioritization for critical assets
            critical_repairs = []
            for asset_id in critical_failures[:3]:  # Top 3 critical failures
                asset = self.infrastructure_assets[asset_id]
                critical_repairs.append({
                    "asset_id": asset_id,
                    "asset_type": asset.asset_type.value,
                    "estimated_repair_hours": asset.estimated_repair_time,
                    "priority": "critical"
                })
            
            repair_prioritization = {
                "repair_strategy": "critical_first",
                "critical_repairs_identified": len(critical_repairs),
                "immediate_repairs": critical_repairs
            }
            
        except Exception as e:
            logger.warning(f"Infrastructure impact phase error: {e}")
            damage_assessment = {"damaged_assets": 5, "critical_failures": ["power_plant_main"]}
            repair_prioritization = {"critical_repairs_identified": 2}
        
        return {
            "damage_assessment": damage_assessment,
            "initial_repair_prioritization": repair_prioritization,
            "phase_focus": "infrastructure_damage_assessment"
        }
    
    async def _process_response_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process coordinated infrastructure response"""
        
        try:
            # Schedule critical repairs
            damaged_assets = [
                (asset_id, asset) for asset_id, asset in self.infrastructure_assets.items()
                if asset.status != InfrastructureStatus.OPERATIONAL
            ]
            
            # Sort by criticality
            damaged_assets.sort(key=lambda x: (
                4 if x[1].criticality == "critical" else 3,
                4 if x[1].status == InfrastructureStatus.OFFLINE else 3
            ), reverse=True)
            
            repair_schedule = []
            current_time = datetime.utcnow()
            available_crews = [crew_id for crew_id, crew in self.available_crews.items() if crew["available"]]
            
            for i, (asset_id, asset) in enumerate(damaged_assets[:len(available_crews)]):
                if i < len(available_crews):
                    crew_id = available_crews[i]
                    completion_time = current_time + timedelta(hours=asset.estimated_repair_time)
                    
                    repair_schedule.append({
                        "asset_id": asset_id,
                        "crew_assigned": crew_id,
                        "estimated_completion": completion_time.isoformat(),
                        "duration_hours": asset.estimated_repair_time
                    })
                    
                    self.available_crews[crew_id]["available"] = False
            
            repair_scheduling = {
                "total_repairs_scheduled": len(repair_schedule),
                "repair_schedule": repair_schedule,
                "crews_deployed": len(repair_schedule)
            }
            
            # Coordinate utility restoration for power (most critical)
            power_assets = [
                asset for asset in self.infrastructure_assets.values()
                if asset.asset_type == InfrastructureType.POWER_GRID
            ]
            
            power_availability = sum(asset.capacity_percentage for asset in power_assets) / max(1, len(power_assets))
            
            utility_coordination = {
                "utility_type": "power",
                "current_availability": round(power_availability, 1),
                "coordination_level": "full",
                "restoration_priority": "essential_services"
            }
            
        except Exception as e:
            logger.warning(f"Infrastructure response phase error: {e}")
            repair_scheduling = {"total_repairs_scheduled": 3}
            utility_coordination = {"utility_type": "power", "current_availability": 65}
        
        return {
            "repair_scheduling": repair_scheduling,
            "utility_coordination": utility_coordination,
            "phase_focus": "coordinated_infrastructure_repair"
        }
    
    async def _process_recovery_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process infrastructure recovery monitoring"""
        
        try:
            # Simulate recovery progress
            completed_repairs = []
            ongoing_repairs = []
            
            # Check repair progress
            for repair in self.active_repairs:
                # Simulate completion of some repairs
                if repair.priority == "critical":
                    # Critical repairs complete faster
                    completed_repairs.append({
                        "operation_id": repair.operation_id,
                        "asset_id": repair.asset_id,
                        "completion_status": "completed"
                    })
                    
                    # Restore asset
                    asset = self.infrastructure_assets[repair.asset_id]
                    asset.status = InfrastructureStatus.OPERATIONAL
                    asset.capacity_percentage = 100.0
                    asset.estimated_repair_time = 0
                    
                    # Free crew
                    self.available_crews[repair.crew_assigned]["available"] = True
                else:
                    ongoing_repairs.append({
                        "operation_id": repair.operation_id,
                        "asset_id": repair.asset_id,
                        "progress_percentage": 75
                    })
            
            # Update active repairs
            self.active_repairs = [r for r in self.active_repairs if r.priority != "critical"]
            
            # Update system availability
            self._update_system_availability()
            
            # Calculate recovery metrics
            total_assets = len(self.infrastructure_assets)
            operational_assets = len([asset for asset in self.infrastructure_assets.values() 
                                    if asset.status == InfrastructureStatus.OPERATIONAL])
            recovery_rate = (operational_assets / total_assets) * 100
            
            recovery_monitoring = {
                "recovery_progress": {
                    "completed_repairs": completed_repairs,
                    "ongoing_repairs": ongoing_repairs,
                    "repairs_remaining": len(self.active_repairs)
                },
                "system_health": {
                    "overall_recovery_percentage": round(recovery_rate, 1),
                    "operational_assets": operational_assets,
                    "total_assets": total_assets
                },
                "system_availability": {
                    "power_grid": round(self.power_grid_availability, 1),
                    "communications": round(self.communications_availability, 1),
                    "transportation": round(self.transportation_availability, 1),
                    "utilities": round(self.utilities_availability, 1)
                }
            }
            
        except Exception as e:
            logger.warning(f"Infrastructure recovery phase error: {e}")
            recovery_monitoring = {
                "system_health": {"overall_recovery_percentage": 85},
                "system_availability": {"power_grid": 90, "communications": 85}
            }
        
        return {
            "recovery_monitoring": recovery_monitoring,
            "phase_focus": "infrastructure_recovery_assessment"
        }
    
    async def _generate_infrastructure_metrics(self) -> Dict[str, Any]:
        """Generate comprehensive infrastructure metrics"""
        
        # Asset status breakdown
        status_counts = {status.value: 0 for status in InfrastructureStatus}
        criticality_breakdown = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        type_breakdown = {inf_type.value: {"operational": 0, "total": 0} for inf_type in InfrastructureType}
        
        for asset in self.infrastructure_assets.values():
            status_counts[asset.status.value] += 1
            criticality_breakdown[asset.criticality] += 1
            type_breakdown[asset.asset_type.value]["total"] += 1
            if asset.status == InfrastructureStatus.OPERATIONAL:
                type_breakdown[asset.asset_type.value]["operational"] += 1
        
        # Calculate type availability percentages
        for type_data in type_breakdown.values():
            type_data["availability_percentage"] = round((type_data["operational"] / max(1, type_data["total"])) * 100, 1)
        
        return {
            "system_metrics": {
                "overall_infrastructure_health": round(self.overall_infrastructure_health, 1),
                "power_grid_availability": round(self.power_grid_availability, 1),
                "communications_availability": round(self.communications_availability, 1),
                "transportation_availability": round(self.transportation_availability, 1),
                "utilities_availability": round(self.utilities_availability, 1)
            },
            "asset_metrics": {
                "total_assets": len(self.infrastructure_assets),
                "status_breakdown": status_counts,
                "criticality_breakdown": criticality_breakdown,
                "type_breakdown": type_breakdown
            },
            "repair_metrics": {
                "active_repairs": len(self.active_repairs),
                "available_crews": len([crew for crew in self.available_crews.values() if crew["available"]]),
                "total_crews": len(self.available_crews),
                "crew_utilization_percentage": round(((len(self.available_crews) - len([crew for crew in self.available_crews.values() if crew["available"]])) / len(self.available_crews)) * 100, 1)
            },
            "operational_metrics": {
                "system_resilience": "high" if self.overall_infrastructure_health > 80 else "medium" if self.overall_infrastructure_health > 60 else "low",
                "critical_systems_operational": status_counts[InfrastructureStatus.OPERATIONAL.value] >= criticality_breakdown["critical"],
                "recovery_phase": self.current_phase.value
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _save_infrastructure_state(self, metrics: Dict[str, Any]):
        """Save infrastructure state to Firestore"""
        try:
            state_data = {
                "agent_id": self.agent_id,
                "simulation_id": self.simulation_id,
                "infrastructure_assets": {
                    asset_id: {
                        "asset_type": asset.asset_type.value,
                        "location": asset.location,
                        "criticality": asset.criticality,
                        "status": asset.status.value,
                        "capacity_percentage": asset.capacity_percentage,
                        "estimated_repair_time": asset.estimated_repair_time
                    }
                    for asset_id, asset in self.infrastructure_assets.items()
                },
                "active_repairs": [
                    {
                        "operation_id": repair.operation_id,
                        "asset_id": repair.asset_id,
                        "priority": repair.priority,
                        "crew_assigned": repair.crew_assigned
                    }
                    for repair in self.active_repairs
                ],
                "crew_status": self.available_crews.copy(),
                "metrics": metrics,
                "phase": self.current_phase.value,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.cloud.firestore.save_agent_state(self.agent_id, self.simulation_id, state_data)
            
        except Exception as e:
            logger.error(f"Failed to save infrastructure state: {e}")
    
    async def _log_infrastructure_event(self, phase: SimulationPhase, metrics: Dict[str, Any]):
        """Log infrastructure events to BigQuery"""
        try:
            event_data = {
                "event_type": "infrastructure_update",
                "agent_id": self.agent_id,
                "phase": phase.value,
                "overall_health": metrics["system_metrics"]["overall_infrastructure_health"],
                "power_availability": metrics["system_metrics"]["power_grid_availability"],
                "communications_availability": metrics["system_metrics"]["communications_availability"],
                "transportation_availability": metrics["system_metrics"]["transportation_availability"],
                "active_repairs": metrics["repair_metrics"]["active_repairs"],
                "crew_utilization": metrics["repair_metrics"]["crew_utilization_percentage"]
            }
            
            await self.cloud.bigquery.log_simulation_event(
                simulation_id=self.simulation_id,
                event_type="infrastructure_update",
                event_data=event_data,
                agent_id=self.agent_id,
                phase=phase.value
            )
            
        except Exception as e:
            logger.error(f"Failed to log infrastructure event: {e}")


def create_infrastructure_manager_agent(cloud_services: CloudServices) -> InfrastructureManagerAgent:
    """Factory function to create an Infrastructure Manager Agent"""
    return InfrastructureManagerAgent(cloud_services)
