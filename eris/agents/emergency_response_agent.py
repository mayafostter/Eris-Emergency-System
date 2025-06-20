"""
ERIS Emergency Response Agent - Central command and control coordination
Manages resource allocation, priority setting, and cross-department coordination
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

class ResponsePriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class EmergencyResource:
    resource_type: str
    total_available: int
    currently_deployed: int
    response_time_minutes: float
    coverage_area: str

@dataclass
class IncidentReport:
    incident_id: str
    location: str
    severity: int
    resource_needs: List[str]
    priority: ResponsePriority
    status: str

class EmergencyResponseAgent:
    """
    Emergency Response Coordinator Agent for ERIS disaster simulation.
    Central command for coordinating first responders and emergency resources.
    """
    
    def __init__(self, cloud_services: CloudServices):
        self.cloud = cloud_services
        self.agent_id = "emergency_response_coordinator"
        self.agent_type = "emergency_response"
        
        # Emergency resources inventory
        self.resources = {
            "fire_rescue": EmergencyResource("fire_rescue", 25, 0, 8.5, "city_wide"),
            "police_units": EmergencyResource("police_units", 40, 0, 5.2, "city_wide"),
            "ambulances": EmergencyResource("ambulances", 18, 0, 7.1, "city_wide"),
            "search_rescue": EmergencyResource("search_rescue", 12, 0, 15.3, "regional"),
            "hazmat_teams": EmergencyResource("hazmat_teams", 6, 0, 22.0, "regional"),
            "helicopter_rescue": EmergencyResource("helicopter_rescue", 3, 0, 12.8, "regional"),
            "mobile_command": EmergencyResource("mobile_command", 2, 0, 18.5, "disaster_zone")
        }
        
        # Active incidents and coordination state
        self.active_incidents = []
        self.resource_deployments = {}
        self.coordination_log = []
        
        # Simulation state
        self.simulation_id = None
        self.current_phase = SimulationPhase.IMPACT
        self.disaster_type = None
        self.disaster_severity = 5
        self.response_effectiveness = 0.0
        
        # Performance metrics
        self.response_time_avg = 0.0
        self.resource_utilization = 0.0
        self.incidents_resolved = 0
        self.coordination_requests = 0
        
        # Create the ADK agent
        self.adk_agent = self._create_emergency_agent()
        
    def _create_emergency_agent(self) -> LlmAgent:
        """Create the ADK agent for emergency response coordination"""
        
        def deploy_emergency_resources(incident_location: str, resource_types: str, priority_level: str, estimated_duration: int = 120) -> Dict[str, Any]:
            """
            Deploy emergency resources to incident location.
            
            Args:
                incident_location: Location requiring emergency response
                resource_types: Comma-separated list of resource types needed
                priority_level: Priority (critical, high, medium, low)
                estimated_duration: Expected deployment duration in minutes
                
            Returns:
                Deployment status and resource allocation
            """
            requested_resources = [r.strip() for r in resource_types.split(',')]
            deployment_result = {
                "incident_location": incident_location,
                "deployment_id": f"deploy_{len(self.resource_deployments)}_{datetime.utcnow().strftime('%H%M%S')}",
                "deployed_resources": {},
                "unavailable_resources": [],
                "estimated_response_time": 0.0,
                "deployment_status": "partial"
            }
            
            total_response_time = 0.0
            resources_deployed = 0
            
            for resource_type in requested_resources:
                if resource_type in self.resources:
                    resource = self.resources[resource_type]
                    available = resource.total_available - resource.currently_deployed
                    
                    if available > 0:
                        # Deploy based on priority
                        units_to_deploy = 1
                        if priority_level == "critical":
                            units_to_deploy = min(available, 3)
                        elif priority_level == "high":
                            units_to_deploy = min(available, 2)
                        
                        # Update resource allocation
                        resource.currently_deployed += units_to_deploy
                        
                        deployment_result["deployed_resources"][resource_type] = {
                            "units_deployed": units_to_deploy,
                            "response_time_minutes": resource.response_time_minutes,
                            "coverage_area": resource.coverage_area
                        }
                        
                        total_response_time += resource.response_time_minutes
                        resources_deployed += 1
                        
                        # Log deployment
                        self.resource_deployments[deployment_result["deployment_id"]] = {
                            "resource_type": resource_type,
                            "units": units_to_deploy,
                            "location": incident_location,
                            "start_time": datetime.utcnow(),
                            "estimated_end": datetime.utcnow() + timedelta(minutes=estimated_duration)
                        }
                    else:
                        deployment_result["unavailable_resources"].append(resource_type)
            
            # Calculate metrics
            if resources_deployed > 0:
                deployment_result["estimated_response_time"] = round(total_response_time / resources_deployed, 1)
                deployment_result["deployment_status"] = "success" if not deployment_result["unavailable_resources"] else "partial"
            else:
                deployment_result["deployment_status"] = "failed"
                deployment_result["estimated_response_time"] = 0.0
            
            # Update coordination log
            self.coordination_log.append({
                "action": "resource_deployment",
                "location": incident_location,
                "priority": priority_level,
                "timestamp": datetime.utcnow().isoformat(),
                "result": deployment_result["deployment_status"]
            })
            
            logger.info(f"Emergency resources deployed to {incident_location}: {len(deployment_result['deployed_resources'])} types")
            return deployment_result
        
        def coordinate_multi_agency_response(agencies: str, coordination_type: str, incident_details: str = "{}") -> Dict[str, Any]:
            """
            Coordinate response across multiple agencies and departments.
            
            Args:
                agencies: Comma-separated list of agencies to coordinate
                coordination_type: Type of coordination (command, communication, resource_sharing)
                incident_details: JSON string with incident information
                
            Returns:
                Coordination status and communication plan
            """
            agency_list = [a.strip() for a in agencies.split(',')]
            self.coordination_requests += 1
            
            # Parse incident details
            try:
                details = json.loads(incident_details) if incident_details != "{}" else {}
            except:
                details = {}
            
            coordination_result = {
                "coordination_id": f"coord_{self.coordination_requests}_{datetime.utcnow().strftime('%H%M%S')}",
                "agencies_involved": agency_list,
                "coordination_type": coordination_type,
                "status": "initiated",
                "communication_plan": {},
                "resource_sharing": {},
                "command_structure": {}
            }
            
            # Establish communication plan
            for agency in agency_list:
                coordination_result["communication_plan"][agency] = {
                    "contact_established": True,
                    "communication_channel": f"channel_{agency.lower()}",
                    "update_frequency": "15_minutes" if coordination_type == "command" else "30_minutes"
                }
            
            # Set up command structure
            if coordination_type == "command":
                coordination_result["command_structure"] = {
                    "unified_command": True,
                    "lead_agency": "emergency_response",
                    "supporting_agencies": [a for a in agency_list if a != "emergency_response"],
                    "command_post_location": details.get("command_post", "primary_eoc")
                }
            
            # Resource sharing agreements
            if coordination_type in ["resource_sharing", "command"]:
                coordination_result["resource_sharing"] = {
                    "mutual_aid_activated": True,
                    "shared_resources": ["communications", "transportation", "personnel"],
                    "cost_sharing_agreement": "standard_mutual_aid"
                }
            
            # Update coordination effectiveness
            coordination_effectiveness = min(1.0, 0.7 + (len(agency_list) * 0.05))
            self.response_effectiveness = (self.response_effectiveness + coordination_effectiveness) / 2
            
            # Log coordination action
            self.coordination_log.append({
                "action": "multi_agency_coordination",
                "agencies": agency_list,
                "type": coordination_type,
                "timestamp": datetime.utcnow().isoformat(),
                "effectiveness": coordination_effectiveness
            })
            
            logger.info(f"Multi-agency coordination initiated: {len(agency_list)} agencies, type: {coordination_type}")
            return coordination_result
        
        def assess_response_effectiveness(time_period_hours: float = 1.0) -> Dict[str, Any]:
            """
            Assess overall emergency response effectiveness.
            
            Args:
                time_period_hours: Time period to assess (hours)
                
            Returns:
                Response effectiveness metrics and recommendations
            """
            current_time = datetime.utcnow()
            assessment_start = current_time - timedelta(hours=time_period_hours)
            
            # Filter recent activities
            recent_deployments = [
                d for d in self.resource_deployments.values()
                if d["start_time"] >= assessment_start
            ]
            
            recent_coordination = [
                c for c in self.coordination_log
                if datetime.fromisoformat(c["timestamp"].replace('Z', '+00:00')) >= assessment_start
            ]
            
            # Calculate resource utilization
            total_resources = sum(r.total_available for r in self.resources.values())
            deployed_resources = sum(r.currently_deployed for r in self.resources.values())
            self.resource_utilization = (deployed_resources / total_resources) * 100 if total_resources > 0 else 0
            
            # Calculate average response time
            if recent_deployments:
                response_times = []
                for deployment in recent_deployments:
                    resource_type = deployment["resource_type"]
                    if resource_type in self.resources:
                        response_times.append(self.resources[resource_type].response_time_minutes)
                
                self.response_time_avg = sum(response_times) / len(response_times) if response_times else 0
            
            # Assess coordination effectiveness
            coordination_success_rate = 0.0
            if recent_coordination:
                successful_coords = len([c for c in recent_coordination if c.get("result") in ["success", "partial"]])
                coordination_success_rate = (successful_coords / len(recent_coordination)) * 100
            
            # Overall effectiveness score
            effectiveness_factors = [
                min(100, self.resource_utilization * 1.2),  # Resource utilization (capped)
                max(0, 100 - (self.response_time_avg / 20) * 100),  # Response time (inverse)
                coordination_success_rate,  # Coordination success
                self.response_effectiveness * 100  # Overall effectiveness
            ]
            
            overall_effectiveness = sum(effectiveness_factors) / len(effectiveness_factors)
            
            # Generate recommendations
            recommendations = []
            if self.resource_utilization > 85:
                recommendations.append("Consider requesting mutual aid - high resource utilization")
            if self.response_time_avg > 15:
                recommendations.append("Deploy mobile command units to reduce response times")
            if coordination_success_rate < 80:
                recommendations.append("Improve inter-agency communication protocols")
            if overall_effectiveness < 70:
                recommendations.append("Activate emergency operations center coordination")
            
            assessment = {
                "assessment_period_hours": time_period_hours,
                "resource_utilization_percentage": round(self.resource_utilization, 1),
                "average_response_time_minutes": round(self.response_time_avg, 1),
                "coordination_success_rate": round(coordination_success_rate, 1),
                "overall_effectiveness_score": round(overall_effectiveness, 1),
                "active_deployments": len([d for d in self.resource_deployments.values() 
                                         if d["estimated_end"] > current_time]),
                "recent_coordination_events": len(recent_coordination),
                "recommendations": recommendations,
                "status": "excellent" if overall_effectiveness > 85 else 
                         "good" if overall_effectiveness > 70 else
                         "needs_improvement" if overall_effectiveness > 50 else "critical"
            }
            
            logger.info(f"Response effectiveness assessed: {overall_effectiveness:.1f}% overall score")
            return assessment
        
        def prioritize_emergency_incidents(incidents_json: str = "[]") -> Dict[str, Any]:
            """
            Prioritize and triage multiple emergency incidents.
            
            Args:
                incidents_json: JSON array of incident objects
                
            Returns:
                Prioritized incident list and resource allocation plan
            """
            try:
                incidents_data = json.loads(incidents_json) if incidents_json != "[]" else []
            except:
                incidents_data = []
            
            # Add default incidents if none provided
            if not incidents_data:
                incidents_data = [
                    {
                        "location": "Downtown District",
                        "type": "building_collapse",
                        "severity": 8,
                        "estimated_casualties": 15,
                        "resources_needed": ["search_rescue", "ambulances", "fire_rescue"]
                    },
                    {
                        "location": "Coastal Area",
                        "type": "flood_evacuation",
                        "severity": 6,
                        "estimated_casualties": 5,
                        "resources_needed": ["police_units", "ambulances"]
                    }
                ]
            
            # Convert to incident reports and prioritize
            prioritized_incidents = []
            for i, incident_data in enumerate(incidents_data):
                # Calculate priority based on severity and casualties
                severity = incident_data.get("severity", 5)
                casualties = incident_data.get("estimated_casualties", 0)
                
                priority_score = (severity * 0.6) + (min(casualties, 20) * 0.4)
                
                if priority_score >= 8:
                    priority = ResponsePriority.CRITICAL
                elif priority_score >= 6:
                    priority = ResponsePriority.HIGH
                elif priority_score >= 4:
                    priority = ResponsePriority.MEDIUM
                else:
                    priority = ResponsePriority.LOW
                
                incident = IncidentReport(
                    incident_id=f"incident_{i+1}_{datetime.utcnow().strftime('%H%M')}",
                    location=incident_data.get("location", f"Location_{i+1}"),
                    severity=severity,
                    resource_needs=incident_data.get("resources_needed", ["police_units"]),
                    priority=priority,
                    status="pending"
                )
                
                prioritized_incidents.append(incident)
            
            # Sort by priority and severity
            priority_order = {ResponsePriority.CRITICAL: 4, ResponsePriority.HIGH: 3, 
                            ResponsePriority.MEDIUM: 2, ResponsePriority.LOW: 1}
            
            prioritized_incidents.sort(key=lambda x: (priority_order[x.priority], x.severity), reverse=True)
            
            # Generate resource allocation plan
            allocation_plan = {}
            for incident in prioritized_incidents:
                allocation_plan[incident.incident_id] = {
                    "priority_rank": prioritized_incidents.index(incident) + 1,
                    "priority_level": incident.priority.value,
                    "location": incident.location,
                    "recommended_resources": incident.resource_needs,
                    "estimated_response_time": self._estimate_response_time(incident.resource_needs),
                    "coordination_needed": len(incident.resource_needs) > 2
                }
            
            # Update active incidents
            self.active_incidents = prioritized_incidents
            
            result = {
                "total_incidents": len(prioritized_incidents),
                "critical_incidents": len([i for i in prioritized_incidents if i.priority == ResponsePriority.CRITICAL]),
                "high_priority_incidents": len([i for i in prioritized_incidents if i.priority == ResponsePriority.HIGH]),
                "prioritized_list": [
                    {
                        "incident_id": inc.incident_id,
                        "location": inc.location,
                        "priority": inc.priority.value,
                        "severity": inc.severity,
                        "resources_needed": inc.resource_needs
                    }
                    for inc in prioritized_incidents
                ],
                "resource_allocation_plan": allocation_plan,
                "immediate_action_required": len([i for i in prioritized_incidents 
                                                if i.priority == ResponsePriority.CRITICAL]) > 0
            }
            
            logger.info(f"Emergency incidents prioritized: {len(prioritized_incidents)} total, "
                       f"{result['critical_incidents']} critical")
            return result
        
        # Create emergency response ADK agent
        return ERISAgentFactory.create_eris_agent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            model="gemini-2.0-flash",
            custom_tools=[deploy_emergency_resources, coordinate_multi_agency_response, 
                         assess_response_effectiveness, prioritize_emergency_incidents]
        )
    
    def _estimate_response_time(self, resource_types: List[str]) -> float:
        """Estimate response time for resource combination"""
        if not resource_types:
            return 0.0
        
        response_times = []
        for resource_type in resource_types:
            if resource_type in self.resources:
                response_times.append(self.resources[resource_type].response_time_minutes)
        
        return max(response_times) if response_times else 15.0  # Max response time determines arrival
    
    async def initialize_for_simulation(self, simulation_id: str, disaster_type: str, severity: int, location: str):
        """Initialize agent for a specific simulation"""
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.disaster_severity = severity
        
        # Adjust resource availability based on disaster type and severity
        await self._adjust_emergency_resources(disaster_type, severity)
        
        # Reset metrics
        self.response_effectiveness = 0.0
        self.response_time_avg = 0.0
        self.resource_utilization = 0.0
        self.incidents_resolved = 0
        self.coordination_requests = 0
        self.active_incidents = []
        self.resource_deployments = {}
        self.coordination_log = []
        
        logger.info(f"Emergency Response Agent initialized for {disaster_type} severity {severity}")
    
    async def _adjust_emergency_resources(self, disaster_type: str, severity: int):
        """Adjust resource availability based on disaster characteristics"""
        
        # Disaster-specific resource adjustments
        if disaster_type in ['earthquake', 'building_collapse']:
            self.resources["search_rescue"].total_available = min(20, self.resources["search_rescue"].total_available + 8)
            self.resources["fire_rescue"].total_available = min(35, self.resources["fire_rescue"].total_available + 10)
        
        elif disaster_type in ['flood', 'tsunami']:
            self.resources["search_rescue"].total_available = min(18, self.resources["search_rescue"].total_available + 6)
            self.resources["helicopter_rescue"].total_available = min(5, self.resources["helicopter_rescue"].total_available + 2)
        
        elif disaster_type in ['wildfire', 'volcanic_eruption']:
            self.resources["fire_rescue"].total_available = min(40, self.resources["fire_rescue"].total_available + 15)
            self.resources["helicopter_rescue"].total_available = min(6, self.resources["helicopter_rescue"].total_available + 3)
        
        elif disaster_type in ['hurricane', 'severe_storm']:
            self.resources["police_units"].total_available = min(60, self.resources["police_units"].total_available + 20)
            self.resources["mobile_command"].total_available = min(4, self.resources["mobile_command"].total_available + 2)
        
        # Severity-based response time adjustments
        severity_factor = 1.0 - (severity / 20)  # Higher severity = faster response
        for resource in self.resources.values():
            resource.response_time_minutes *= max(0.5, severity_factor)
        
        logger.info(f"Emergency resources adjusted for {disaster_type} severity {severity}")
    
    async def process_phase(self, phase: SimulationPhase, simulation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process emergency response for a specific simulation phase"""
        self.current_phase = phase
        
        # Execute phase-specific emergency response logic
        phase_results = await self._process_phase_specific_logic(phase, simulation_context)
        
        # Generate comprehensive metrics
        metrics = await self._generate_emergency_metrics()
        
        # Save state to cloud services
        await self._save_emergency_state(metrics)
        
        # Log to BigQuery for analytics
        await self._log_emergency_event(phase, metrics)
        
        return {
            "agent_id": self.agent_id,
            "phase": phase.value,
            "emergency_metrics": metrics,
            "phase_actions": phase_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _process_phase_specific_logic(self, phase: SimulationPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute phase-specific emergency response logic"""
        
        if phase == SimulationPhase.IMPACT:
            return await self._process_impact_phase(context)
        elif phase == SimulationPhase.RESPONSE:
            return await self._process_response_phase(context)
        elif phase == SimulationPhase.RECOVERY:
            return await self._process_recovery_phase(context)
        
        return {}
    
    async def _process_impact_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process immediate impact response"""
        
        # Create initial incidents based on disaster severity
        initial_incidents = self._generate_initial_incidents(context)
        
        try:
            # Prioritize incidents
            incidents_json = json.dumps([
                {
                    "location": inc.location,
                    "type": inc.incident_id.split('_')[0],
                    "severity": inc.severity,
                    "estimated_casualties": inc.severity * 2,
                    "resources_needed": inc.resource_needs
                }
                for inc in initial_incidents
            ])
            
            # Use internal implementation instead of ADK tool call
            prioritized_incidents = []
            for i, incident_data in enumerate([
                {
                    "location": inc.location,
                    "type": inc.incident_id.split('_')[0],
                    "severity": inc.severity,
                    "estimated_casualties": inc.severity * 2,
                    "resources_needed": inc.resource_needs
                }
                for inc in initial_incidents
            ]):
                severity = incident_data.get("severity", 5)
                casualties = incident_data.get("estimated_casualties", 0)
                priority_score = (severity * 0.6) + (min(casualties, 20) * 0.4)
                
                if priority_score >= 8:
                    priority = ResponsePriority.CRITICAL
                elif priority_score >= 6:
                    priority = ResponsePriority.HIGH
                elif priority_score >= 4:
                    priority = ResponsePriority.MEDIUM
                else:
                    priority = ResponsePriority.LOW
                
                incident = IncidentReport(
                    incident_id=f"impact_{i+1}_{datetime.utcnow().strftime('%H%M')}",
                    location=incident_data.get("location", f"Location_{i+1}"),
                    severity=severity,
                    resource_needs=incident_data.get("resources_needed", ["police_units"]),
                    priority=priority,
                    status="pending"
                )
                prioritized_incidents.append(incident)
            
            self.active_incidents = prioritized_incidents
            
            prioritization_result = {
                "total_incidents": len(prioritized_incidents),
                "critical_incidents": len([i for i in prioritized_incidents if i.priority == ResponsePriority.CRITICAL]),
                "high_priority_incidents": len([i for i in prioritized_incidents if i.priority == ResponsePriority.HIGH])
            }
            
            # Deploy initial resources to critical incidents
            critical_incidents = [i for i in prioritized_incidents if i.priority == ResponsePriority.CRITICAL]
            deployment_results = []
            
            for incident in critical_incidents[:3]:  # Handle top 3 critical incidents
                # Deploy resources using internal logic
                resource_types = incident.resource_needs
                deployed_resources = {}
                unavailable_resources = []
                
                for resource_type in resource_types:
                    if resource_type in self.resources:
                        resource = self.resources[resource_type]
                        available = resource.total_available - resource.currently_deployed
                        
                        if available > 0:
                            units_to_deploy = min(available, 3)  # Critical priority gets up to 3 units
                            resource.currently_deployed += units_to_deploy
                            
                            deployed_resources[resource_type] = {
                                "units_deployed": units_to_deploy,
                                "response_time_minutes": resource.response_time_minutes
                            }
                        else:
                            unavailable_resources.append(resource_type)
                
                deployment_result = {
                    "incident_location": incident.location,
                    "deployed_resources": deployed_resources,
                    "unavailable_resources": unavailable_resources,
                    "deployment_status": "success" if deployed_resources else "failed"
                }
                deployment_results.append(deployment_result)
            
            # Coordinate with multiple agencies
            agencies = ["fire_department", "police", "medical_services", "emergency_management"]
            coordination_result = {
                "agencies_involved": agencies,
                "coordination_type": "command",
                "status": "initiated",
                "unified_command": True
            }
            
            self.coordination_requests += 1
            self.coordination_log.append({
                "action": "multi_agency_coordination",
                "agencies": agencies,
                "type": "command",
                "timestamp": datetime.utcnow().isoformat(),
                "effectiveness": 0.8
            })
            
        except Exception as e:
            logger.warning(f"Emergency response impact phase error: {e}")
            # Create fallback results
            prioritization_result = {"total_incidents": 3, "critical_incidents": 2}
            deployment_results = [{"deployment_status": "partial"}]
            coordination_result = {"status": "initiated"}
        
        return {
            "incident_prioritization": prioritization_result,
            "resource_deployments": deployment_results,
            "multi_agency_coordination": coordination_result,
            "phase_focus": "immediate_life_safety_response"
        }
    
    async def _process_response_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process coordinated response phase"""
        
        try:
            # Assess current response effectiveness
            current_time = datetime.utcnow()
            
            # Calculate resource utilization
            total_resources = sum(r.total_available for r in self.resources.values())
            deployed_resources = sum(r.currently_deployed for r in self.resources.values())
            self.resource_utilization = (deployed_resources / total_resources) * 100 if total_resources > 0 else 0
            
            # Calculate average response time
            response_times = [r.response_time_minutes for r in self.resources.values()]
            self.response_time_avg = sum(response_times) / len(response_times) if response_times else 0
            
            # Overall effectiveness
            effectiveness_factors = [
                min(100, self.resource_utilization * 1.2),
                max(0, 100 - (self.response_time_avg / 20) * 100),
                80,  # Coordination success rate
                75   # Overall effectiveness
            ]
            overall_effectiveness = sum(effectiveness_factors) / len(effectiveness_factors)
            
            effectiveness_result = {
                "resource_utilization_percentage": round(self.resource_utilization, 1),
                "average_response_time_minutes": round(self.response_time_avg, 1),
                "overall_effectiveness_score": round(overall_effectiveness, 1),
                "status": "good" if overall_effectiveness > 70 else "needs_improvement"
            }
            
            # Continue resource deployments for ongoing incidents
            ongoing_deployments = []
            for incident in self.active_incidents:
                if incident.status == "pending":
                    deployment = {
                        "incident_location": incident.location,
                        "deployed_resources": {res: {"units_deployed": 1} for res in incident.resource_needs[:2]},
                        "deployment_status": "ongoing"
                    }
                    ongoing_deployments.append(deployment)
                    incident.status = "responding"
            
            # Enhanced coordination
            coordination_result = {
                "coordination_type": "resource_sharing",
                "agencies_involved": ["emergency_management", "public_works", "utilities"],
                "mutual_aid_activated": True,
                "status": "active"
            }
            
        except Exception as e:
            logger.warning(f"Emergency response response phase error: {e}")
            effectiveness_result = {"overall_effectiveness_score": 75, "status": "good"}
            ongoing_deployments = [{"deployment_status": "ongoing"}]
            coordination_result = {"status": "active"}
        
        return {
            "response_effectiveness": effectiveness_result,
            "ongoing_deployments": ongoing_deployments,
            "enhanced_coordination": coordination_result,
            "phase_focus": "sustained_response_operations"
        }
    
    async def _process_recovery_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process recovery coordination"""
        
        try:
            # Transition resources from emergency to recovery operations
            recovery_deployments = []
            
            # Return some emergency resources to normal operations
            for resource in self.resources.values():
                if resource.currently_deployed > 0:
                    units_to_return = min(resource.currently_deployed, 2)  # Return 2 units max
                    resource.currently_deployed -= units_to_return
                    
                    recovery_deployments.append({
                        "resource_type": resource.resource_type,
                        "units_returned": units_to_return,
                        "remaining_deployed": resource.currently_deployed
                    })
            
            # Assessment of recovery operations
            recovery_effectiveness = min(95, 60 + (self.disaster_severity * 3))
            
            effectiveness_result = {
                "recovery_effectiveness_score": recovery_effectiveness,
                "resource_transition": "successful",
                "emergency_phase_complete": True,
                "lessons_learned": [
                    "Multi-agency coordination protocols effective",
                    "Resource allocation met critical needs",
                    "Communication systems performed adequately"
                ]
            }
            
            # Coordination transition to recovery agencies
            coordination_result = {
                "coordination_type": "recovery_transition",
                "agencies_involved": ["emergency_management", "public_works", "community_services", "insurance"],
                "transition_status": "initiated",
                "recovery_planning": "active"
            }
            
            # Mark most incidents as resolved
            for incident in self.active_incidents:
                if incident.status != "resolved":
                    incident.status = "resolved"
                    self.incidents_resolved += 1
            
        except Exception as e:
            logger.warning(f"Emergency response recovery phase error: {e}")
            recovery_deployments = [{"units_returned": 5}]
            effectiveness_result = {"recovery_effectiveness_score": 80}
            coordination_result = {"transition_status": "initiated"}
        
        return {
            "resource_transition": recovery_deployments,
            "recovery_effectiveness": effectiveness_result,
            "coordination_transition": coordination_result,
            "incidents_resolved": self.incidents_resolved,
            "phase_focus": "transition_to_recovery_operations"
        }
    
    def _generate_initial_incidents(self, context: Dict[str, Any]) -> List[IncidentReport]:
        """Generate initial incidents based on disaster context"""
        incidents = []
        severity = self.disaster_severity
        
        # Base incidents by disaster type
        disaster_incidents = {
            'earthquake': [
                ("building_collapse", ["search_rescue", "fire_rescue", "ambulances"], 8),
                ("infrastructure_damage", ["fire_rescue", "police_units"], 6),
                ("trapped_people", ["search_rescue", "ambulances"], 9)
            ],
            'tsunami': [
                ("evacuation_assistance", ["police_units", "ambulances"], 7),
                ("water_rescue", ["search_rescue", "helicopter_rescue"], 9),
                ("structural_damage", ["fire_rescue", "hazmat_teams"], 6)
            ],
            'flood': [
                ("water_rescue", ["search_rescue", "helicopter_rescue"], 7),
                ("evacuation_center", ["police_units", "ambulances"], 5),
                ("infrastructure_failure", ["fire_rescue", "hazmat_teams"], 6)
            ],
            'wildfire': [
                ("evacuation_support", ["police_units", "fire_rescue"], 8),
                ("medical_emergency", ["ambulances", "helicopter_rescue"], 6),
                ("structure_protection", ["fire_rescue"], 7)
            ]
        }
        
        incident_templates = disaster_incidents.get(self.disaster_type, disaster_incidents['earthquake'])
        
        for i, (incident_type, resources, base_severity) in enumerate(incident_templates):
            adjusted_severity = min(10, base_severity + (severity - 5))
            
            incident = IncidentReport(
                incident_id=f"{incident_type}_{i+1}",
                location=f"Zone_{i+1}",
                severity=adjusted_severity,
                resource_needs=resources,
                priority=ResponsePriority.CRITICAL if adjusted_severity >= 8 else ResponsePriority.HIGH,
                status="pending"
            )
            incidents.append(incident)
        
        return incidents
    
    async def _generate_emergency_metrics(self) -> Dict[str, Any]:
        """Generate comprehensive emergency response metrics"""
        
        # Calculate current resource status
        resource_status = {}
        for name, resource in self.resources.items():
            utilization = (resource.currently_deployed / resource.total_available) * 100 if resource.total_available > 0 else 0
            resource_status[name] = {
                "total_available": resource.total_available,
                "currently_deployed": resource.currently_deployed,
                "utilization_percentage": round(utilization, 1),
                "average_response_time": resource.response_time_minutes
            }
        
        # Calculate overall metrics
        total_resources = sum(r.total_available for r in self.resources.values())
        total_deployed = sum(r.currently_deployed for r in self.resources.values())
        overall_utilization = (total_deployed / total_resources) * 100 if total_resources > 0 else 0
        
        return {
            "resource_metrics": {
                "overall_utilization_percentage": round(overall_utilization, 1),
                "total_resources_available": total_resources,
                "resources_currently_deployed": total_deployed,
                "resource_status": resource_status
            },
            "response_metrics": {
                "average_response_time_minutes": self.response_time_avg,
                "response_effectiveness_score": round(self.response_effectiveness * 100, 1),
                "active_incidents": len([i for i in self.active_incidents if i.status != "resolved"]),
                "incidents_resolved": self.incidents_resolved,
                "coordination_requests": self.coordination_requests
            },
            "operational_metrics": {
                "active_deployments": len(self.resource_deployments),
                "multi_agency_coordinations": len([c for c in self.coordination_log if c["action"] == "multi_agency_coordination"]),
                "system_status": "operational" if overall_utilization < 90 else "strained",
                "phase": self.current_phase.value
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _save_emergency_state(self, metrics: Dict[str, Any]):
        """Save emergency response state to Firestore"""
        try:
            state_data = {
                "agent_id": self.agent_id,
                "simulation_id": self.simulation_id,
                "resource_status": {
                    name: {
                        "total": resource.total_available,
                        "deployed": resource.currently_deployed,
                        "response_time": resource.response_time_minutes
                    }
                    for name, resource in self.resources.items()
                },
                "active_incidents": [
                    {
                        "incident_id": inc.incident_id,
                        "location": inc.location,
                        "severity": inc.severity,
                        "priority": inc.priority.value,
                        "status": inc.status
                    }
                    for inc in self.active_incidents
                ],
                "metrics": metrics,
                "phase": self.current_phase.value,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.cloud.firestore.save_agent_state(self.agent_id, self.simulation_id, state_data)
            
        except Exception as e:
            logger.error(f"Failed to save emergency response state: {e}")
    
    async def _log_emergency_event(self, phase: SimulationPhase, metrics: Dict[str, Any]):
        """Log emergency response events to BigQuery"""
        try:
            event_data = {
                "event_type": "emergency_response_update",
                "agent_id": self.agent_id,
                "phase": phase.value,
                "resource_utilization": metrics["resource_metrics"]["overall_utilization_percentage"],
                "response_effectiveness": metrics["response_metrics"]["response_effectiveness_score"],
                "active_incidents": metrics["response_metrics"]["active_incidents"],
                "incidents_resolved": metrics["response_metrics"]["incidents_resolved"],
                "average_response_time": metrics["response_metrics"]["average_response_time_minutes"]
            }
            
            await self.cloud.bigquery.log_simulation_event(
                simulation_id=self.simulation_id,
                event_type="emergency_response_update",
                event_data=event_data,
                agent_id=self.agent_id,
                phase=phase.value
            )
            
        except Exception as e:
            logger.error(f"Failed to log emergency response event: {e}")


def create_emergency_response_agent(cloud_services: CloudServices) -> EmergencyResponseAgent:
    """Factory function to create an Emergency Response Agent"""
    return EmergencyResponseAgent(cloud_services)