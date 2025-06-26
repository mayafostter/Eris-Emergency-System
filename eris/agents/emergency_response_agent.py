"""
ERIS Emergency Response Agent with Dynamic Content Generation
Central command and control coordination for disaster response with AI-powered updates
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass
import random

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
    efficiency: float = 0.8

@dataclass
class EmergencyUpdate:
    content: str
    priority: str
    timestamp: datetime
    source: str
    update_type: str
    affected_areas: List[str]

class EnhancedEmergencyResponseAgent:
    def __init__(self, cloud_services: CloudServices):
        self.cloud = cloud_services
        self.agent_id = "emergency_response_coordinator"
        self.agent_type = "emergency_response"
        
        # Resource tracking
        self.resources = {
            "fire_rescue": EmergencyResource("fire_rescue", 25, 0, 8.5, 0.85),
            "police_units": EmergencyResource("police_units", 40, 0, 5.2, 0.90),
            "ambulances": EmergencyResource("ambulances", 18, 0, 7.1, 0.82),
            "search_rescue": EmergencyResource("search_rescue", 12, 0, 15.3, 0.78),
            "medical_teams": EmergencyResource("medical_teams", 15, 0, 12.0, 0.88),
            "evacuation_buses": EmergencyResource("evacuation_buses", 30, 0, 10.5, 0.75)
        }
        
        # Dynamic metrics
        self.response_effectiveness = 0.0
        self.incidents_resolved = 0
        self.incidents_active = 0
        self.emergency_updates = []
        self.coordination_events = []
        
        # Simulation state
        self.simulation_id = None
        self.current_phase = SimulationPhase.IMPACT
        self.disaster_type = None
        self.disaster_severity = 5
        self.location = "Phuket, Thailand"
        
        # Performance tracking
        self.last_update_time = datetime.utcnow()
        self.response_time_avg = 0.0
        
        self.adk_agent = self._create_agent()
    
    def _create_agent(self) -> LlmAgent:
        def deploy_resources_dynamically(location: str, resource_types: str, priority: str = "medium", incident_type: str = "general") -> Dict[str, Any]:
            """Resource deployment with dynamic allocation"""
            requested = [r.strip() for r in resource_types.split(',')]
            deployed = {}
            deployment_success = True
            
            # Priority-based deployment multipliers
            priority_multipliers = {
                "critical": 3,
                "high": 2, 
                "medium": 1,
                "low": 1
            }
            
            multiplier = priority_multipliers.get(priority.lower(), 1)
            
            for resource_type in requested:
                if resource_type in self.resources:
                    resource = self.resources[resource_type]
                    available = resource.total - resource.deployed
                    
                    if available > 0:
                        # Calculate units needed based on priority and incident type
                        base_units = 1
                        if incident_type in ["fire", "medical_emergency", "search_rescue"]:
                            base_units = 2
                        
                        units_needed = min(available, base_units * multiplier)
                        
                        # Deployment
                        resource.deployed += units_needed
                        
                        # Calculate adjusted response time based on current load
                        load_factor = resource.deployed / resource.total
                        adjusted_response_time = resource.response_time * (1 + load_factor * 0.5)
                        
                        deployed[resource_type] = {
                            "units": units_needed,
                            "response_time": round(adjusted_response_time, 2),
                            "efficiency": round(resource.efficiency * (1 - load_factor * 0.2), 2),
                            "status": "dispatched"
                        }
                    else:
                        deployed[resource_type] = {
                            "units": 0,
                            "response_time": "N/A",
                            "efficiency": 0,
                            "status": "unavailable"
                        }
                        deployment_success = False
            
            # Log deployment
            self._log_deployment_event(location, deployed, priority, incident_type)
            
            return {
                "location": location,
                "deployed_resources": deployed,
                "status": "success" if deployment_success else "partial",
                "deployment_time": datetime.utcnow().isoformat(),
                "priority": priority,
                "incident_type": incident_type
            }
        
        def coordinate_multi_agency_response(agencies: str, coordination_type: str = "standard", scenario_complexity: str = "medium") -> Dict[str, Any]:
            """Multi-agency coordination with complexity handling"""
            agency_list = [a.strip() for a in agencies.split(',')]
            
            # Base effectiveness calculation
            base_effectiveness = 0.7
            
            # Agency synergy bonuses
            synergy_bonus = len(agency_list) * 0.05  # 5% per additional agency
            
            # Scenario complexity impacts
            complexity_modifiers = {
                "low": 1.1,
                "medium": 1.0,
                "high": 0.85,
                "critical": 0.7
            }
            
            complexity_modifier = complexity_modifiers.get(scenario_complexity.lower(), 1.0)
            
            # Coordination type bonuses
            coordination_bonuses = {
                "standard": 1.0,
                "enhanced": 1.15,
                "unified_command": 1.25,
                "emergency_operations": 1.3
            }
            
            coordination_bonus = coordination_bonuses.get(coordination_type.lower(), 1.0)
            
            # Calculate final effectiveness
            self.response_effectiveness = min(1.0, 
                (base_effectiveness + synergy_bonus) * complexity_modifier * coordination_bonus
            )
            
            # Create coordination event
            coordination_event = {
                "agencies": agency_list,
                "coordination_type": coordination_type,
                "effectiveness": round(self.response_effectiveness, 3),
                "complexity": scenario_complexity,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "coordinated"
            }
            
            self.coordination_events.append(coordination_event)
            
            return coordination_event
        
        def assess_dynamic_response_status() -> Dict[str, Any]:
            """Response status assessment with real-time calculations"""
            # Calculate resource utilization
            total_resources = sum(r.total for r in self.resources.values())
            deployed_resources = sum(r.deployed for r in self.resources.values())
            utilization = (deployed_resources / total_resources) * 100 if total_resources > 0 else 0
            
            # Calculate weighted effectiveness based on resource efficiency
            total_efficiency = 0
            total_weight = 0
            
            for resource in self.resources.values():
                if resource.deployed > 0:
                    weight = resource.deployed
                    total_efficiency += resource.efficiency * weight
                    total_weight += weight
            
            avg_efficiency = (total_efficiency / total_weight) if total_weight > 0 else 0.8
            
            # Calculate response time performance
            active_resources = [r for r in self.resources.values() if r.deployed > 0]
            if active_resources:
                self.response_time_avg = sum(r.response_time for r in active_resources) / len(active_resources)
            
            # Determine overall status
            if utilization > 90:
                status = "critical_load"
            elif utilization > 75:
                status = "high_load"
            elif utilization > 50:
                status = "operational"
            else:
                status = "ready"
            
            return {
                "resource_utilization": round(utilization, 1),
                "response_effectiveness": round(self.response_effectiveness * 100, 1),
                "incidents_resolved": self.incidents_resolved,
                "incidents_active": self.incidents_active,
                "avg_response_time": round(self.response_time_avg, 1),
                "avg_efficiency": round(avg_efficiency * 100, 1),
                "status": status,
                "resource_breakdown": {
                    name: {
                        "total": res.total,
                        "deployed": res.deployed,
                        "available": res.total - res.deployed,
                        "efficiency": round(res.efficiency * 100, 1)
                    } for name, res in self.resources.items()
                },
                "last_update": datetime.utcnow().isoformat()
            }
        
        def generate_emergency_update(update_type: str = "status", priority: str = "medium") -> Dict[str, Any]:
            """Generate dynamic emergency updates based on current situation"""
            try:
                # Create context for AI generation
                context = {
                    "disaster_type": self.disaster_type,
                    "location": self.location,
                    "severity": self.disaster_severity,
                    "phase": self.current_phase.value,
                    "resource_utilization": sum(r.deployed for r in self.resources.values()),
                    "incidents_active": self.incidents_active,
                    "response_effectiveness": self.response_effectiveness
                }
                
                # Generate update content
                update_content = self._generate_contextual_update(update_type, priority, context)
                
                # Create emergency update
                emergency_update = EmergencyUpdate(
                    content=update_content,
                    priority=priority,
                    timestamp=datetime.utcnow(),
                    source="@EmergencyCoordinator",
                    update_type=update_type,
                    affected_areas=[self.location.split(',')[0]]
                )
                
                self.emergency_updates.append(emergency_update)
                
                return {
                    "content": emergency_update.content,
                    "priority": emergency_update.priority,
                    "timestamp": emergency_update.timestamp.isoformat(),
                    "source": emergency_update.source,
                    "update_type": emergency_update.update_type,
                    "affected_areas": emergency_update.affected_areas,
                    "ai_generated": True
                }
                
            except Exception as e:
                logger.error(f"Failed to generate emergency update: {e}")
                return self._generate_fallback_update(update_type, priority)
        
        return ERISAgentFactory.create_eris_agent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            model="gemini-2.0-flash",
            custom_tools=[deploy_resources_dynamically, coordinate_multi_agency_response, assess_dynamic_response_status, generate_emergency_update]
        )
    
    def _generate_contextual_update(self, update_type: str, priority: str, context: Dict[str, Any]) -> str:
        """Generate contextual emergency updates"""
        templates = {
            "status": {
                "high": [
                    f"🚨 EMERGENCY UPDATE: {context['disaster_type'].title()} response in {context['location']} - {sum(r.deployed for r in self.resources.values())} units deployed. Response effectiveness at {int(context['response_effectiveness'] * 100)}%.",
                    f"CRITICAL: {context['disaster_type'].title()} situation in {context['location']} requires immediate attention. {context['incidents_active']} active incidents under management.",
                    f"URGENT: Emergency services coordinating response to {context['disaster_type'].title()} in {context['location']}. All available resources mobilized."
                ],
                "medium": [
                    f"Emergency Response Update: {context['disaster_type'].title()} response operations ongoing in {context['location']}. Current status: {self.current_phase.value.title()}.",
                    f"Coordination Update: Multi-agency response active for {context['disaster_type'].title()} in {context['location']}. Resource deployment proceeding as planned.",
                    f"Status Report: Emergency services responding to {context['disaster_type'].title()} - {sum(r.deployed for r in self.resources.values())} units in field."
                ],
                "low": [
                    f"Routine Update: {context['disaster_type'].title()} response in {context['location']} proceeding. Monitoring situation closely.",
                    f"Advisory: Emergency response teams positioned for {context['disaster_type'].title()} in {context['location']}. Situation under control."
                ]
            },
            "resource": {
                "high": [
                    f"RESOURCE ALERT: High demand for emergency services in {context['location']}. {sum(r.deployed for r in self.resources.values())}/{sum(r.total for r in self.resources.values())} units deployed.",
                    f"CRITICAL DEPLOYMENT: All available resources committed to {context['disaster_type'].title()} response in {context['location']}."
                ],
                "medium": [
                    f"Resource Deployment: Emergency units responding to {context['disaster_type'].title()} in {context['location']}. Capacity management active.",
                    f"Operational Status: Resource allocation optimized for {context['disaster_type'].title()} response in {context['location']}."
                ],
                "low": [
                    f"Resource Update: Emergency services maintaining readiness for {context['disaster_type'].title()} in {context['location']}."
                ]
            },
            "coordination": {
                "high": [
                    f"COORDINATION ALERT: Unified command activated for {context['disaster_type'].title()} in {context['location']}. Multi-agency response in effect.",
                    f"EMERGENCY COORDINATION: {len(self.coordination_events)} agencies coordinating response to {context['disaster_type'].title()}."
                ],
                "medium": [
                    f"Multi-Agency Coordination: Emergency services working together on {context['disaster_type'].title()} response in {context['location']}.",
                    f"Coordination Update: Response effectiveness at {int(context['response_effectiveness'] * 100)}% for {context['disaster_type'].title()} operations."
                ],
                "low": [
                    f"Coordination Status: Standard protocols in effect for {context['disaster_type'].title()} response in {context['location']}."
                ]
            }
        }
        
        update_templates = templates.get(update_type, templates["status"])
        priority_templates = update_templates.get(priority, update_templates["medium"])
        
        return random.choice(priority_templates)
    
    def _generate_fallback_update(self, update_type: str, priority: str) -> Dict[str, Any]:
        """Generate fallback emergency update if AI generation fails"""
        fallback_content = f"Emergency Response Coordinator: {update_type.title()} update for {self.disaster_type} in {self.location}. Priority: {priority.title()}. Response operations ongoing."
        
        return {
            "content": fallback_content,
            "priority": priority,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "@EmergencyCoordinator",
            "update_type": update_type,
            "affected_areas": [self.location.split(',')[0]],
            "ai_generated": False
        }
    
    def _log_deployment_event(self, location: str, deployed: Dict[str, Any], priority: str, incident_type: str):
        """Log deployment event for tracking"""
        try:
            event = {
                "event_type": "resource_deployment",
                "location": location,
                "deployed_resources": deployed,
                "priority": priority,
                "incident_type": incident_type,
                "timestamp": datetime.utcnow().isoformat(),
                "total_units": sum(d.get("units", 0) for d in deployed.values())
            }
            
            # Incidents tracking
            if incident_type != "general":
                self.incidents_active += 1
            
            logger.info(f"Resource deployment logged: {event['total_units']} units to {location}")
            
        except Exception as e:
            logger.error(f"Failed to log deployment event: {e}")
    
    async def initialize_for_simulation(self, simulation_id: str, disaster_type: str, severity: int, location: str):
        """Initialize for enhanced simulation"""
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.disaster_severity = severity
        self.location = location
        
        # Reset all resources
        for resource in self.resources.values():
            resource.deployed = 0
            resource.efficiency = random.uniform(0.75, 0.95)  # Add some variance
        
        # Reset metrics
        self.response_effectiveness = 0.0
        self.incidents_resolved = 0
        self.incidents_active = 0
        self.emergency_updates = []
        self.coordination_events = []
        self.last_update_time = datetime.utcnow()
        
        logger.info(f"Enhanced Emergency Response Agent initialized for {disaster_type} in {location}")
    
    async def process_phase(self, phase: SimulationPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced phase processing with dynamic content generation"""
        self.current_phase = phase
        
        try:
            # Phase-specific actions
            if phase == SimulationPhase.IMPACT:
                await self._handle_impact_phase(context)
            elif phase == SimulationPhase.RESPONSE:
                await self._handle_response_phase(context)
            else:  # RECOVERY
                await self._handle_recovery_phase(context)
            
            # Generate emergency updates
            updates = await self._generate_phase_updates(phase, context)
            
            # Calculate comprehensive metrics
            metrics = await self._calculate_comprehensive_metrics()
            
            # Save state
            await self._save_enhanced_state(metrics, updates)
            
            return {
                "agent_id": self.agent_id,
                "phase": phase.value,
                "metrics": metrics,
                "emergency_updates": updates,
                "timestamp": datetime.utcnow().isoformat(),
                "ai_enhanced": True
            }
            
        except Exception as e:
            logger.error(f"Enhanced emergency response processing error: {e}")
            return await self._generate_fallback_response(phase, context)
    
    async def _handle_impact_phase(self, context: Dict[str, Any]):
        """Handle impact phase with dynamic resource deployment"""
        # Rapid deployment based on disaster severity
        deployment_percentage = min(0.8, 0.4 + (self.disaster_severity / 20))
        
        for resource in self.resources.values():
            initial_deployment = int(resource.total * deployment_percentage)
            resource.deployed = min(resource.total, initial_deployment)
        
        # Generate initial incidents
        self.incidents_active = random.randint(3, 8) + self.disaster_severity
        
        # Set initial response effectiveness
        self.response_effectiveness = max(0.3, 0.8 - (self.disaster_severity / 20))
    
    async def _handle_response_phase(self, context: Dict[str, Any]):
        """Handle response phase with coordination improvements"""
        # Improve coordination
        self.response_effectiveness = min(0.95, self.response_effectiveness + 0.1)
        
        # Resolve some incidents
        incidents_resolved = random.randint(1, 3)
        self.incidents_resolved += incidents_resolved
        self.incidents_active = max(0, self.incidents_active - incidents_resolved)
        
        # Optimize resource deployment
        for resource in self.resources.values():
            if resource.deployed < resource.total and random.random() < 0.3:
                additional_units = min(resource.total - resource.deployed, 2)
                resource.deployed += additional_units
    
    async def _handle_recovery_phase(self, context: Dict[str, Any]):
        """Handle recovery phase with resource scaling down"""
        # Scale down operations
        for resource in self.resources.values():
            if resource.deployed > 2:
                resource.deployed = max(2, resource.deployed - random.randint(1, 3))
        
        # Resolve remaining incidents
        if self.incidents_active > 0:
            resolved = min(self.incidents_active, random.randint(1, 2))
            self.incidents_resolved += resolved
            self.incidents_active -= resolved
        
        # Maintain high effectiveness in recovery
        self.response_effectiveness = min(0.9, self.response_effectiveness + 0.05)
    
    async def _generate_phase_updates(self, phase: SimulationPhase, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate emergency updates for the current phase"""
        updates = []
        
        try:
            # Generate 2-4 updates per phase
            update_count = random.randint(2, 4)
            
            for i in range(update_count):
                # Vary update types and priorities
                update_types = ["status", "resource", "coordination"]
                priorities = ["high", "medium", "low"]
                
                # Weight priorities based on phase
                if phase == SimulationPhase.IMPACT:
                    priority_weights = [0.6, 0.3, 0.1]  # More high priority
                elif phase == SimulationPhase.RESPONSE:
                    priority_weights = [0.3, 0.5, 0.2]  # Balanced
                else:  # RECOVERY
                    priority_weights = [0.1, 0.4, 0.5]  # More low priority
                
                update_type = random.choice(update_types)
                priority = random.choices(priorities, weights=priority_weights)[0]
                
                # Generate update using the agent's tool
                try:
                    update_result = await self.adk_agent.use_tool("generate_emergency_update", {
                        "update_type": update_type,
                        "priority": priority
                    })
                    
                    if update_result and "content" in update_result:
                        updates.append(update_result)
                    
                except Exception as tool_error:
                    logger.warning(f"Tool execution failed: {tool_error}")
                    # Generate fallback update
                    fallback_update = self._generate_fallback_update(update_type, priority)
                    updates.append(fallback_update)
                
                # Add delay between updates for realism
                await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Failed to generate phase updates: {e}")
            # Generate at least one fallback update
            updates.append(self._generate_fallback_update("status", "medium"))
        
        return updates
    
    async def _calculate_comprehensive_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive metrics including AI-generated insights"""
        # Basic metrics
        total_resources = sum(r.total for r in self.resources.values())
        deployed_resources = sum(r.deployed for r in self.resources.values())
        resource_utilization = (deployed_resources / total_resources) * 100 if total_resources > 0 else 0
        
        # Response time calculation
        active_resources = [r for r in self.resources.values() if r.deployed > 0]
        avg_response_time = sum(r.response_time for r in active_resources) / len(active_resources) if active_resources else 0
        
        # Efficiency calculation
        total_efficiency = 0
        total_weight = 0
        for resource in self.resources.values():
            if resource.deployed > 0:
                weight = resource.deployed
                total_efficiency += resource.efficiency * weight
                total_weight += weight
        
        weighted_efficiency = (total_efficiency / total_weight) if total_weight > 0 else 0.8
        
        # Performance indicators
        incident_resolution_rate = (self.incidents_resolved / max(1, self.incidents_resolved + self.incidents_active)) * 100
        
        return {
            "resource_utilization": round(resource_utilization, 1),
            "response_effectiveness": round(self.response_effectiveness * 100, 1),
            "incidents_resolved": self.incidents_resolved,
            "incidents_active": self.incidents_active,
            "incident_resolution_rate": round(incident_resolution_rate, 1),
            "avg_response_time": round(avg_response_time, 1),
            "operational_efficiency": round(weighted_efficiency * 100, 1),
            "total_deployments": deployed_resources,
            "coordination_events": len(self.coordination_events),
            "emergency_updates_generated": len(self.emergency_updates),
            "status": self._determine_operational_status(resource_utilization, self.response_effectiveness),
            "resource_breakdown": {
                name: {
                    "total": res.total,
                    "deployed": res.deployed,
                    "available": res.total - res.deployed,
                    "efficiency": round(res.efficiency * 100, 1),
                    "utilization": round((res.deployed / res.total) * 100, 1)
                } for name, res in self.resources.items()
            },
            "recent_updates": [
                {
                    "content": update.content[:100] + "..." if len(update.content) > 100 else update.content,
                    "priority": update.priority,
                    "timestamp": update.timestamp.isoformat(),
                    "type": update.update_type
                } for update in self.emergency_updates[-3:]  # Last 3 updates
            ]
        }
    
    def _determine_operational_status(self, utilization: float, effectiveness: float) -> str:
        """Determine operational status based on metrics"""
        if utilization > 90 or effectiveness < 0.5:
            return "critical"
        elif utilization > 75 or effectiveness < 0.7:
            return "strained"
        elif utilization > 50 and effectiveness > 0.8:
            return "optimal"
        else:
            return "standby"
    
    async def get_recent_updates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent emergency updates for external consumption"""
        recent_updates = sorted(self.emergency_updates, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return [
            {
                "content": update.content,
                "priority": update.priority,
                "timestamp": update.timestamp.isoformat(),
                "source": update.source,
                "update_type": update.update_type,
                "affected_areas": update.affected_areas
            } for update in recent_updates
        ]
    
    async def generate_live_update(self, update_type: str = "status", priority: str = "medium") -> Dict[str, Any]:
        """Generate a live emergency update for real-time feeds"""
        try:
            # Use the agent's tool to generate update
            result = await self.adk_agent.use_tool("generate_emergency_update", {
                "update_type": update_type,
                "priority": priority
            })
            
            return result if result else self._generate_fallback_update(update_type, priority)
            
        except Exception as e:
            logger.error(f"Failed to generate live update: {e}")
            return self._generate_fallback_update(update_type, priority)
    
    async def _save_enhanced_state(self, metrics: Dict[str, Any], updates: List[Dict[str, Any]]):
        """Save agent state to cloud services"""
        try:
            state_data = {
                "agent_id": self.agent_id,
                "simulation_id": self.simulation_id,
                "metrics": metrics,
                "emergency_updates": updates,
                "resource_state": {
                    name: {
                        "total": res.total,
                        "deployed": res.deployed,
                        "efficiency": res.efficiency,
                        "response_time": res.response_time
                    } for name, res in self.resources.items()
                },
                "coordination_events": len(self.coordination_events),
                "phase": self.current_phase.value,
                "timestamp": datetime.utcnow().isoformat(),
                "ai_enhanced": True
            }
            
            await self.cloud.firestore.save_agent_state(self.agent_id, self.simulation_id, state_data)
            
        except Exception as e:
            logger.error(f"Failed to save enhanced emergency response state: {e}")
    
    async def _generate_fallback_response(self, phase: SimulationPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback response if processing fails"""
        fallback_metrics = {
            "resource_utilization": 75.0,
            "response_effectiveness": 85.0,
            "incidents_resolved": self.incidents_resolved,
            "incidents_active": max(1, self.incidents_active),
            "status": "operational"
        }
        
        fallback_updates = [
            self._generate_fallback_update("status", "medium")
        ]
        
        return {
            "agent_id": self.agent_id,
            "phase": phase.value,
            "metrics": fallback_metrics,
            "emergency_updates": fallback_updates,
            "timestamp": datetime.utcnow().isoformat(),
            "fallback_mode": True
        }


def create_enhanced_emergency_response_agent(cloud_services: CloudServices) -> EnhancedEmergencyResponseAgent:
    """Factory function to create Enhanced Emergency Response Agent"""
    return EnhancedEmergencyResponseAgent(cloud_services)
