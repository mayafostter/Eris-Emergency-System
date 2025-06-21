"""
ERIS Recovery Coordinator Agent - Compact version
Long-term restoration and recovery planning coordination
"""

import logging
from datetime import datetime
from typing import Dict, Any, List

from google.adk.agents import Agent as LlmAgent
from agents.base_agent import ERISAgentFactory
from services import CloudServices
from utils.time_utils import SimulationPhase

logger = logging.getLogger(__name__)

class RecoveryCoordinatorAgent:
    def __init__(self, cloud_services: CloudServices):
        self.cloud = cloud_services
        self.agent_id = "recovery_coordinator"
        self.agent_type = "recovery"
        
        # Recovery sectors
        self.recovery_sectors = {
            "housing": {"progress": 0, "projects": 0, "budget": 25000000},
            "infrastructure": {"progress": 0, "projects": 0, "budget": 40000000},
            "economic": {"progress": 0, "projects": 0, "budget": 15000000},
            "social_services": {"progress": 0, "projects": 0, "budget": 10000000}
        }
        
        # Recovery metrics
        self.overall_recovery = 0.0
        self.community_resilience = 0.6
        self.lessons_learned = []
        self.stakeholder_engagement = 0.7
        
        # Simulation state
        self.simulation_id = None
        self.current_phase = SimulationPhase.RECOVERY
        self.disaster_type = None
        self.disaster_severity = 5
        
        self.adk_agent = self._create_agent()
    
    def _create_agent(self) -> LlmAgent:
        def develop_recovery_plan(sectors: str, timeline_months: int = 24, budget: float = 100000000) -> Dict[str, Any]:
            """Develop comprehensive recovery plan"""
            sector_list = [s.strip() for s in sectors.split(',')]
            
            # Generate projects for each sector
            total_projects = 0
            allocated_budget = 0
            
            for sector in sector_list:
                if sector in self.recovery_sectors:
                    # Simple project allocation
                    if sector == "housing":
                        projects = 5
                        sector_budget = 25000000
                    elif sector == "infrastructure":
                        projects = 8
                        sector_budget = 40000000
                    elif sector == "economic":
                        projects = 3
                        sector_budget = 15000000
                    else:  # social_services
                        projects = 4
                        sector_budget = 10000000
                    
                    self.recovery_sectors[sector]["projects"] = projects
                    total_projects += projects
                    allocated_budget += sector_budget
            
            return {
                "planning_scope": "comprehensive",
                "sectors": sector_list,
                "timeline_months": timeline_months,
                "total_projects": total_projects,
                "allocated_budget": allocated_budget,
                "funding_coverage": round((allocated_budget / budget) * 100, 1),
                "status": "planned"
            }
        
        def coordinate_rebuilding(sector: str, coordination_level: str = "standard") -> Dict[str, Any]:
            """Coordinate rebuilding efforts for specific sector"""
            if sector not in self.recovery_sectors:
                return {"status": "failed", "error": "Unknown sector"}
            
            sector_data = self.recovery_sectors[sector]
            
            # Simulate progress based on coordination level
            if coordination_level == "comprehensive":
                progress_increase = 30
            elif coordination_level == "standard":
                progress_increase = 20
            else:  # basic
                progress_increase = 15
            
            sector_data["progress"] = min(100, sector_data["progress"] + progress_increase)
            
            # Update overall recovery
            self.overall_recovery = sum(s["progress"] for s in self.recovery_sectors.values()) / len(self.recovery_sectors)
            
            return {
                "sector": sector,
                "coordination_level": coordination_level,
                "current_progress": sector_data["progress"],
                "projects_active": sector_data["projects"],
                "overall_recovery": round(self.overall_recovery, 1),
                "status": "coordinated"
            }
        
        def assess_community_needs(scope: str = "city_wide", focus: str = "all") -> Dict[str, Any]:
            """Assess community recovery needs"""
            # Simulate needs assessment
            needs_identified = {
                "housing": {"critical": 3, "high": 5, "medium": 8},
                "services": {"critical": 2, "high": 4, "medium": 6},
                "economic": {"critical": 1, "high": 3, "medium": 4}
            }
            
            total_needs = sum(
                sum(severity.values()) for severity in needs_identified.values()
            )
            
            critical_needs = sum(
                severity["critical"] for severity in needs_identified.values()
            )
            
            # Update community resilience
            resilience_factor = 1 - (critical_needs / max(1, total_needs))
            self.community_resilience = min(1.0, self.community_resilience + (resilience_factor * 0.1))
            
            return {
                "assessment_scope": scope,
                "needs_identified": needs_identified,
                "total_needs": total_needs,
                "critical_needs": critical_needs,
                "community_resilience": round(self.community_resilience, 2),
                "priority_recommendations": ["housing_assistance", "economic_support"],
                "status": "assessed"
            }
        
        def generate_lessons_learned(scope: str = "comprehensive") -> Dict[str, Any]:
            """Generate lessons learned from response and recovery"""
            # Simulate lessons learned collection
            lessons = {
                "impact": [
                    "Early warning systems effective",
                    "Emergency supplies distribution improved",
                    "Communication systems need backup"
                ],
                "response": [
                    "Multi-agency coordination successful",
                    "Resource allocation process efficient",
                    "Public messaging was clear"
                ],
                "recovery": [
                    "Community engagement crucial",
                    "Funding processes need streamlining",
                    "Build-back-better principles applied"
                ]
            }
            
            recommendations = [
                "Strengthen communication redundancy",
                "Improve inter-agency protocols",
                "Enhance community preparedness",
                "Streamline recovery funding"
            ]
            
            # Store lessons
            lesson_doc = {
                "scope": scope,
                "disaster_type": self.disaster_type,
                "lessons": lessons,
                "recommendations": recommendations,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.lessons_learned.append(lesson_doc)
            
            return {
                "evaluation_scope": scope,
                "lessons_by_phase": lessons,
                "recommendations": recommendations,
                "total_lessons": sum(len(phase_lessons) for phase_lessons in lessons.values()),
                "improvement_areas": len(recommendations),
                "documentation_complete": True
            }
        
        return ERISAgentFactory.create_eris_agent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            custom_tools=[develop_recovery_plan, coordinate_rebuilding, 
                         assess_community_needs, generate_lessons_learned]
        )
    
    async def initialize_for_simulation(self, simulation_id: str, disaster_type: str, severity: int, location: str):
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.disaster_severity = severity
        
        # Reset recovery state
        for sector in self.recovery_sectors.values():
            sector["progress"] = 0
            sector["projects"] = 0
        
        self.overall_recovery = 0.0
        self.community_resilience = 0.6
        self.lessons_learned = []
        self.stakeholder_engagement = 0.7
        
        logger.info(f"Recovery Coordinator Agent initialized for {disaster_type}")
    
    async def process_phase(self, phase: SimulationPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        self.current_phase = phase
        
        # Simple phase processing
        if phase == SimulationPhase.IMPACT:
            # Initial recovery planning
            for sector in self.recovery_sectors.values():
                sector["projects"] = 2  # Initial planning projects
            result = {"action": "recovery_planning", "sectors_planned": len(self.recovery_sectors)}
            
        elif phase == SimulationPhase.RESPONSE:
            # Begin coordination
            for sector in self.recovery_sectors.values():
                sector["progress"] = 20  # 20% progress
                sector["projects"] = 4
            
            self.overall_recovery = 20.0
            self.stakeholder_engagement = 0.8
            result = {"action": "rebuilding_coordination", "progress": self.overall_recovery}
            
        else:  # RECOVERY
            # Full recovery operations
            for sector in self.recovery_sectors.values():
                sector["progress"] = 75  # 75% progress
                sector["projects"] = 6
            
            self.overall_recovery = 75.0
            self.community_resilience = 0.8
            
            # Generate lessons learned
            lesson_doc = {
                "scope": "comprehensive",
                "disaster_type": self.disaster_type,
                "lessons_count": 9,
                "recommendations_count": 4,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.lessons_learned.append(lesson_doc)
            
            result = {"action": "recovery_operations", "progress": self.overall_recovery, "lessons_documented": True}
        
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
        total_projects = sum(s["projects"] for s in self.recovery_sectors.values())
        avg_progress = sum(s["progress"] for s in self.recovery_sectors.values()) / len(self.recovery_sectors)
        
        return {
            "overall_recovery": round(self.overall_recovery, 1),
            "community_resilience": round(self.community_resilience, 2),
            "total_projects": total_projects,
            "average_progress": round(avg_progress, 1),
            "stakeholder_engagement": round(self.stakeholder_engagement, 2),
            "lessons_learned": len(self.lessons_learned),
            "recovery_status": "excellent" if self.overall_recovery > 80 else "good" if self.overall_recovery > 60 else "progressing"
        }
    
    async def _save_state(self, metrics: Dict[str, Any]):
        try:
            await self.cloud.firestore.save_agent_state(self.agent_id, self.simulation_id, {
                "metrics": metrics,
                "sectors": self.recovery_sectors,
                "lessons_learned_count": len(self.lessons_learned),
                "phase": self.current_phase.value,
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")


def create_recovery_coordinator_agent(cloud_services: CloudServices) -> RecoveryCoordinatorAgent:
    return RecoveryCoordinatorAgent(cloud_services)
