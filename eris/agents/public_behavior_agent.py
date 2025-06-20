"""
ERIS Public Behavior Agent - Models civilian behavior during disasters
Simulates panic, evacuation decisions, shelter use, and social influence patterns
"""

import asyncio
import logging
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Google ADK imports
from google.adk.agents import Agent as LlmAgent

# ERIS imports
from agents.base_agent import ERISAgentFactory
from services import CloudServices
from utils.time_utils import SimulationPhase

logger = logging.getLogger(__name__)

class BehaviorType(Enum):
    """Types of public behavior patterns"""
    EVACUATE = "evacuate"
    SHELTER_IN_PLACE = "shelter_in_place"
    SEEK_INFORMATION = "seek_information"
    PANIC_BUYING = "panic_buying"
    SOCIAL_GATHERING = "social_gathering"
    IGNORE_WARNINGS = "ignore_warnings"

@dataclass
class PopulationSegment:
    """Population segment with specific characteristics"""
    name: str
    size: int
    compliance_rate: float  # 0-1, willingness to follow official guidance
    mobility: float  # 0-1, ability to evacuate quickly
    information_access: float  # 0-1, access to official information
    social_influence: float  # 0-1, susceptibility to social media/rumors
    panic_threshold: float  # 0-1, threshold for panic behavior

@dataclass
class EvacuationMetrics:
    """Evacuation and movement metrics"""
    total_population: int
    evacuated_count: int
    shelter_in_place: int
    still_in_danger_zone: int
    evacuation_compliance: float
    average_evacuation_time: float  # hours
    transportation_bottlenecks: List[str]

class PublicBehaviorAgent:
    """
    Public Behavior Agent for ERIS disaster simulation.
    Models realistic civilian behavior patterns during disasters.
    """
    
    def __init__(self, cloud_services: CloudServices):
        self.cloud = cloud_services
        self.agent_id = "public_behavior_coordinator"
        self.agent_type = "public_behavior"
        
        # Population segments with realistic characteristics
        self.population_segments = [
            PopulationSegment("local_residents", 80000, 0.75, 0.8, 0.9, 0.4, 0.6),
            PopulationSegment("tourists", 25000, 0.6, 0.9, 0.5, 0.8, 0.4),
            PopulationSegment("elderly", 12000, 0.9, 0.3, 0.7, 0.2, 0.8),
            PopulationSegment("families_with_children", 35000, 0.85, 0.6, 0.8, 0.5, 0.7),
            PopulationSegment("business_travelers", 8000, 0.7, 0.9, 0.8, 0.6, 0.5),
            PopulationSegment("low_income", 15000, 0.6, 0.4, 0.6, 0.7, 0.8)
        ]
        
        # Simulation state
        self.simulation_id = None
        self.current_phase = SimulationPhase.IMPACT
        self.disaster_type = None
        self.disaster_severity = 5
        self.panic_index = 0.0
        self.social_media_influence = 0.0
        self.official_communication_reach = 0.8
        
        # Behavior tracking
        self.evacuation_metrics = EvacuationMetrics(
            total_population=sum(seg.size for seg in self.population_segments),
            evacuated_count=0,
            shelter_in_place=0,
            still_in_danger_zone=0,
            evacuation_compliance=0.0,
            average_evacuation_time=0.0,
            transportation_bottlenecks=[]
        )
        
        self.behavior_distribution = {behavior.value: 0 for behavior in BehaviorType}
        
        # Create the ADK agent with behavior-specific tools
        self.adk_agent = self._create_behavior_agent()
        
    def _create_behavior_agent(self) -> LlmAgent:
        """Create the ADK agent for public behavior modeling"""
        
        def assess_evacuation_behavior(disaster_type: str, severity: int, official_guidance: str) -> Dict[str, Any]:
            """
            Assess public evacuation behavior and compliance.
            
            Args:
                disaster_type: Type of disaster
                severity: Disaster severity (1-10)
                official_guidance: Type of official guidance issued
                
            Returns:
                Evacuation behavior assessment and metrics
            """
            
            # Base evacuation rates by disaster type
            base_evacuation_rates = {
                'tsunami': 0.85,
                'hurricane': 0.70,
                'wildfire': 0.75,
                'flood': 0.60,
                'earthquake': 0.45,
                'volcanic_eruption': 0.80,
                'pandemic': 0.30,
                'severe_storm': 0.50,
                'epidemic': 0.25,
                'landslide': 0.65
            }
            
            base_rate = base_evacuation_rates.get(disaster_type, 0.6)
            severity_factor = (severity / 10) * 0.3 + 0.7  # 0.7 to 1.0 multiplier
            
            # Calculate evacuation for each population segment
            segment_evacuations = {}
            total_evacuated = 0
            
            for segment in self.population_segments:
                # Factors affecting evacuation decision
                guidance_factor = 1.2 if official_guidance == "mandatory" else 1.0 if official_guidance == "recommended" else 0.8
                compliance_factor = segment.compliance_rate
                mobility_factor = segment.mobility
                panic_factor = 1 + (self.panic_index * 0.3)
                
                # Final evacuation rate for this segment
                segment_evac_rate = min(1.0, base_rate * severity_factor * guidance_factor * 
                                      compliance_factor * mobility_factor * panic_factor)
                
                evacuated = int(segment.size * segment_evac_rate)
                segment_evacuations[segment.name] = {
                    'total': segment.size,
                    'evacuated': evacuated,
                    'evacuation_rate': round(segment_evac_rate, 3),
                    'compliance_score': round(compliance_factor, 2)
                }
                total_evacuated += evacuated
            
            # Update metrics
            self.evacuation_metrics.evacuated_count = total_evacuated
            self.evacuation_metrics.still_in_danger_zone = self.evacuation_metrics.total_population - total_evacuated
            self.evacuation_metrics.evacuation_compliance = total_evacuated / self.evacuation_metrics.total_population
            
            # Calculate average evacuation time
            self.evacuation_metrics.average_evacuation_time = self._calculate_evacuation_time(disaster_type, severity)
            
            result = {
                "total_population": self.evacuation_metrics.total_population,
                "total_evacuated": total_evacuated,
                "evacuation_compliance_rate": round(self.evacuation_metrics.evacuation_compliance, 3),
                "average_evacuation_time_hours": self.evacuation_metrics.average_evacuation_time,
                "segment_breakdown": segment_evacuations,
                "still_in_danger_zone": self.evacuation_metrics.still_in_danger_zone,
                "evacuation_effectiveness": "high" if self.evacuation_metrics.evacuation_compliance > 0.8 else 
                                          "moderate" if self.evacuation_metrics.evacuation_compliance > 0.6 else "low"
            }
            
            logger.info(f"Evacuation assessment: {total_evacuated}/{self.evacuation_metrics.total_population} evacuated ({self.evacuation_metrics.evacuation_compliance:.1%})")
            return result
        
        def model_panic_behavior(trigger_events: List[str], social_media_activity: float, misinformation_level: float) -> Dict[str, Any]:
            """
            Model panic behavior patterns in the population.
            
            Args:
                trigger_events: Events that may trigger panic
                social_media_activity: Level of social media activity (0-1)
                misinformation_level: Level of misinformation spread (0-1)
                
            Returns:
                Panic behavior analysis and impact metrics
            """
            
            # Calculate panic triggers
            panic_triggers = {
                'infrastructure_failure': 0.4,
                'casualty_reports': 0.6,
                'official_communication_gap': 0.5,
                'social_media_rumors': 0.3,
                'supply_shortage_reports': 0.5,
                'evacuation_bottleneck': 0.7
            }
            
            # Base panic from triggers
            trigger_panic = sum(panic_triggers.get(trigger, 0.2) for trigger in trigger_events) / len(trigger_events) if trigger_events else 0
            
            # Social media amplification
            social_amplification = social_media_activity * misinformation_level * 0.4
            
            # Disaster severity contribution
            severity_contribution = (self.disaster_severity / 10) * 0.3
            
            # Calculate overall panic index
            raw_panic = min(1.0, trigger_panic + social_amplification + severity_contribution)
            
            # Apply population segment responses to panic
            segment_panic_responses = {}
            overall_panic_behaviors = {behavior.value: 0 for behavior in BehaviorType}
            
            for segment in self.population_segments:
                # Segment-specific panic response
                segment_panic_level = min(1.0, raw_panic * (2 - segment.panic_threshold))
                
                # Behavior distribution based on panic level and segment characteristics
                if segment_panic_level > 0.7:
                    # High panic - evacuation, panic buying, information seeking
                    behaviors = {
                        BehaviorType.EVACUATE.value: 0.4,
                        BehaviorType.PANIC_BUYING.value: 0.3,
                        BehaviorType.SEEK_INFORMATION.value: 0.2,
                        BehaviorType.SOCIAL_GATHERING.value: 0.1
                    }
                elif segment_panic_level > 0.4:
                    # Moderate panic - mixed behaviors
                    behaviors = {
                        BehaviorType.SEEK_INFORMATION.value: 0.35,
                        BehaviorType.SHELTER_IN_PLACE.value: 0.25,
                        BehaviorType.EVACUATE.value: 0.2,
                        BehaviorType.PANIC_BUYING.value: 0.15,
                        BehaviorType.IGNORE_WARNINGS.value: 0.05
                    }
                else:
                    # Low panic - mostly rational behaviors
                    behaviors = {
                        BehaviorType.SHELTER_IN_PLACE.value: 0.4,
                        BehaviorType.SEEK_INFORMATION.value: 0.3,
                        BehaviorType.EVACUATE.value: 0.15,
                        BehaviorType.IGNORE_WARNINGS.value: 0.1,
                        BehaviorType.SOCIAL_GATHERING.value: 0.05
                    }
                
                segment_panic_responses[segment.name] = {
                    'panic_level': round(segment_panic_level, 3),
                    'population': segment.size,
                    'behaviors': behaviors
                }
                
                # Aggregate to overall behavior distribution
                for behavior, percentage in behaviors.items():
                    overall_panic_behaviors[behavior] += int(segment.size * percentage)
            
            # Update global panic index
            self.panic_index = raw_panic
            self.behavior_distribution = overall_panic_behaviors
            
            result = {
                "overall_panic_index": round(raw_panic, 3),
                "panic_level": "critical" if raw_panic > 0.8 else "high" if raw_panic > 0.6 else 
                             "moderate" if raw_panic > 0.4 else "low",
                "trigger_contribution": round(trigger_panic, 3),
                "social_media_amplification": round(social_amplification, 3),
                "segment_responses": segment_panic_responses,
                "overall_behavior_distribution": overall_panic_behaviors,
                "population_in_panic": sum(int(segment.size * min(1.0, raw_panic * (2 - segment.panic_threshold))) 
                                         for segment in self.population_segments if min(1.0, raw_panic * (2 - segment.panic_threshold)) > 0.5)
            }
            
            logger.info(f"Panic modeling: index {raw_panic:.3f}, {result['population_in_panic']} people in panic state")
            return result
        
        def analyze_shelter_behavior(shelter_capacity: int, transportation_status: str, weather_conditions: str) -> Dict[str, Any]:
            """
            Analyze public shelter-seeking behavior.
            
            Args:
                shelter_capacity: Available shelter capacity
                transportation_status: Status of transportation systems
                weather_conditions: Current weather conditions
                
            Returns:
                Shelter behavior analysis and capacity metrics
            """
            
            # Calculate shelter demand based on population segments
            total_shelter_demand = 0
            segment_shelter_needs = {}
            
            for segment in self.population_segments:
                # Factors affecting shelter demand
                mobility_factor = 1.0 - segment.mobility  # Less mobile = more likely to need shelter
                evacuation_factor = 1.0 - (self.evacuation_metrics.evacuated_count / self.evacuation_metrics.total_population)
                
                # Weather impact
                weather_factor = {
                    'severe': 1.4,
                    'poor': 1.2,
                    'moderate': 1.0,
                    'good': 0.8
                }.get(weather_conditions, 1.0)
                
                # Transportation impact
                transport_factor = {
                    'disrupted': 1.3,
                    'limited': 1.1,
                    'normal': 1.0
                }.get(transportation_status, 1.0)
                
                # Calculate shelter need for this segment
                shelter_rate = min(0.8, mobility_factor * evacuation_factor * weather_factor * transport_factor * 0.3)
                shelter_demand = int(segment.size * shelter_rate)
                
                segment_shelter_needs[segment.name] = {
                    'population': segment.size,
                    'shelter_demand': shelter_demand,
                    'shelter_rate': round(shelter_rate, 3)
                }
                total_shelter_demand += shelter_demand
            
            # Calculate shelter metrics
            shelter_utilization = min(1.0, total_shelter_demand / shelter_capacity) if shelter_capacity > 0 else 1.0
            unmet_shelter_need = max(0, total_shelter_demand - shelter_capacity)
            
            # Update evacuation metrics
            self.evacuation_metrics.shelter_in_place = min(total_shelter_demand, shelter_capacity)
            
            result = {
                "total_shelter_demand": total_shelter_demand,
                "shelter_capacity": shelter_capacity,
                "shelter_utilization_rate": round(shelter_utilization, 3),
                "unmet_shelter_need": unmet_shelter_need,
                "people_in_shelters": min(total_shelter_demand, shelter_capacity),
                "shelter_overflow": unmet_shelter_need > 0,
                "segment_breakdown": segment_shelter_needs,
                "shelter_adequacy": "adequate" if unmet_shelter_need == 0 else "insufficient",
                "capacity_status": f"{round(shelter_utilization * 100, 1)}% utilized"
            }
            
            logger.info(f"Shelter analysis: {total_shelter_demand} demand vs {shelter_capacity} capacity ({shelter_utilization:.1%} utilization)")
            return result
        
        def track_information_seeking(official_channels_reach: float, social_media_reach: float, rumor_spread_rate: float) -> Dict[str, Any]:
            """
            Track information-seeking behavior and influence.
            
            Args:
                official_channels_reach: Reach of official information (0-1)
                social_media_reach: Reach of social media information (0-1)
                rumor_spread_rate: Rate of rumor/misinformation spread (0-1)
                
            Returns:
                Information behavior analysis and influence metrics
            """
            
            # Calculate information access by segment
            segment_info_behavior = {}
            total_seeking_official = 0
            total_seeking_social = 0
            total_exposed_rumors = 0
            
            for segment in self.population_segments:
                # Segment-specific information behavior
                official_access = segment.information_access * official_channels_reach
                social_access = segment.social_influence * social_media_reach
                rumor_exposure = segment.social_influence * rumor_spread_rate
                
                # Calculate numbers
                seeking_official = int(segment.size * official_access * 0.8)  # 80% actively seek official info
                seeking_social = int(segment.size * social_access * 0.9)      # 90% check social media
                exposed_rumors = int(segment.size * rumor_exposure * 0.6)     # 60% exposed to rumors
                
                segment_info_behavior[segment.name] = {
                    'population': segment.size,
                    'seeking_official_info': seeking_official,
                    'seeking_social_media': seeking_social,
                    'exposed_to_rumors': exposed_rumors,
                    'information_access_score': round(segment.information_access, 2),
                    'social_influence_score': round(segment.social_influence, 2)
                }
                
                total_seeking_official += seeking_official
                total_seeking_social += seeking_social
                total_exposed_rumors += exposed_rumors
            
            # Calculate overall information metrics
            total_pop = sum(seg.size for seg in self.population_segments)
            official_info_penetration = total_seeking_official / total_pop
            social_media_penetration = total_seeking_social / total_pop
            rumor_penetration = total_exposed_rumors / total_pop
            
            # Information quality score
            info_quality = official_info_penetration / max(0.1, official_info_penetration + rumor_penetration)
            
            # Update social media influence
            self.social_media_influence = social_media_reach
            self.official_communication_reach = official_channels_reach
            
            result = {
                "total_seeking_official_info": total_seeking_official,
                "total_seeking_social_media": total_seeking_social,
                "total_exposed_to_rumors": total_exposed_rumors,
                "official_info_penetration": round(official_info_penetration, 3),
                "social_media_penetration": round(social_media_penetration, 3),
                "rumor_penetration": round(rumor_penetration, 3),
                "information_quality_score": round(info_quality, 3),
                "segment_breakdown": segment_info_behavior,
                "dominant_info_source": "official" if official_info_penetration > social_media_penetration else "social_media",
                "misinformation_risk": "high" if rumor_penetration > 0.4 else "moderate" if rumor_penetration > 0.2 else "low"
            }
            
            logger.info(f"Information tracking: {official_info_penetration:.1%} official, {social_media_penetration:.1%} social, {rumor_penetration:.1%} rumors")
            return result
        
        # Create behavior-specific ADK agent
        return ERISAgentFactory.create_eris_agent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            model="gemini-2.0-flash",
            custom_tools=[assess_evacuation_behavior, model_panic_behavior, analyze_shelter_behavior, track_information_seeking]
        )
    
    def _calculate_evacuation_time(self, disaster_type: str, severity: int) -> float:
        """Calculate average evacuation time based on disaster and severity"""
        
        # Base evacuation times by disaster type (hours)
        base_times = {
            'tsunami': 1.5,      # Very urgent
            'wildfire': 3.0,     # Fast-moving threat
            'hurricane': 8.0,    # Advance warning available
            'flood': 4.0,        # Moderate urgency
            'earthquake': 2.0,   # Immediate aftermath
            'volcanic_eruption': 6.0,  # Some advance warning
            'pandemic': 24.0     # Gradual evacuation
        }
        
        base_time = base_times.get(disaster_type, 4.0)
        
        # Severity affects evacuation speed (higher severity = more urgency = faster evacuation)
        severity_factor = 1.5 - (severity / 20)  # Range: 1.0 to 1.5
        
        # Transportation bottlenecks increase time
        bottleneck_factor = 1.0 + (len(self.evacuation_metrics.transportation_bottlenecks) * 0.2)
        
        return round(base_time * severity_factor * bottleneck_factor, 1)
    
    async def initialize_for_simulation(self, simulation_id: str, disaster_type: str, severity: int, location: str):
        """Initialize agent for a specific simulation"""
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.disaster_severity = severity
        
        # Adjust population segments based on location
        await self._adjust_population_for_location(location)
        
        # Reset metrics
        self.evacuation_metrics = EvacuationMetrics(
            total_population=sum(seg.size for seg in self.population_segments),
            evacuated_count=0,
            shelter_in_place=0,
            still_in_danger_zone=0,
            evacuation_compliance=0.0,
            average_evacuation_time=0.0,
            transportation_bottlenecks=[]
        )
        
        logger.info(f"Public Behavior Agent initialized for simulation {simulation_id} in {location}")
    
    async def _adjust_population_for_location(self, location: str):
        """Adjust population segments based on location characteristics"""
        
        # Location-based adjustments
        if 'phuket' in location.lower():
            # Tourist destination - increase tourist population
            for segment in self.population_segments:
                if segment.name == 'tourists':
                    segment.size = int(segment.size * 1.5)
                elif segment.name == 'business_travelers':
                    segment.size = int(segment.size * 1.3)
        
        elif 'bangkok' in location.lower():
            # Major city - increase all populations
            for segment in self.population_segments:
                segment.size = int(segment.size * 2.0)
        
        elif any(word in location.lower() for word in ['rural', 'village', 'remote']):
            # Rural area - different population mix
            for segment in self.population_segments:
                if segment.name == 'tourists':
                    segment.size = int(segment.size * 0.3)
                elif segment.name == 'elderly':
                    segment.size = int(segment.size * 1.4)
                elif segment.name == 'business_travelers':
                    segment.size = int(segment.size * 0.2)
        
        logger.info(f"Population adjusted for {location}: {sum(seg.size for seg in self.population_segments)} total")
    
    async def process_phase(self, phase: SimulationPhase, simulation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process public behavior for a specific simulation phase"""
        self.current_phase = phase
        
        # Extract context variables
        infrastructure_status = simulation_context.get('infrastructure_status', 'operational')
        communication_reach = simulation_context.get('official_communication_reach', 0.8)
        social_media_activity = simulation_context.get('social_media_activity', 0.6)
        
        # Phase-specific processing
        phase_results = await self._process_phase_specific_behavior(phase, simulation_context)
        
        # Generate comprehensive behavior metrics
        metrics = await self._generate_behavior_metrics()
        
        # Save state to cloud services
        await self._save_behavior_state(metrics)
        
        # Log to BigQuery for analytics
        await self._log_behavior_event(phase, metrics)
        
        return {
            "agent_id": self.agent_id,
            "phase": phase.value,
            "behavior_metrics": metrics,
            "phase_actions": phase_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _process_phase_specific_behavior(self, phase: SimulationPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute phase-specific behavior modeling"""
        
        if phase == SimulationPhase.IMPACT:
            return await self._process_impact_behavior(context)
        elif phase == SimulationPhase.RESPONSE:
            return await self._process_response_behavior(context)
        elif phase == SimulationPhase.RECOVERY:
            return await self._process_recovery_behavior(context)
        
        return {}
    
    async def _process_impact_behavior(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process immediate impact phase public behavior"""
        
        # FIXED: Replace ADK tool calls with direct implementations
        try:
            # Initial evacuation assessment
            official_guidance = "recommended" if self.disaster_severity < 7 else "mandatory"
            
            # Base evacuation rates by disaster type
            base_evacuation_rates = {
                'tsunami': 0.85, 'hurricane': 0.70, 'wildfire': 0.75, 'flood': 0.60,
                'earthquake': 0.45, 'volcanic_eruption': 0.80, 'pandemic': 0.30,
                'severe_storm': 0.50, 'epidemic': 0.25, 'landslide': 0.65
            }
            
            base_rate = base_evacuation_rates.get(self.disaster_type, 0.6)
            severity_factor = (self.disaster_severity / 10) * 0.3 + 0.7  # 0.7 to 1.0 multiplier
            guidance_factor = 1.2 if official_guidance == "mandatory" else 1.0
            
            # Calculate total evacuation
            avg_compliance = sum(seg.compliance_rate for seg in self.population_segments) / len(self.population_segments)
            avg_mobility = sum(seg.mobility for seg in self.population_segments) / len(self.population_segments)
            
            total_evacuation_rate = min(1.0, base_rate * severity_factor * guidance_factor * avg_compliance * avg_mobility)
            total_evacuated = int(self.evacuation_metrics.total_population * total_evacuation_rate)
            
            # Update evacuation metrics
            self.evacuation_metrics.evacuated_count = total_evacuated
            self.evacuation_metrics.still_in_danger_zone = self.evacuation_metrics.total_population - total_evacuated
            self.evacuation_metrics.evacuation_compliance = total_evacuation_rate
            self.evacuation_metrics.average_evacuation_time = self._calculate_evacuation_time(self.disaster_type, self.disaster_severity)
            
            evacuation_result = {
                "total_population": self.evacuation_metrics.total_population,
                "total_evacuated": total_evacuated,
                "evacuation_compliance_rate": round(total_evacuation_rate, 3),
                "average_evacuation_time_hours": self.evacuation_metrics.average_evacuation_time,
                "still_in_danger_zone": self.evacuation_metrics.still_in_danger_zone,
                "evacuation_effectiveness": "high" if total_evacuation_rate > 0.8 else "moderate" if total_evacuation_rate > 0.6 else "low"
            }
            
            # Panic behavior modeling
            trigger_events = [
                'infrastructure_failure' if context.get('infrastructure_damage', 0) > 50 else None,
                'casualty_reports' if self.disaster_severity > 6 else None,
                'social_media_rumors'
            ]
            trigger_events = [event for event in trigger_events if event is not None]
            
            # Calculate panic index
            panic_triggers = {
                'infrastructure_failure': 0.4, 'casualty_reports': 0.6, 'official_communication_gap': 0.5,
                'social_media_rumors': 0.3, 'supply_shortage_reports': 0.5, 'evacuation_bottleneck': 0.7
            }
            
            trigger_panic = sum(panic_triggers.get(trigger, 0.2) for trigger in trigger_events) / len(trigger_events) if trigger_events else 0
            social_amplification = 0.8 * 0.3 * 0.4  # High social media, moderate misinformation
            severity_contribution = (self.disaster_severity / 10) * 0.3
            
            raw_panic = min(1.0, trigger_panic + social_amplification + severity_contribution)
            self.panic_index = raw_panic
            
            # Calculate population in panic (using average panic threshold)
            avg_panic_threshold = sum(seg.panic_threshold for seg in self.population_segments) / len(self.population_segments)
            population_in_panic = int(self.evacuation_metrics.total_population * min(1.0, raw_panic * (2 - avg_panic_threshold)))
            
            panic_result = {
                "overall_panic_index": round(raw_panic, 3),
                "panic_level": "critical" if raw_panic > 0.8 else "high" if raw_panic > 0.6 else "moderate" if raw_panic > 0.4 else "low",
                "trigger_contribution": round(trigger_panic, 3),
                "social_media_amplification": round(social_amplification, 3),
                "population_in_panic": population_in_panic
            }
            
            # Shelter behavior
            shelter_capacity = context.get('shelter_capacity', 15000)
            transportation_status = 'disrupted' if context.get('infrastructure_damage', 0) > 30 else 'normal'
            weather_conditions = context.get('weather_conditions', 'moderate')
            
            # Calculate shelter demand
            avg_mobility = sum(seg.mobility for seg in self.population_segments) / len(self.population_segments)
            mobility_factor = 1.0 - avg_mobility  # Less mobile = more likely to need shelter
            evacuation_factor = 1.0 - (total_evacuated / self.evacuation_metrics.total_population)
            
            weather_factor = {'severe': 1.4, 'poor': 1.2, 'moderate': 1.0, 'good': 0.8}.get(weather_conditions, 1.0)
            transport_factor = {'disrupted': 1.3, 'limited': 1.1, 'normal': 1.0}.get(transportation_status, 1.0)
            
            shelter_rate = min(0.8, mobility_factor * evacuation_factor * weather_factor * transport_factor * 0.3)
            total_shelter_demand = int(self.evacuation_metrics.total_population * shelter_rate)
            
            shelter_utilization = min(1.0, total_shelter_demand / shelter_capacity) if shelter_capacity > 0 else 1.0
            unmet_shelter_need = max(0, total_shelter_demand - shelter_capacity)
            
            # Update evacuation metrics
            self.evacuation_metrics.shelter_in_place = min(total_shelter_demand, shelter_capacity)
            
            shelter_result = {
                "total_shelter_demand": total_shelter_demand,
                "shelter_capacity": shelter_capacity,
                "shelter_utilization_rate": round(shelter_utilization, 3),
                "unmet_shelter_need": unmet_shelter_need,
                "people_in_shelters": min(total_shelter_demand, shelter_capacity),
                "shelter_overflow": unmet_shelter_need > 0,
                "shelter_adequacy": "adequate" if unmet_shelter_need == 0 else "insufficient",
                "capacity_status": f"{round(shelter_utilization * 100, 1)}% utilized"
            }
            
            # Information seeking
            official_reach = context.get('official_communication_reach', 0.7)
            social_reach = 0.9  # High social media activity during disasters
            rumor_rate = 0.4 if self.panic_index > 0.5 else 0.2
            
            # Calculate information behavior
            avg_info_access = sum(seg.information_access for seg in self.population_segments) / len(self.population_segments)
            avg_social_influence = sum(seg.social_influence for seg in self.population_segments) / len(self.population_segments)
            
            total_seeking_official = int(self.evacuation_metrics.total_population * avg_info_access * official_reach * 0.8)
            total_seeking_social = int(self.evacuation_metrics.total_population * avg_social_influence * social_reach * 0.9)
            total_exposed_rumors = int(self.evacuation_metrics.total_population * avg_social_influence * rumor_rate * 0.6)
            
            official_info_penetration = total_seeking_official / self.evacuation_metrics.total_population
            social_media_penetration = total_seeking_social / self.evacuation_metrics.total_population
            rumor_penetration = total_exposed_rumors / self.evacuation_metrics.total_population
            
            # Update social media influence
            self.social_media_influence = social_reach
            self.official_communication_reach = official_reach
            
            info_result = {
                "total_seeking_official_info": total_seeking_official,
                "total_seeking_social_media": total_seeking_social,
                "total_exposed_to_rumors": total_exposed_rumors,
                "official_info_penetration": round(official_info_penetration, 3),
                "social_media_penetration": round(social_media_penetration, 3),
                "rumor_penetration": round(rumor_penetration, 3),
                "dominant_info_source": "official" if official_info_penetration > social_media_penetration else "social_media",
                "misinformation_risk": "high" if rumor_penetration > 0.4 else "moderate" if rumor_penetration > 0.2 else "low"
            }
            
            # Identify transportation bottlenecks
            self.evacuation_metrics.transportation_bottlenecks = self._identify_transport_bottlenecks(context)
            
        except Exception as e:
            logger.warning(f"Behavior agent impact phase tool call error: {e}")
            # Create fallback results
            evacuation_result = {"evacuation_compliance_rate": 0.6, "total_evacuated": int(self.evacuation_metrics.total_population * 0.6)}
            panic_result = {"overall_panic_index": 0.4, "panic_level": "moderate"}
            shelter_result = {"shelter_adequacy": "adequate", "shelter_utilization_rate": 0.7}
            info_result = {"official_info_penetration": 0.7, "misinformation_risk": "moderate"}
        
        return {
            "evacuation_assessment": evacuation_result,
            "panic_modeling": panic_result,
            "shelter_analysis": shelter_result,
            "information_behavior": info_result,
            "transportation_bottlenecks": self.evacuation_metrics.transportation_bottlenecks,
            "phase_focus": "immediate_response_and_evacuation"
        }
    
    async def _process_response_behavior(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process coordinated response phase public behavior"""
        
        # FIXED: Replace ADK tool calls with direct implementations
        try:
            # Continued evacuation with improved coordination
            official_guidance = "mandatory"
            
            # Improved evacuation rates in response phase
            base_evacuation_rates = {
                'tsunami': 0.85, 'hurricane': 0.70, 'wildfire': 0.75, 'flood': 0.60,
                'earthquake': 0.45, 'volcanic_eruption': 0.80, 'pandemic': 0.30,
                'severe_storm': 0.50, 'epidemic': 0.25, 'landslide': 0.65
            }
            
            base_rate = base_evacuation_rates.get(self.disaster_type, 0.6)
            severity_factor = (self.disaster_severity / 10) * 0.3 + 0.7
            guidance_factor = 1.2  # Mandatory guidance
            coordination_improvement = 1.1  # 10% improvement in response phase
            
            avg_compliance = sum(seg.compliance_rate for seg in self.population_segments) / len(self.population_segments)
            avg_mobility = sum(seg.mobility for seg in self.population_segments) / len(self.population_segments)
            
            total_evacuation_rate = min(1.0, base_rate * severity_factor * guidance_factor * avg_compliance * avg_mobility * coordination_improvement)
            total_evacuated = int(self.evacuation_metrics.total_population * total_evacuation_rate)
            
            # Update evacuation metrics
            self.evacuation_metrics.evacuated_count = total_evacuated
            self.evacuation_metrics.still_in_danger_zone = self.evacuation_metrics.total_population - total_evacuated
            self.evacuation_metrics.evacuation_compliance = total_evacuation_rate
            
            evacuation_result = {
                "total_population": self.evacuation_metrics.total_population,
                "total_evacuated": total_evacuated,
                "evacuation_compliance_rate": round(total_evacuation_rate, 3),
                "evacuation_effectiveness": "high" if total_evacuation_rate > 0.8 else "moderate" if total_evacuation_rate > 0.6 else "low"
            }
            
            # Reduced panic due to better information
            trigger_events = ['official_communication_gap'] if context.get('communication_effectiveness', 0.8) < 0.6 else []
            
            # Lower panic in response phase
            trigger_panic = 0.2 if trigger_events else 0.0
            social_amplification = 0.7 * 0.2 * 0.4  # Reduced misinformation
            severity_contribution = (self.disaster_severity / 10) * 0.2  # Reduced severity impact
            
            raw_panic = min(1.0, trigger_panic + social_amplification + severity_contribution)
            self.panic_index = raw_panic
            
            avg_panic_threshold = sum(seg.panic_threshold for seg in self.population_segments) / len(self.population_segments)
            population_in_panic = int(self.evacuation_metrics.total_population * min(1.0, raw_panic * (2 - avg_panic_threshold)))
            
            panic_result = {
                "overall_panic_index": round(raw_panic, 3),
                "panic_level": "critical" if raw_panic > 0.8 else "high" if raw_panic > 0.6 else "moderate" if raw_panic > 0.4 else "low",
                "population_in_panic": population_in_panic
            }
            
            # Improved shelter coordination
            shelter_capacity = context.get('shelter_capacity', 15000) * 1.2  # Additional shelters opened
            transportation_status = 'limited' if context.get('infrastructure_repair_progress', 0) < 50 else 'normal'
            
            # Calculate shelter demand
            avg_mobility = sum(seg.mobility for seg in self.population_segments) / len(self.population_segments)
            mobility_factor = 1.0 - avg_mobility
            evacuation_factor = 1.0 - (total_evacuated / self.evacuation_metrics.total_population)
            
            weather_factor = 1.0  # Moderate weather assumed
            transport_factor = {'limited': 1.1, 'normal': 1.0}.get(transportation_status, 1.0)
            
            shelter_rate = min(0.8, mobility_factor * evacuation_factor * weather_factor * transport_factor * 0.3)
            total_shelter_demand = int(self.evacuation_metrics.total_population * shelter_rate)
            
            shelter_utilization = min(1.0, total_shelter_demand / shelter_capacity) if shelter_capacity > 0 else 1.0
            unmet_shelter_need = max(0, total_shelter_demand - int(shelter_capacity))
            
            self.evacuation_metrics.shelter_in_place = min(total_shelter_demand, int(shelter_capacity))
            
            shelter_result = {
                "total_shelter_demand": total_shelter_demand,
                "shelter_capacity": int(shelter_capacity),
                "shelter_utilization_rate": round(shelter_utilization, 3),
                "unmet_shelter_need": unmet_shelter_need,
                "people_in_shelters": min(total_shelter_demand, int(shelter_capacity)),
                "shelter_adequacy": "adequate" if unmet_shelter_need == 0 else "insufficient"
            }
            
            # Better official information reach
            official_reach = min(0.9, context.get('official_communication_reach', 0.8) + 0.1)
            social_reach = 0.8
            rumor_rate = 0.15  # Reduced rumors
            
            avg_info_access = sum(seg.information_access for seg in self.population_segments) / len(self.population_segments)
            avg_social_influence = sum(seg.social_influence for seg in self.population_segments) / len(self.population_segments)
            
            total_seeking_official = int(self.evacuation_metrics.total_population * avg_info_access * official_reach * 0.8)
            total_seeking_social = int(self.evacuation_metrics.total_population * avg_social_influence * social_reach * 0.9)
            total_exposed_rumors = int(self.evacuation_metrics.total_population * avg_social_influence * rumor_rate * 0.6)
            
            official_info_penetration = total_seeking_official / self.evacuation_metrics.total_population
            rumor_penetration = total_exposed_rumors / self.evacuation_metrics.total_population
            
            self.social_media_influence = social_reach
            self.official_communication_reach = official_reach
            
            info_result = {
                "total_seeking_official_info": total_seeking_official,
                "official_info_penetration": round(official_info_penetration, 3),
                "rumor_penetration": round(rumor_penetration, 3),
                "misinformation_risk": "low" if rumor_penetration < 0.2 else "moderate"
            }
            
        except Exception as e:
            logger.warning(f"Behavior agent response phase tool call error: {e}")
            # Create fallback results
            evacuation_result = {"evacuation_compliance_rate": 0.75, "total_evacuated": int(self.evacuation_metrics.total_population * 0.75)}
            panic_result = {"overall_panic_index": 0.3, "panic_level": "moderate"}
            shelter_result = {"shelter_adequacy": "adequate", "shelter_utilization_rate": 0.6}
            info_result = {"official_info_penetration": 0.85, "misinformation_risk": "low"}
        
        return {
            "improved_evacuation": evacuation_result,
            "reduced_panic": panic_result,
            "expanded_shelter": shelter_result,
            "enhanced_information": info_result,
            "phase_focus": "coordinated_response_and_stabilization"
        }
    
    async def _process_recovery_behavior(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process recovery phase public behavior"""
        
        # FIXED: Replace ADK tool calls with direct implementations
        try:
            # Return movement begins
            return_rate = 0.3 if context.get('infrastructure_repair_progress', 0) > 70 else 0.1
            returning_population = int(self.evacuation_metrics.evacuated_count * return_rate)
            
            self.evacuation_metrics.evacuated_count -= returning_population
            self.evacuation_metrics.still_in_danger_zone += returning_population
            
            # Minimal panic in recovery phase
            raw_panic = min(0.2, self.disaster_severity / 20)  # Very low panic
            self.panic_index = raw_panic
            
            avg_panic_threshold = sum(seg.panic_threshold for seg in self.population_segments) / len(self.population_segments)
            population_in_panic = int(self.evacuation_metrics.total_population * min(1.0, raw_panic * (2 - avg_panic_threshold)))
            
            panic_result = {
                "overall_panic_index": round(raw_panic, 3),
                "panic_level": "low",
                "population_in_panic": population_in_panic
            }
            
            # Shelter transition
            shelter_capacity = context.get('shelter_capacity', 15000)
            
            # Lower shelter demand in recovery
            avg_mobility = sum(seg.mobility for seg in self.population_segments) / len(self.population_segments)
            mobility_factor = 1.0 - avg_mobility
            evacuation_factor = 1.0 - (self.evacuation_metrics.evacuated_count / self.evacuation_metrics.total_population)
            
            shelter_rate = min(0.6, mobility_factor * evacuation_factor * 0.8 * 1.0 * 0.25)  # Reduced shelter need
            total_shelter_demand = int(self.evacuation_metrics.total_population * shelter_rate)
            
            shelter_utilization = min(1.0, total_shelter_demand / shelter_capacity) if shelter_capacity > 0 else 1.0
            self.evacuation_metrics.shelter_in_place = min(total_shelter_demand, shelter_capacity)
            
            shelter_result = {
                "total_shelter_demand": total_shelter_demand,
                "shelter_capacity": shelter_capacity,
                "shelter_utilization_rate": round(shelter_utilization, 3),
                "people_in_shelters": min(total_shelter_demand, shelter_capacity),
                "shelter_adequacy": "adequate"
            }
            
            # Focus on recovery information
            official_reach = 0.85
            social_reach = 0.6
            rumor_rate = 0.05  # Very low rumors
            
            avg_info_access = sum(seg.information_access for seg in self.population_segments) / len(self.population_segments)
            avg_social_influence = sum(seg.social_influence for seg in self.population_segments) / len(self.population_segments)
            
            total_seeking_official = int(self.evacuation_metrics.total_population * avg_info_access * official_reach * 0.8)
            total_seeking_social = int(self.evacuation_metrics.total_population * avg_social_influence * social_reach * 0.9)
            total_exposed_rumors = int(self.evacuation_metrics.total_population * avg_social_influence * rumor_rate * 0.6)
            
            official_info_penetration = total_seeking_official / self.evacuation_metrics.total_population
            rumor_penetration = total_exposed_rumors / self.evacuation_metrics.total_population
            
            self.social_media_influence = social_reach
            self.official_communication_reach = official_reach
            
            info_result = {
                "total_seeking_official_info": total_seeking_official,
                "official_info_penetration": round(official_info_penetration, 3),
                "rumor_penetration": round(rumor_penetration, 3),
                "misinformation_risk": "low"
            }
            
        except Exception as e:
            logger.warning(f"Behavior agent recovery phase tool call error: {e}")
            # Create fallback results
            returning_population = int(self.evacuation_metrics.evacuated_count * 0.2)
            panic_result = {"overall_panic_index": 0.1, "panic_level": "low"}
            shelter_result = {"shelter_adequacy": "adequate", "shelter_utilization_rate": 0.4}
            info_result = {"official_info_penetration": 0.85, "misinformation_risk": "low"}
        
        return {
            "return_movement": {
                "returning_population": returning_population,
                "still_evacuated": self.evacuation_metrics.evacuated_count,
                "return_rate": return_rate if 'return_rate' in locals() else 0.2
            },
            "minimal_panic": panic_result,
            "shelter_transition": shelter_result,
            "recovery_information": info_result,
            "phase_focus": "recovery_and_return_planning"
        }
    
    def _identify_transport_bottlenecks(self, context: Dict[str, Any]) -> List[str]:
        """Identify transportation bottlenecks affecting evacuation"""
        bottlenecks = []
        
        infrastructure_damage = context.get('infrastructure_damage', 0)
        
        if infrastructure_damage > 60:
            bottlenecks.extend(['major_roads_blocked', 'bridge_damage'])
        if infrastructure_damage > 40:
            bottlenecks.extend(['traffic_congestion', 'fuel_shortages'])
        if infrastructure_damage > 20:
            bottlenecks.append('limited_public_transport')
            
        return bottlenecks
    
    async def _generate_behavior_metrics(self) -> Dict[str, Any]:
        """Generate comprehensive public behavior metrics"""
        
        total_pop = sum(seg.size for seg in self.population_segments)
        
        return {
            "evacuation_metrics": {
                "total_population": total_pop,
                "evacuated_count": self.evacuation_metrics.evacuated_count,
                "evacuation_compliance_rate": round(self.evacuation_metrics.evacuation_compliance, 3),
                "still_in_danger_zone": self.evacuation_metrics.still_in_danger_zone,
                "average_evacuation_time": self.evacuation_metrics.average_evacuation_time,
                "shelter_occupancy": self.evacuation_metrics.shelter_in_place
            },
            "behavioral_metrics": {
                "panic_index": round(self.panic_index, 3),
                "behavior_distribution": self.behavior_distribution,
                "social_media_influence": round(self.social_media_influence, 3),
                "official_communication_reach": round(self.official_communication_reach, 3)
            },
            "population_segments": {
                seg.name: {
                    "size": seg.size,
                    "compliance_rate": seg.compliance_rate,
                    "mobility": seg.mobility
                } for seg in self.population_segments
            },
            "transportation_impact": {
                "bottlenecks": self.evacuation_metrics.transportation_bottlenecks,
                "bottleneck_count": len(self.evacuation_metrics.transportation_bottlenecks)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _save_behavior_state(self, metrics: Dict[str, Any]):
        """Save behavior state to Firestore"""
        try:
            state_data = {
                "agent_id": self.agent_id,
                "simulation_id": self.simulation_id,
                "evacuation_metrics": metrics["evacuation_metrics"],
                "behavioral_metrics": metrics["behavioral_metrics"],
                "population_segments": metrics["population_segments"],
                "phase": self.current_phase.value,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.cloud.firestore.save_agent_state(self.agent_id, self.simulation_id, state_data)
            
        except Exception as e:
            logger.error(f"Failed to save behavior state: {e}")
    
    async def _log_behavior_event(self, phase: SimulationPhase, metrics: Dict[str, Any]):
        """Log behavior events to BigQuery for analytics"""
        try:
            event_data = {
                "event_type": "public_behavior_update",
                "agent_id": self.agent_id,
                "phase": phase.value,
                "evacuation_compliance": metrics["evacuation_metrics"]["evacuation_compliance_rate"],
                "panic_index": metrics["behavioral_metrics"]["panic_index"],
                "evacuated_population": metrics["evacuation_metrics"]["evacuated_count"],
                "behavior_distribution": metrics["behavioral_metrics"]["behavior_distribution"]
            }
            
            await self.cloud.bigquery.log_simulation_event(
                simulation_id=self.simulation_id,
                event_type="public_behavior_update",
                event_data=event_data,
                agent_id=self.agent_id,
                phase=phase.value
            )
            
        except Exception as e:
            logger.error(f"Failed to log behavior event: {e}")


def create_public_behavior_agent(cloud_services: CloudServices) -> PublicBehaviorAgent:
    """Factory function to create a Public Behavior Agent"""
    return PublicBehaviorAgent(cloud_services)