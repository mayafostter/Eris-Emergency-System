"""
ERIS Public Health Agent - Health crisis management and medical coordination
Manages medical resource distribution, health advisories, and epidemic tracking
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

class HealthRiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class MedicalResourceType(Enum):
    VACCINES = "vaccines"
    ANTIBIOTICS = "antibiotics"
    MEDICAL_SUPPLIES = "medical_supplies"
    PERSONAL_PROTECTIVE_EQUIPMENT = "ppe"
    BLOOD_PRODUCTS = "blood_products"
    OXYGEN = "oxygen"

@dataclass
class HealthAlert:
    alert_id: str
    alert_type: str
    risk_level: HealthRiskLevel
    affected_population: int
    health_advisory: str
    recommended_actions: List[str]

@dataclass
class MedicalResource:
    resource_type: MedicalResourceType
    total_available: int
    allocated: int
    distribution_points: List[str]
    expiration_concern: bool

class PublicHealthAgent:
    """
    Public Health Official Agent for ERIS disaster simulation.
    Manages health crisis response, medical resource distribution, and health surveillance.
    """
    
    def __init__(self, cloud_services: CloudServices):
        self.cloud = cloud_services
        self.agent_id = "public_health_official"
        self.agent_type = "public_health"
        
        # Medical resource inventory
        self.medical_resources = {
            MedicalResourceType.VACCINES: MedicalResource(
                MedicalResourceType.VACCINES, 5000, 0, ["central_clinic", "mobile_units"], False
            ),
            MedicalResourceType.ANTIBIOTICS: MedicalResource(
                MedicalResourceType.ANTIBIOTICS, 2500, 0, ["hospitals", "clinics"], False
            ),
            MedicalResourceType.MEDICAL_SUPPLIES: MedicalResource(
                MedicalResourceType.MEDICAL_SUPPLIES, 10000, 0, ["regional_warehouse", "hospitals"], False
            ),
            MedicalResourceType.PERSONAL_PROTECTIVE_EQUIPMENT: MedicalResource(
                MedicalResourceType.PERSONAL_PROTECTIVE_EQUIPMENT, 15000, 0, ["distribution_centers"], False
            ),
            MedicalResourceType.BLOOD_PRODUCTS: MedicalResource(
                MedicalResourceType.BLOOD_PRODUCTS, 800, 0, ["blood_bank", "hospitals"], True
            ),
            MedicalResourceType.OXYGEN: MedicalResource(
                MedicalResourceType.OXYGEN, 1200, 0, ["hospitals", "emergency_centers"], False
            )
        }
        
        # Health surveillance data
        self.active_health_alerts = []
        self.disease_surveillance = {
            "waterborne_diseases": 0,
            "respiratory_infections": 0,
            "vector_borne_diseases": 0,
            "food_poisoning": 0,
            "injuries": 0,
            "mental_health_concerns": 0
        }
        
        # Public health metrics
        self.population_at_risk = 0
        self.health_advisories_issued = 0
        self.vaccination_coverage = 0.0
        self.water_safety_status = "safe"
        self.air_quality_index = 50  # Good air quality baseline
        
        # Simulation state
        self.simulation_id = None
        self.current_phase = SimulationPhase.IMPACT
        self.disaster_type = None
        self.disaster_severity = 5
        self.location = "Phuket, Thailand"
        
        # Create the ADK agent
        self.adk_agent = self._create_health_agent()
        
    def _create_health_agent(self) -> LlmAgent:
        """Create the ADK agent for public health management"""
        
        def assess_health_risks(disaster_type: str, severity: int, affected_population: int, duration_hours: int = 72) -> Dict[str, Any]:
            """
            Assess health risks associated with the disaster.
            
            Args:
                disaster_type: Type of disaster affecting the region
                severity: Disaster severity (1-10)
                affected_population: Number of people affected
                duration_hours: Expected disaster duration
                
            Returns:
                Health risk assessment and recommended interventions
            """
            self.population_at_risk = affected_population
            
            # Disaster-specific health risks
            risk_profiles = {
                'earthquake': {
                    'injuries': HealthRiskLevel.HIGH,
                    'respiratory_infections': HealthRiskLevel.MODERATE,
                    'mental_health_concerns': HealthRiskLevel.HIGH,
                    'water_contamination': HealthRiskLevel.MODERATE
                },
                'flood': {
                    'waterborne_diseases': HealthRiskLevel.HIGH,
                    'vector_borne_diseases': HealthRiskLevel.HIGH,
                    'respiratory_infections': HealthRiskLevel.MODERATE,
                    'injuries': HealthRiskLevel.MODERATE
                },
                'tsunami': {
                    'injuries': HealthRiskLevel.CRITICAL,
                    'waterborne_diseases': HealthRiskLevel.HIGH,
                    'respiratory_infections': HealthRiskLevel.HIGH,
                    'mental_health_concerns': HealthRiskLevel.CRITICAL
                },
                'wildfire': {
                    'respiratory_infections': HealthRiskLevel.CRITICAL,
                    'burns': HealthRiskLevel.HIGH,
                    'mental_health_concerns': HealthRiskLevel.HIGH,
                    'heat_related_illness': HealthRiskLevel.HIGH
                },
                'hurricane': {
                    'injuries': HealthRiskLevel.HIGH,
                    'waterborne_diseases': HealthRiskLevel.MODERATE,
                    'vector_borne_diseases': HealthRiskLevel.MODERATE,
                    'mental_health_concerns': HealthRiskLevel.HIGH
                }
            }
            
            disaster_risks = risk_profiles.get(disaster_type, risk_profiles['earthquake'])
            
            # Adjust risk levels based on severity
            severity_multiplier = severity / 5.0  # Base severity 5
            risk_assessment = {}
            
            for risk_type, base_level in disaster_risks.items():
                # Convert enum to numeric, apply multiplier, convert back
                risk_values = {HealthRiskLevel.LOW: 1, HealthRiskLevel.MODERATE: 2, 
                             HealthRiskLevel.HIGH: 3, HealthRiskLevel.CRITICAL: 4}
                
                numeric_risk = risk_values[base_level] * severity_multiplier
                
                if numeric_risk >= 4:
                    adjusted_level = HealthRiskLevel.CRITICAL
                elif numeric_risk >= 3:
                    adjusted_level = HealthRiskLevel.HIGH
                elif numeric_risk >= 2:
                    adjusted_level = HealthRiskLevel.MODERATE
                else:
                    adjusted_level = HealthRiskLevel.LOW
                
                risk_assessment[risk_type] = adjusted_level.value
            
            # Calculate overall risk score
            total_risk_score = sum(risk_values.get(HealthRiskLevel(level), 1) for level in risk_assessment.values())
            max_possible_score = len(risk_assessment) * 4
            overall_risk_percentage = (total_risk_score / max_possible_score) * 100
            
            # Generate recommended interventions
            interventions = []
            if HealthRiskLevel.CRITICAL.value in risk_assessment.values():
                interventions.extend([
                    "Activate emergency medical response teams",
                    "Establish field hospitals if needed",
                    "Implement mass casualty protocols"
                ])
            
            if 'waterborne_diseases' in risk_assessment and risk_assessment['waterborne_diseases'] in ['high', 'critical']:
                interventions.extend([
                    "Deploy water purification systems",
                    "Issue boil-water advisories",
                    "Distribute bottled water"
                ])
            
            if 'respiratory_infections' in risk_assessment and risk_assessment['respiratory_infections'] in ['high', 'critical']:
                interventions.extend([
                    "Distribute N95 masks",
                    "Set up air filtration in shelters",
                    "Monitor air quality continuously"
                ])
            
            # Update air quality based on disaster type
            if disaster_type in ['wildfire', 'volcanic_eruption']:
                self.air_quality_index = min(300, 150 + (severity * 15))  # Unhealthy to hazardous
            elif disaster_type in ['earthquake', 'building_collapse']:
                self.air_quality_index = min(200, 80 + (severity * 10))   # Moderate to unhealthy
            
            result = {
                "overall_risk_percentage": round(overall_risk_percentage, 1),
                "risk_assessment": risk_assessment,
                "population_at_risk": affected_population,
                "high_priority_risks": [risk for risk, level in risk_assessment.items() 
                                      if level in ['high', 'critical']],
                "recommended_interventions": interventions,
                "estimated_health_impact": self._estimate_health_impact(disaster_type, severity, affected_population),
                "air_quality_index": self.air_quality_index,
                "monitoring_required": duration_hours > 48
            }
            
            logger.info(f"Health risks assessed for {disaster_type}: {overall_risk_percentage:.1f}% overall risk")
            return result
        
        def distribute_medical_resources(resource_type: str, quantity: int, distribution_locations: str, priority_level: str = "medium") -> Dict[str, Any]:
            """
            Distribute medical resources to affected areas.
            
            Args:
                resource_type: Type of medical resource to distribute
                quantity: Quantity to distribute
                distribution_locations: Comma-separated list of distribution points
                priority_level: Distribution priority (low, medium, high, critical)
                
            Returns:
                Distribution status and logistics information
            """
            locations = [loc.strip() for loc in distribution_locations.split(',')]
            
            try:
                resource_enum = MedicalResourceType(resource_type.lower())
                resource = self.medical_resources[resource_enum]
            except (ValueError, KeyError):
                return {
                    "status": "failed",
                    "error": f"Unknown resource type: {resource_type}",
                    "available_resources": [r.value for r in MedicalResourceType]
                }
            
            available_quantity = resource.total_available - resource.allocated
            
            if quantity > available_quantity:
                # Partial allocation based on priority
                allocated_quantity = available_quantity
                allocation_status = "partial"
            else:
                allocated_quantity = quantity
                allocation_status = "complete"
            
            # Update resource allocation
            resource.allocated += allocated_quantity
            
            # Calculate distribution per location
            quantity_per_location = allocated_quantity // len(locations) if locations else 0
            remainder = allocated_quantity % len(locations) if locations else 0
            
            distribution_plan = {}
            for i, location in enumerate(locations):
                location_quantity = quantity_per_location
                if i < remainder:  # Distribute remainder to first locations
                    location_quantity += 1
                
                distribution_plan[location] = {
                    "quantity_allocated": location_quantity,
                    "estimated_delivery_time": self._calculate_delivery_time(location, priority_level),
                    "distribution_method": self._get_distribution_method(resource_enum, priority_level)
                }
            
            # Update surveillance based on resource type
            if resource_enum == MedicalResourceType.VACCINES:
                self.vaccination_coverage = min(1.0, self.vaccination_coverage + (allocated_quantity / max(1, self.population_at_risk)))
            
            result = {
                "resource_type": resource_type,
                "requested_quantity": quantity,
                "allocated_quantity": allocated_quantity,
                "allocation_status": allocation_status,
                "distribution_plan": distribution_plan,
                "remaining_available": resource.total_available - resource.allocated,
                "priority_level": priority_level,
                "estimated_coverage": round((allocated_quantity / max(1, self.population_at_risk)) * 100, 1),
                "logistics_coordinator": "public_health_logistics",
                "tracking_enabled": True
            }
            
            logger.info(f"Medical resources distributed: {allocated_quantity} units of {resource_type} to {len(locations)} locations")
            return result
        
        def issue_health_advisory(advisory_type: str, target_population: str, severity_level: str, message_content: str = "") -> Dict[str, Any]:
            """
            Issue public health advisory to the population.
            
            Args:
                advisory_type: Type of advisory (water_safety, air_quality, disease_outbreak, evacuation_health)
                target_population: Target population group
                severity_level: Advisory severity (low, moderate, high, critical)
                message_content: Custom advisory message content
                
            Returns:
                Advisory issuance status and distribution information
            """
            try:
                risk_level = HealthRiskLevel(severity_level.lower())
            except ValueError:
                risk_level = HealthRiskLevel.MODERATE
            
            # Generate advisory content if not provided
            if not message_content:
                message_content = self._generate_advisory_content(advisory_type, risk_level)
            
            # Create health alert
            alert_id = f"health_alert_{len(self.active_health_alerts)+1}_{datetime.utcnow().strftime('%H%M%S')}"
            
            health_alert = HealthAlert(
                alert_id=alert_id,
                alert_type=advisory_type,
                risk_level=risk_level,
                affected_population=self._estimate_affected_population(target_population),
                health_advisory=message_content,
                recommended_actions=self._get_recommended_actions(advisory_type, risk_level)
            )
            
            self.active_health_alerts.append(health_alert)
            self.health_advisories_issued += 1
            
            # Determine distribution channels
            distribution_channels = self._get_distribution_channels(risk_level, target_population)
            
            # Calculate reach and effectiveness
            estimated_reach = self._calculate_advisory_reach(target_population, distribution_channels)
            
            result = {
                "alert_id": alert_id,
                "advisory_type": advisory_type,
                "severity_level": severity_level,
                "target_population": target_population,
                "message_content": message_content,
                "distribution_channels": distribution_channels,
                "estimated_reach_percentage": round(estimated_reach * 100, 1),
                "recommended_actions": health_alert.recommended_actions,
                "issuance_timestamp": datetime.utcnow().isoformat(),
                "follow_up_required": risk_level in [HealthRiskLevel.HIGH, HealthRiskLevel.CRITICAL],
                "multilingual_support": True,
                "accessibility_features": ["audio_announcements", "visual_displays", "mobile_alerts"]
            }
            
            # Update water safety status if water advisory
            if advisory_type == "water_safety":
                self.water_safety_status = "unsafe" if risk_level in [HealthRiskLevel.HIGH, HealthRiskLevel.CRITICAL] else "caution"
            
            logger.info(f"Health advisory issued: {advisory_type} - {severity_level} level to {target_population}")
            return result
        
        def monitor_disease_surveillance(surveillance_period_hours: int = 24, population_sample_size: int = 1000) -> Dict[str, Any]:
            """
            Monitor disease surveillance and outbreak detection.
            
            Args:
                surveillance_period_hours: Monitoring period duration
                population_sample_size: Sample size for surveillance
                
            Returns:
                Disease surveillance data and outbreak alerts
            """
            # Simulate disease surveillance based on disaster type and conditions
            disaster_multipliers = {
                'flood': {'waterborne_diseases': 3.0, 'vector_borne_diseases': 2.5, 'respiratory_infections': 1.5},
                'tsunami': {'waterborne_diseases': 4.0, 'injuries': 3.0, 'mental_health_concerns': 2.5},
                'earthquake': {'injuries': 2.5, 'respiratory_infections': 2.0, 'mental_health_concerns': 2.0},
                'wildfire': {'respiratory_infections': 4.0, 'mental_health_concerns': 2.0},
                'hurricane': {'waterborne_diseases': 2.0, 'vector_borne_diseases': 1.8, 'injuries': 2.0}
            }
            
            base_rates = {  # Per 1000 people per 24 hours
                'waterborne_diseases': 2,
                'respiratory_infections': 5,
                'vector_borne_diseases': 1,
                'food_poisoning': 3,
                'injuries': 8,
                'mental_health_concerns': 12
            }
            
            multipliers = disaster_multipliers.get(self.disaster_type, {})
            time_factor = surveillance_period_hours / 24.0
            population_factor = population_sample_size / 1000.0
            
            # Calculate disease incidence
            surveillance_data = {}
            outbreak_alerts = []
            
            for disease, base_rate in base_rates.items():
                multiplier = multipliers.get(disease, 1.0)
                severity_factor = (self.disaster_severity / 5.0)  # Normalize to base severity 5
                
                expected_cases = base_rate * multiplier * severity_factor * time_factor * population_factor
                
                # Add some randomness but keep it realistic
                import random
                actual_cases = max(0, int(expected_cases * random.uniform(0.7, 1.3)))
                
                surveillance_data[disease] = actual_cases
                self.disease_surveillance[disease] += actual_cases
                
                # Check for outbreak thresholds
                outbreak_threshold = base_rate * 2.5 * population_factor  # 2.5x normal rate
                if actual_cases > outbreak_threshold:
                    outbreak_alerts.append({
                        "disease_type": disease,
                        "cases_detected": actual_cases,
                        "threshold_exceeded": round((actual_cases / outbreak_threshold) * 100, 1),
                        "risk_level": "high" if actual_cases > outbreak_threshold * 1.5 else "moderate",
                        "recommended_response": self._get_outbreak_response(disease)
                    })
            
            # Calculate overall health system stress
            total_cases = sum(surveillance_data.values())
            health_system_capacity = population_sample_size * 0.1  # 10% capacity threshold
            system_stress = min(100, (total_cases / health_system_capacity) * 100)
            
            result = {
                "surveillance_period_hours": surveillance_period_hours,
                "population_monitored": population_sample_size,
                "disease_incidence": surveillance_data,
                "total_cases_detected": total_cases,
                "outbreak_alerts": outbreak_alerts,
                "health_system_stress_percentage": round(system_stress, 1),
                "vaccination_coverage_percentage": round(self.vaccination_coverage * 100, 1),
                "water_safety_status": self.water_safety_status,
                "air_quality_index": self.air_quality_index,
                "recommendations": self._generate_surveillance_recommendations(surveillance_data, outbreak_alerts),
                "next_surveillance_window": (datetime.utcnow() + timedelta(hours=surveillance_period_hours)).isoformat()
            }
            
            logger.info(f"Disease surveillance completed: {total_cases} total cases detected, {len(outbreak_alerts)} outbreak alerts")
            return result
        
        # Create public health ADK agent
        return ERISAgentFactory.create_eris_agent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            model="gemini-2.0-flash",
            custom_tools=[assess_health_risks, distribute_medical_resources, issue_health_advisory, monitor_disease_surveillance]
        )
    
    def _estimate_health_impact(self, disaster_type: str, severity: int, population: int) -> Dict[str, int]:
        """Estimate health impact based on disaster parameters"""
        
        # Impact rates per 1000 people by disaster type
        impact_rates = {
            'earthquake': {'injuries': 25, 'fatalities': 2, 'displaced': 150},
            'tsunami': {'injuries': 40, 'fatalities': 8, 'displaced': 300},
            'flood': {'injuries': 15, 'fatalities': 1, 'displaced': 200},
            'wildfire': {'injuries': 20, 'fatalities': 3, 'displaced': 250},
            'hurricane': {'injuries': 18, 'fatalities': 2, 'displaced': 180}
        }
        
        rates = impact_rates.get(disaster_type, impact_rates['earthquake'])
        severity_multiplier = severity / 5.0  # Normalize to severity 5
        
        return {
            'estimated_injuries': int((rates['injuries'] / 1000) * population * severity_multiplier),
            'estimated_fatalities': int((rates['fatalities'] / 1000) * population * severity_multiplier),
            'estimated_displaced': int((rates['displaced'] / 1000) * population * severity_multiplier)
        }
    
    def _calculate_delivery_time(self, location: str, priority: str) -> str:
        """Calculate estimated delivery time for medical resources"""
        base_times = {
            'critical': 2,  # 2 hours
            'high': 4,      # 4 hours
            'medium': 8,    # 8 hours
            'low': 24       # 24 hours
        }
        
        base_time = base_times.get(priority, 8)
        
        # Location-based adjustments
        if 'remote' in location.lower() or 'rural' in location.lower():
            base_time *= 1.5
        elif 'central' in location.lower() or 'hospital' in location.lower():
            base_time *= 0.8
        
        return f"{int(base_time)} hours"
    
    def _get_distribution_method(self, resource_type: MedicalResourceType, priority: str) -> str:
        """Determine distribution method based on resource and priority"""
        if priority == 'critical':
            return "helicopter_delivery"
        elif resource_type in [MedicalResourceType.VACCINES, MedicalResourceType.BLOOD_PRODUCTS]:
            return "refrigerated_transport"
        elif resource_type == MedicalResourceType.OXYGEN:
            return "specialized_medical_transport"
        else:
            return "standard_medical_logistics"
    
    def _generate_advisory_content(self, advisory_type: str, risk_level: HealthRiskLevel) -> str:
        """Generate appropriate advisory content"""
        
        content_templates = {
            'water_safety': {
                HealthRiskLevel.CRITICAL: "CRITICAL WATER ALERT: Do not use tap water for drinking, cooking, or brushing teeth. Use only bottled or properly boiled water.",
                HealthRiskLevel.HIGH: "Water contamination detected. Boil all water for 3 minutes before use or use bottled water.",
                HealthRiskLevel.MODERATE: "Water quality advisory: Consider boiling water as a precaution, especially for vulnerable populations.",
                HealthRiskLevel.LOW: "Water systems are being monitored. Continue normal usage but report any unusual taste, odor, or appearance."
            },
            'air_quality': {
                HealthRiskLevel.CRITICAL: "HAZARDOUS AIR QUALITY: Stay indoors, seal windows and doors, use air purifiers if available. Avoid all outdoor activities.",
                HealthRiskLevel.HIGH: "Unhealthy air quality. Limit outdoor activities, especially for sensitive groups. Wear N95 masks if outdoors.",
                HealthRiskLevel.MODERATE: "Air quality is poor. Sensitive individuals should limit outdoor activities and consider wearing masks.",
                HealthRiskLevel.LOW: "Air quality is acceptable but may be sensitive for some individuals with respiratory conditions."
            },
            'disease_outbreak': {
                HealthRiskLevel.CRITICAL: "Disease outbreak confirmed. Seek immediate medical attention for symptoms. Follow all quarantine directives.",
                HealthRiskLevel.HIGH: "Increased disease activity detected. Practice enhanced hygiene, avoid crowded areas, monitor for symptoms.",
                HealthRiskLevel.MODERATE: "Health officials are monitoring increased illness reports. Practice good hygiene and seek medical care if symptomatic.",
                HealthRiskLevel.LOW: "Minor increase in illness reports. Continue normal activities with enhanced attention to hygiene."
            },
            'evacuation_health': {
                HealthRiskLevel.CRITICAL: "Immediate health-related evacuation required. Bring medications, medical devices, and health records.",
                HealthRiskLevel.HIGH: "Health-related evacuation recommended for vulnerable populations. Ensure medical needs are met at shelters.",
                HealthRiskLevel.MODERATE: "Evacuation may be necessary. Prepare medical supplies and ensure medication availability.",
                HealthRiskLevel.LOW: "Monitor evacuation orders. Prepare emergency medical kit and important health documents."
            }
        }
        
        return content_templates.get(advisory_type, {}).get(risk_level, "Health advisory issued. Follow guidance from local health authorities.")
    
    def _get_recommended_actions(self, advisory_type: str, risk_level: HealthRiskLevel) -> List[str]:
        """Get recommended actions for health advisory"""
        
        action_map = {
            'water_safety': [
                "Use bottled or boiled water only",
                "Avoid ice made from tap water",
                "Use bottled water for brushing teeth",
                "Report water system damage to authorities"
            ],
            'air_quality': [
                "Stay indoors when possible",
                "Use air purifiers or clean air rooms",
                "Wear N95 masks outdoors",
                "Limit physical exertion outdoors"
            ],
            'disease_outbreak': [
                "Practice frequent handwashing",
                "Maintain social distancing",
                "Seek medical attention for symptoms",
                "Follow quarantine guidelines if exposed"
            ],
            'evacuation_health': [
                "Gather prescription medications",
                "Bring medical devices and supplies",
                "Carry health insurance cards and medical records",
                "Identify medical needs for family members"
            ]
        }
        
        base_actions = action_map.get(advisory_type, ["Follow local health authority guidance"])
        
        if risk_level in [HealthRiskLevel.HIGH, HealthRiskLevel.CRITICAL]:
            base_actions.append("Contact emergency services if needed")
        
        return base_actions
    
    def _estimate_affected_population(self, target_population: str) -> int:
        """Estimate affected population size"""
        
        population_estimates = {
            'general_public': self.population_at_risk or 175000,
            'elderly': int((self.population_at_risk or 175000) * 0.15),  # 15% elderly
            'children': int((self.population_at_risk or 175000) * 0.20),  # 20% children
            'pregnant_women': int((self.population_at_risk or 175000) * 0.02),  # 2% pregnant
            'immunocompromised': int((self.population_at_risk or 175000) * 0.05),  # 5% immunocompromised
            'residents': self.population_at_risk or 175000,
            'tourists': int((self.population_at_risk or 175000) * 0.25),  # 25% tourists in Phuket
            'healthcare_workers': int((self.population_at_risk or 175000) * 0.02)  # 2% healthcare workers
        }
        
        return population_estimates.get(target_population.lower(), self.population_at_risk or 175000)
    
    def _get_distribution_channels(self, risk_level: HealthRiskLevel, target_population: str) -> List[str]:
        """Determine appropriate distribution channels for advisory"""
        
        base_channels = ["emergency_broadcast", "mobile_alerts", "social_media", "local_radio"]
        
        if risk_level in [HealthRiskLevel.HIGH, HealthRiskLevel.CRITICAL]:
            base_channels.extend(["door_to_door", "emergency_sirens", "helicopter_announcements"])
        
        # Population-specific channels
        if 'elderly' in target_population:
            base_channels.extend(["local_radio", "community_centers", "healthcare_facilities"])
        elif 'tourists' in target_population:
            base_channels.extend(["hotels", "tourist_information", "embassy_notifications"])
        
        return list(set(base_channels))  # Remove duplicates
    
    def _calculate_advisory_reach(self, target_population: str, channels: List[str]) -> float:
        """Calculate estimated reach percentage for advisory"""
        
        channel_effectiveness = {
            'emergency_broadcast': 0.85,
            'mobile_alerts': 0.90,
            'social_media': 0.70,
            'local_radio': 0.60,
            'door_to_door': 0.95,
            'emergency_sirens': 0.80,
            'helicopter_announcements': 0.50,
            'community_centers': 0.40,
            'healthcare_facilities': 0.85,
            'hotels': 0.90,
            'tourist_information': 0.60,
            'embassy_notifications': 0.95
        }
        
        # Calculate combined reach (not simply additive due to overlap)
        total_reach = 0.0
        remaining_population = 1.0
        
        # Sort channels by effectiveness (most effective first)
        sorted_channels = sorted(channels, key=lambda x: channel_effectiveness.get(x, 0.5), reverse=True)
        
        for channel in sorted_channels:
            effectiveness = channel_effectiveness.get(channel, 0.5)
            channel_reach = effectiveness * remaining_population
            total_reach += channel_reach
            remaining_population *= (1 - effectiveness * 0.7)  # Reduce overlap factor
        
        return min(0.98, total_reach)  # Cap at 98% (never 100% reach)
    
    def _get_outbreak_response(self, disease_type: str) -> List[str]:
        """Get recommended response for disease outbreak"""
        
        response_map = {
            'waterborne_diseases': [
                "Implement water safety protocols",
                "Distribute water purification tablets",
                "Increase sanitation measures",
                "Monitor water sources"
            ],
            'respiratory_infections': [
                "Distribute masks and PPE",
                "Improve ventilation in shelters",
                "Implement social distancing",
                "Monitor air quality"
            ],
            'vector_borne_diseases': [
                "Implement vector control measures",
                "Distribute insect repellent",
                "Eliminate standing water",
                "Monitor vector populations"
            ],
            'food_poisoning': [
                "Inspect food distribution systems",
                "Implement food safety protocols",
                "Monitor food storage temperatures",
                "Distribute safe food supplies"
            ],
            'injuries': [
                "Enhance trauma care capacity",
                "Deploy mobile medical units",
                "Implement triage protocols",
                "Coordinate medical evacuations"
            ],
            'mental_health_concerns': [
                "Deploy mental health teams",
                "Establish counseling services",
                "Implement stress management programs",
                "Monitor vulnerable populations"
            ]
        }
        
        return response_map.get(disease_type, ["Implement general disease control measures"])
    
    def _generate_surveillance_recommendations(self, surveillance_data: Dict[str, int], outbreak_alerts: List[Dict]) -> List[str]:
        """Generate recommendations based on surveillance data"""
        
        recommendations = []
        
        # Check total case load
        total_cases = sum(surveillance_data.values())
        if total_cases > 50:  # High case load
            recommendations.append("Increase surveillance frequency to every 12 hours")
            recommendations.append("Consider expanding medical response capacity")
        
        # Disease-specific recommendations
        if surveillance_data.get('waterborne_diseases', 0) > 10:
            recommendations.append("Intensify water safety monitoring and treatment")
        
        if surveillance_data.get('respiratory_infections', 0) > 20:
            recommendations.append("Enhance air quality monitoring and respiratory protection")
        
        if surveillance_data.get('mental_health_concerns', 0) > 30:
            recommendations.append("Deploy additional mental health support resources")
        
        # Outbreak-specific recommendations
        if outbreak_alerts:
            recommendations.append("Activate outbreak response protocols")
            recommendations.append("Consider declaring public health emergency")
        
        # Vaccination recommendations
        if self.vaccination_coverage < 0.6:  # Less than 60% coverage
            recommendations.append("Intensify vaccination campaign efforts")
        
        return recommendations
    
    async def initialize_for_simulation(self, simulation_id: str, disaster_type: str, severity: int, location: str):
        """Initialize agent for a specific simulation"""
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.disaster_severity = severity
        self.location = location
        
        # Reset metrics and state
        self.population_at_risk = 0
        self.health_advisories_issued = 0
        self.vaccination_coverage = 0.0
        self.water_safety_status = "safe"
        self.air_quality_index = 50
        self.active_health_alerts = []
        
        # Reset disease surveillance
        for disease in self.disease_surveillance:
            self.disease_surveillance[disease] = 0
        
        # Reset resource allocations
        for resource in self.medical_resources.values():
            resource.allocated = 0
        
        logger.info(f"Public Health Agent initialized for {disaster_type} severity {severity} in {location}")
    
    async def process_phase(self, phase: SimulationPhase, simulation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process public health response for a specific simulation phase"""
        self.current_phase = phase
        
        # Extract relevant context
        self.population_at_risk = simulation_context.get('total_population', 175000)
        
        # Execute phase-specific public health logic
        phase_results = await self._process_phase_specific_logic(phase, simulation_context)
        
        # Generate comprehensive metrics
        metrics = await self._generate_health_metrics()
        
        # Save state to cloud services
        await self._save_health_state(metrics)
        
        # Log to BigQuery for analytics
        await self._log_health_event(phase, metrics)
        
        return {
            "agent_id": self.agent_id,
            "phase": phase.value,
            "health_metrics": metrics,
            "phase_actions": phase_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _process_phase_specific_logic(self, phase: SimulationPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute phase-specific public health logic"""
        
        if phase == SimulationPhase.IMPACT:
            return await self._process_impact_phase(context)
        elif phase == SimulationPhase.RESPONSE:
            return await self._process_response_phase(context)
        elif phase == SimulationPhase.RECOVERY:
            return await self._process_recovery_phase(context)
        
        return {}
    
    async def _process_impact_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process immediate health impact response"""
        
        affected_population = context.get('total_population', 175000)
        
        try:
            # Assess health risks using internal implementation
            disaster_risks = {
                'earthquake': {
                    'injuries': HealthRiskLevel.HIGH,
                    'respiratory_infections': HealthRiskLevel.MODERATE,
                    'mental_health_concerns': HealthRiskLevel.HIGH
                },
                'flood': {
                    'waterborne_diseases': HealthRiskLevel.HIGH,
                    'vector_borne_diseases': HealthRiskLevel.HIGH,
                    'respiratory_infections': HealthRiskLevel.MODERATE
                },
                'tsunami': {
                    'injuries': HealthRiskLevel.CRITICAL,
                    'waterborne_diseases': HealthRiskLevel.HIGH,
                    'mental_health_concerns': HealthRiskLevel.CRITICAL
                },
                'wildfire': {
                    'respiratory_infections': HealthRiskLevel.CRITICAL,
                    'mental_health_concerns': HealthRiskLevel.HIGH
                }
            }.get(self.disaster_type, {
                'injuries': HealthRiskLevel.HIGH,
                'respiratory_infections': HealthRiskLevel.MODERATE
            })
            
            # Adjust for severity
            severity_multiplier = self.disaster_severity / 5.0
            risk_assessment = {}
            for risk_type, base_level in disaster_risks.items():
                risk_values = {HealthRiskLevel.LOW: 1, HealthRiskLevel.MODERATE: 2, 
                             HealthRiskLevel.HIGH: 3, HealthRiskLevel.CRITICAL: 4}
                numeric_risk = risk_values[base_level] * severity_multiplier
                
                if numeric_risk >= 4:
                    adjusted_level = HealthRiskLevel.CRITICAL
                elif numeric_risk >= 3:
                    adjusted_level = HealthRiskLevel.HIGH
                elif numeric_risk >= 2:
                    adjusted_level = HealthRiskLevel.MODERATE
                else:
                    adjusted_level = HealthRiskLevel.LOW
                
                risk_assessment[risk_type] = adjusted_level.value
            
            # Calculate overall risk
            total_risk_score = sum(4 if level == 'critical' else 3 if level == 'high' else 2 if level == 'moderate' else 1 
                                 for level in risk_assessment.values())
            max_possible_score = len(risk_assessment) * 4
            overall_risk_percentage = (total_risk_score / max_possible_score) * 100
            
            health_assessment = {
                "overall_risk_percentage": round(overall_risk_percentage, 1),
                "risk_assessment": risk_assessment,
                "population_at_risk": affected_population,
                "high_priority_risks": [risk for risk, level in risk_assessment.items() 
                                      if level in ['high', 'critical']]
            }
            
            # Distribute critical medical resources
            critical_resources = [
                ("medical_supplies", 1000, "hospitals,emergency_centers"),
                ("oxygen", 200, "hospitals,mobile_units"),
                ("antibiotics", 500, "hospitals,clinics")
            ]
            
            resource_distributions = []
            for resource_type, quantity, locations in critical_resources:
                # Internal distribution logic
                try:
                    resource_enum = MedicalResourceType(resource_type.lower())
                    resource = self.medical_resources[resource_enum]
                except:
                    continue
                
                available = resource.total_available - resource.allocated
                allocated = min(quantity, available)
                resource.allocated += allocated
                
                resource_distributions.append({
                    "resource_type": resource_type,
                    "allocated_quantity": allocated,
                    "allocation_status": "complete" if allocated == quantity else "partial",
                    "distribution_locations": locations.split(',')
                })
            
            # Issue critical health advisories
            if self.disaster_type in ['wildfire', 'volcanic_eruption']:
                advisory_result = {
                    "advisory_type": "air_quality",
                    "severity_level": "critical",
                    "target_population": "general_public",
                    "estimated_reach_percentage": 85
                }
                self.health_advisories_issued += 1
                self.air_quality_index = 250  # Hazardous
                
            elif self.disaster_type in ['flood', 'tsunami']:
                advisory_result = {
                    "advisory_type": "water_safety",
                    "severity_level": "high",
                    "target_population": "general_public", 
                    "estimated_reach_percentage": 80
                }
                self.health_advisories_issued += 1
                self.water_safety_status = "unsafe"
                
            else:
                advisory_result = {
                    "advisory_type": "evacuation_health",
                    "severity_level": "high",
                    "target_population": "vulnerable_populations",
                    "estimated_reach_percentage": 75
                }
                self.health_advisories_issued += 1
            
        except Exception as e:
            logger.warning(f"Public health impact phase error: {e}")
            # Create fallback results
            health_assessment = {"overall_risk_percentage": 75, "population_at_risk": affected_population}
            resource_distributions = [{"allocation_status": "partial"}]
            advisory_result = {"advisory_type": "general_health", "estimated_reach_percentage": 70}
        
        return {
            "health_risk_assessment": health_assessment,
            "critical_resource_distribution": resource_distributions,
            "emergency_health_advisory": advisory_result,
            "phase_focus": "immediate_health_impact_mitigation"
        }
    
    async def _process_response_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process coordinated health response phase"""
        
        try:
            # Enhanced disease surveillance
            surveillance_data = {}
            outbreak_alerts = []
            
            # Calculate disease incidence based on disaster type
            disaster_multipliers = {
                'flood': {'waterborne_diseases': 3.0, 'vector_borne_diseases': 2.5},
                'tsunami': {'waterborne_diseases': 4.0, 'injuries': 3.0},
                'earthquake': {'injuries': 2.5, 'respiratory_infections': 2.0},
                'wildfire': {'respiratory_infections': 4.0}
            }.get(self.disaster_type, {'injuries': 2.0})
            
            base_rates = {
                'waterborne_diseases': 2,
                'respiratory_infections': 5,
                'vector_borne_diseases': 1,
                'injuries': 8,
                'mental_health_concerns': 12
            }
            
            total_cases = 0
            for disease, base_rate in base_rates.items():
                multiplier = disaster_multipliers.get(disease, 1.0)
                severity_factor = self.disaster_severity / 5.0
                cases = int(base_rate * multiplier * severity_factor * 2)  # 2-hour period
                surveillance_data[disease] = cases
                total_cases += cases
                self.disease_surveillance[disease] += cases
                
                # Check outbreak threshold
                if cases > base_rate * 2.5:  # 2.5x normal rate
                    outbreak_alerts.append({
                        "disease_type": disease,
                        "cases_detected": cases,
                        "risk_level": "high"
                    })
            
            surveillance_result = {
                "disease_incidence": surveillance_data,
                "total_cases_detected": total_cases,
                "outbreak_alerts": outbreak_alerts,
                "health_system_stress_percentage": min(100, (total_cases / 100) * 100)
            }
            
            # Continue resource distribution
            ongoing_distributions = []
            for resource_type in ["vaccines", "ppe", "blood_products"]:
                try:
                    resource_enum = MedicalResourceType(resource_type.lower())
                    resource = self.medical_resources[resource_enum]
                    
                    quantity = 300
                    available = resource.total_available - resource.allocated
                    allocated = min(quantity, available)
                    resource.allocated += allocated
                    
                    ongoing_distributions.append({
                        "resource_type": resource_type,
                        "allocated_quantity": allocated,
                        "allocation_status": "ongoing"
                    })
                except:
                    continue
            
            # Issue follow-up advisories
            if outbreak_alerts:
                advisory_result = {
                    "advisory_type": "disease_outbreak",
                    "severity_level": "high",
                    "target_population": "general_public",
                    "estimated_reach_percentage": 85
                }
                self.health_advisories_issued += 1
            else:
                advisory_result = {
                    "advisory_type": "health_monitoring",
                    "severity_level": "moderate",
                    "target_population": "at_risk_groups",
                    "estimated_reach_percentage": 70
                }
                self.health_advisories_issued += 1
            
        except Exception as e:
            logger.warning(f"Public health response phase error: {e}")
            surveillance_result = {"total_cases_detected": 50, "outbreak_alerts": []}
            ongoing_distributions = [{"allocation_status": "ongoing"}]
            advisory_result = {"advisory_type": "health_monitoring"}
        
        return {
            "enhanced_surveillance": surveillance_result,
            "ongoing_resource_distribution": ongoing_distributions,
            "health_advisory_updates": advisory_result,
            "phase_focus": "coordinated_health_response"
        }
    
    async def _process_recovery_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process health recovery coordination"""
        
        try:
            # Final surveillance assessment
            recovery_surveillance = {
                "total_cumulative_cases": sum(self.disease_surveillance.values()),
                "vaccination_coverage_achieved": min(85, self.vaccination_coverage * 100 + 30),
                "water_safety_restored": True if self.water_safety_status != "unsafe" else False,
                "air_quality_normalized": self.air_quality_index < 100
            }
            
            # Resource recovery assessment
            resource_status = {}
            for resource_type, resource in self.medical_resources.items():
                utilization = (resource.allocated / resource.total_available) * 100
                resource_status[resource_type.value] = {
                    "utilization_percentage": round(utilization, 1),
                    "needs_restocking": utilization > 70
                }
            
            # Health system recovery metrics
            health_recovery = {
                "health_system_recovery_percentage": 85,
                "advisories_issued_total": self.health_advisories_issued,
                "population_health_status": "recovering",
                "long_term_monitoring_required": sum(self.disease_surveillance.values()) > 100
            }
            
            # Recovery recommendations
            recovery_advisory = {
                "advisory_type": "recovery_health",
                "severity_level": "low",
                "target_population": "general_public",
                "message_focus": "health_system_restoration"
            }
            self.health_advisories_issued += 1
            
        except Exception as e:
            logger.warning(f"Public health recovery phase error: {e}")
            recovery_surveillance = {"total_cumulative_cases": 150}
            resource_status = {"vaccines": {"utilization_percentage": 60}}
            health_recovery = {"health_system_recovery_percentage": 80}
            recovery_advisory = {"advisory_type": "recovery_health"}
        
        return {
            "recovery_surveillance": recovery_surveillance,
            "resource_utilization_assessment": resource_status,
            "health_system_recovery": health_recovery,
            "recovery_health_advisory": recovery_advisory,
            "phase_focus": "health_system_restoration"
        }
    
    async def _generate_health_metrics(self) -> Dict[str, Any]:
        """Generate comprehensive public health metrics"""
        
        # Resource utilization
        resource_metrics = {}
        total_resources = sum(r.total_available for r in self.medical_resources.values())
        total_allocated = sum(r.allocated for r in self.medical_resources.values())
        
        for resource_type, resource in self.medical_resources.items():
            utilization = (resource.allocated / resource.total_available) * 100 if resource.total_available > 0 else 0
            resource_metrics[resource_type.value] = {
                "total_available": resource.total_available,
                "allocated": resource.allocated,
                "utilization_percentage": round(utilization, 1),
                "distribution_points": len(resource.distribution_points)
            }
        
        overall_resource_utilization = (total_allocated / total_resources) * 100 if total_resources > 0 else 0
        
        # Health surveillance metrics
        total_disease_cases = sum(self.disease_surveillance.values())
        
        return {
            "resource_metrics": {
                "overall_utilization_percentage": round(overall_resource_utilization, 1),
                "total_resources_available": total_resources,
                "total_allocated": total_allocated,
                "resource_breakdown": resource_metrics
            },
            "health_surveillance_metrics": {
                "total_disease_cases": total_disease_cases,
                "disease_breakdown": self.disease_surveillance.copy(),
                "active_health_alerts": len(self.active_health_alerts),
                "vaccination_coverage_percentage": round(self.vaccination_coverage * 100, 1),
                "water_safety_status": self.water_safety_status,
                "air_quality_index": self.air_quality_index
            },
            "response_metrics": {
                "health_advisories_issued": self.health_advisories_issued,
                "population_at_risk": self.population_at_risk,
                "health_system_status": "operational" if total_disease_cases < 200 else "strained",
                "public_health_effectiveness": min(100, 85 - (total_disease_cases / 10))
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _save_health_state(self, metrics: Dict[str, Any]):
        """Save public health state to Firestore"""
        try:
            state_data = {
                "agent_id": self.agent_id,
                "simulation_id": self.simulation_id,
                "medical_resources": {
                    resource_type.value: {
                        "total_available": resource.total_available,
                        "allocated": resource.allocated,
                        "distribution_points": resource.distribution_points
                    }
                    for resource_type, resource in self.medical_resources.items()
                },
                "health_surveillance": self.disease_surveillance.copy(),
                "active_alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "alert_type": alert.alert_type,
                        "risk_level": alert.risk_level.value,
                        "affected_population": alert.affected_population
                    }
                    for alert in self.active_health_alerts
                ],
                "metrics": metrics,
                "phase": self.current_phase.value,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.cloud.firestore.save_agent_state(self.agent_id, self.simulation_id, state_data)
            
        except Exception as e:
            logger.error(f"Failed to save public health state: {e}")
    
    async def _log_health_event(self, phase: SimulationPhase, metrics: Dict[str, Any]):
        """Log public health events to BigQuery"""
        try:
            event_data = {
                "event_type": "public_health_update",
                "agent_id": self.agent_id,
                "phase": phase.value,
                "resource_utilization": metrics["resource_metrics"]["overall_utilization_percentage"],
                "total_disease_cases": metrics["health_surveillance_metrics"]["total_disease_cases"],
                "health_advisories_issued": metrics["response_metrics"]["health_advisories_issued"],
                "vaccination_coverage": metrics["health_surveillance_metrics"]["vaccination_coverage_percentage"],
                "air_quality_index": metrics["health_surveillance_metrics"]["air_quality_index"],
                "water_safety_status": metrics["health_surveillance_metrics"]["water_safety_status"]
            }
            
            await self.cloud.bigquery.log_simulation_event(
                simulation_id=self.simulation_id,
                event_type="public_health_update",
                event_data=event_data,
                agent_id=self.agent_id,
                phase=phase.value
            )
            
        except Exception as e:
            logger.error(f"Failed to log public health event: {e}")


def create_public_health_agent(cloud_services: CloudServices) -> PublicHealthAgent:
    """Factory function to create a Public Health Agent"""
    return PublicHealthAgent(cloud_services)
