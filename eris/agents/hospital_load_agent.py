"""
ERIS Hospital Load Agent - Models healthcare system capacity and resource management
Tracks ER capacity, ICU overflow, staff availability, and medical supply depletion
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Google ADK imports
from google.adk.agents import Agent as LlmAgent

# ERIS imports
from agents.base_agent import ERISAgentFactory
from services import CloudServices
from utils.time_utils import SimulationPhase

logger = logging.getLogger(__name__)

@dataclass
class HospitalCapacity:
    """Hospital capacity metrics"""
    total_beds: int
    occupied_beds: int
    icu_beds: int
    icu_occupied: int
    er_capacity: int
    er_current: int
    staff_available: int
    staff_total: int
    oxygen_supply: float  # percentage
    medical_supplies: float  # percentage
    blood_supply: float  # percentage

@dataclass
class PatientLoad:
    """Patient load metrics"""
    critical_patients: int
    serious_patients: int
    moderate_patients: int
    minor_patients: int
    psychiatric_patients: int
    disaster_related: int
    regular_admissions: int

class HospitalLoadAgent:
    """
    Hospital Load Agent for ERIS disaster simulation.
    Models realistic healthcare system response during disasters.
    """
    
    def __init__(self, cloud_services: CloudServices):
        self.cloud = cloud_services
        self.agent_id = "hospital_load_coordinator"
        self.agent_type = "hospital_load"
        
        # Initialize hospital capacity based on location
        self.base_capacity = HospitalCapacity(
            total_beds=850,  # Phuket hospital capacity
            occupied_beds=680,  # 80% baseline occupancy
            icu_beds=45,
            icu_occupied=32,  # 71% baseline ICU occupancy
            er_capacity=120,  # ER patients per day capacity
            er_current=45,  # baseline ER load
            staff_available=420,  # doctors, nurses, support
            staff_total=500,
            oxygen_supply=100.0,
            medical_supplies=100.0,
            blood_supply=100.0
        )
        
        # Current state
        self.current_capacity = self.base_capacity
        self.current_load = PatientLoad(
            critical_patients=8,
            serious_patients=24,
            moderate_patients=35,
            minor_patients=28,
            psychiatric_patients=3,
            disaster_related=0,
            regular_admissions=15
        )
        
        # Simulation state
        self.simulation_id = None
        self.current_phase = SimulationPhase.IMPACT
        self.disaster_type = None
        self.disaster_severity = 5
        self.panic_index = 0.0
        self.supply_chain_disrupted = False
        
        # Create the ADK agent with hospital-specific tools
        self.adk_agent = self._create_hospital_agent()
        
    def _create_hospital_agent(self) -> LlmAgent:
        """Create the ADK agent for hospital load management"""
        
        def assess_hospital_capacity(disaster_type: str, severity: int, phase: str) -> Dict[str, Any]:
            """
            Assess current hospital capacity and resource status.
            
            Args:
                disaster_type: Type of disaster affecting the region
                severity: Disaster severity (1-10)
                phase: Current simulation phase
                
            Returns:
                Current capacity assessment and projections
            """
            capacity_percentage = (self.current_capacity.occupied_beds / self.current_capacity.total_beds) * 100
            icu_percentage = (self.current_capacity.icu_occupied / self.current_capacity.icu_beds) * 100
            er_percentage = (self.current_capacity.er_current / self.current_capacity.er_capacity) * 100
            
            assessment = {
                "bed_capacity_percentage": round(capacity_percentage, 1),
                "icu_capacity_percentage": round(icu_percentage, 1),
                "er_load_percentage": round(er_percentage, 1),
                "staff_availability": round((self.current_capacity.staff_available / self.current_capacity.staff_total) * 100, 1),
                "oxygen_supply": self.current_capacity.oxygen_supply,
                "medical_supplies": self.current_capacity.medical_supplies,
                "blood_supply": self.current_capacity.blood_supply,
                "surge_capacity_available": max(0, self.current_capacity.total_beds - self.current_capacity.occupied_beds),
                "critical_shortages": self._identify_critical_shortages(),
                "projected_overflow_hours": self._calculate_overflow_projection(disaster_type, severity)
            }
            
            logger.info(f"Hospital capacity assessed: {capacity_percentage:.1f}% bed occupancy, {icu_percentage:.1f}% ICU")
            return assessment
        
        def update_patient_surge(surge_patients: int, patient_acuity_distribution: Dict[str, int]) -> Dict[str, str]:
            """
            Process incoming patient surge from disaster.
            
            Args:
                surge_patients: Number of additional patients
                patient_acuity_distribution: Distribution by severity level
                
            Returns:
                Processing status and capacity impact
            """
            # Update patient load
            self.current_load.disaster_related += surge_patients
            self.current_load.critical_patients += patient_acuity_distribution.get('critical', 0)
            self.current_load.serious_patients += patient_acuity_distribution.get('serious', 0)
            self.current_load.moderate_patients += patient_acuity_distribution.get('moderate', 0)
            self.current_load.minor_patients += patient_acuity_distribution.get('minor', 0)
            
            # Calculate bed allocation
            beds_needed = surge_patients
            available_beds = self.current_capacity.total_beds - self.current_capacity.occupied_beds
            
            if beds_needed <= available_beds:
                self.current_capacity.occupied_beds += beds_needed
                status = "accommodated"
            else:
                # Overflow situation
                self.current_capacity.occupied_beds = self.current_capacity.total_beds
                overflow = beds_needed - available_beds
                status = f"overflow_{overflow}_patients"
                logger.warning(f"Hospital overflow: {overflow} patients cannot be accommodated")
            
            # Update ER load
            er_surge = min(surge_patients, self.current_capacity.er_capacity - self.current_capacity.er_current)
            self.current_capacity.er_current += er_surge
            
            result = {
                "status": status,
                "patients_processed": surge_patients,
                "overflow_count": max(0, beds_needed - available_beds),
                "er_surge": er_surge,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Patient surge processed: {surge_patients} patients, status: {status}")
            return result
        
        def manage_medical_resources(resource_type: str, consumption_rate: float, resupply_available: bool) -> Dict[str, Any]:
            """
            Manage medical resource consumption and depletion.
            
            Args:
                resource_type: Type of resource (oxygen, supplies, blood)
                consumption_rate: Rate of consumption (multiplier of normal)
                resupply_available: Whether resupply is possible
                
            Returns:
                Resource status and alerts
            """
            current_level = getattr(self.current_capacity, f"{resource_type}_supply", 100.0)
            
            # Calculate consumption
            base_consumption_per_hour = {
                'oxygen': 2.5,  # 2.5% per hour normal consumption
                'medical_supplies': 1.8,
                'blood_supply': 1.2
            }.get(resource_type, 2.0)
            
            hourly_consumption = base_consumption_per_hour * consumption_rate
            new_level = max(0, current_level - hourly_consumption)
            
            # Apply resupply if available
            if resupply_available and not self.supply_chain_disrupted:
                resupply_amount = min(10.0, 100.0 - new_level)  # 10% resupply rate
                new_level = min(100.0, new_level + resupply_amount)
            
            # Update capacity
            setattr(self.current_capacity, f"{resource_type}_supply", new_level)
            
            # Generate alerts
            alerts = []
            if new_level < 25:
                alerts.append(f"CRITICAL: {resource_type} supply at {new_level:.1f}%")
            elif new_level < 50:
                alerts.append(f"WARNING: {resource_type} supply at {new_level:.1f}%")
            
            result = {
                "resource_type": resource_type,
                "current_level": round(new_level, 1),
                "consumption_rate": consumption_rate,
                "hourly_depletion": round(hourly_consumption, 2),
                "alerts": alerts,
                "estimated_depletion_hours": round(new_level / hourly_consumption, 1) if hourly_consumption > 0 else None,
                "resupply_status": "available" if resupply_available and not self.supply_chain_disrupted else "disrupted"
            }
            
            logger.info(f"Resource managed: {resource_type} at {new_level:.1f}% level")
            return result
        
        def coordinate_staff_response(phase: str, staff_callback_percentage: float) -> Dict[str, Any]:
            """
            Coordinate medical staff response and availability.
            
            Args:
                phase: Current disaster phase
                staff_callback_percentage: Percentage of off-duty staff called back
                
            Returns:
                Staff coordination status
            """
            off_duty_staff = self.current_capacity.staff_total - self.current_capacity.staff_available
            callback_staff = int(off_duty_staff * (staff_callback_percentage / 100))
            
            # Account for staff unable to report (transportation, personal emergencies)
            availability_factor = {
                'impact': 0.7,    # 30% may not be able to report immediately
                'response': 0.85,  # Better coordination in response phase
                'recovery': 0.9    # Most staff available in recovery
            }.get(phase, 0.8)
            
            actual_callback = int(callback_staff * availability_factor)
            self.current_capacity.staff_available = min(
                self.current_capacity.staff_total,
                self.current_capacity.staff_available + actual_callback
            )
            
            # Calculate staff-to-patient ratios
            total_patients = self.current_capacity.occupied_beds
            nurse_ratio = total_patients / max(1, self.current_capacity.staff_available * 0.6)  # 60% nurses
            doctor_ratio = total_patients / max(1, self.current_capacity.staff_available * 0.25)  # 25% doctors
            
            result = {
                "staff_called_back": actual_callback,
                "total_available": self.current_capacity.staff_available,
                "availability_percentage": round((self.current_capacity.staff_available / self.current_capacity.staff_total) * 100, 1),
                "nurse_to_patient_ratio": round(nurse_ratio, 1),
                "doctor_to_patient_ratio": round(doctor_ratio, 1),
                "staffing_adequacy": "adequate" if nurse_ratio < 8 and doctor_ratio < 15 else "strained",
                "phase": phase
            }
            
            logger.info(f"Staff coordination: {actual_callback} staff called back, {self.current_capacity.staff_available} total available")
            return result
        
        # Create hospital-specific ADK agent
        return ERISAgentFactory.create_eris_agent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            model="gemini-2.0-flash",
            custom_tools=[assess_hospital_capacity, update_patient_surge, manage_medical_resources, coordinate_staff_response]
        )
    
    def _identify_critical_shortages(self) -> List[str]:
        """Identify critical resource shortages"""
        shortages = []
        
        if self.current_capacity.oxygen_supply < 25:
            shortages.append("oxygen_critical")
        if self.current_capacity.medical_supplies < 25:
            shortages.append("medical_supplies_critical")
        if self.current_capacity.blood_supply < 25:
            shortages.append("blood_supply_critical")
        
        # Capacity shortages
        bed_utilization = (self.current_capacity.occupied_beds / self.current_capacity.total_beds) * 100
        if bed_utilization > 95:
            shortages.append("bed_capacity_critical")
        
        icu_utilization = (self.current_capacity.icu_occupied / self.current_capacity.icu_beds) * 100
        if icu_utilization > 90:
            shortages.append("icu_capacity_critical")
        
        return shortages
    
    def _calculate_overflow_projection(self, disaster_type: str, severity: int) -> Optional[float]:
        """Calculate projected hours until hospital overflow"""
        current_occupancy = self.current_capacity.occupied_beds / self.current_capacity.total_beds
        
        if current_occupancy >= 1.0:
            return 0.0  # Already at overflow
        
        # Estimate patient influx rate based on disaster
        disaster_influx_rates = {
            'earthquake': 25,  # patients per hour
            'tsunami': 40,
            'hurricane': 15,
            'flood': 12,
            'wildfire': 18,
            'pandemic': 8,
            'volcanic_eruption': 22,
            'severe_storm': 10,
            'epidemic': 15,
            'landslide': 20
        }
        
        base_rate = disaster_influx_rates.get(disaster_type, 15)
        severity_multiplier = (severity / 10) * 1.5 + 0.5  # 0.5x to 2.0x based on severity
        patient_rate = base_rate * severity_multiplier
        
        available_beds = self.current_capacity.total_beds - self.current_capacity.occupied_beds
        hours_to_overflow = available_beds / patient_rate if patient_rate > 0 else None
        
        return round(hours_to_overflow, 1) if hours_to_overflow else None
    
    async def initialize_for_simulation(self, simulation_id: str, disaster_type: str, severity: int, location: str):
        """Initialize agent for a specific simulation"""
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.disaster_severity = severity
        
        # Adjust baseline capacity based on location and disaster type
        await self._adjust_baseline_capacity(location, disaster_type)
        
        logger.info(f"Hospital Load Agent initialized for simulation {simulation_id}")
    
    async def _adjust_baseline_capacity(self, location: str, disaster_type: str):
        """Adjust hospital capacity based on location and expected disaster impact"""
        
        # Location-based capacity adjustments
        location_factors = {
            'phuket': 1.0,  # baseline
            'bangkok': 2.5,  # major city
            'rural': 0.4,   # rural areas
            'island': 0.6   # island locations
        }
        
        location_key = next((key for key in location_factors.keys() if key in location.lower()), 'phuket')
        location_factor = location_factors[location_key]
        
        # Adjust capacity
        self.current_capacity.total_beds = int(self.base_capacity.total_beds * location_factor)
        self.current_capacity.icu_beds = int(self.base_capacity.icu_beds * location_factor)
        self.current_capacity.staff_total = int(self.base_capacity.staff_total * location_factor)
        
        # Disaster-specific pre-positioning
        if disaster_type in ['hurricane', 'tsunami']:
            # Evacuate some patients, free up beds
            evacuation_rate = 0.15  # 15% of stable patients evacuated
            evacuated = int(self.current_capacity.occupied_beds * evacuation_rate)
            self.current_capacity.occupied_beds = max(0, self.current_capacity.occupied_beds - evacuated)
            
        logger.info(f"Hospital capacity adjusted for {location}: {self.current_capacity.total_beds} beds, {self.current_capacity.staff_total} staff")
    
    async def process_phase(self, phase: SimulationPhase, simulation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process hospital load for a specific simulation phase"""
        self.current_phase = phase
        
        # Update panic index from context
        self.panic_index = simulation_context.get('panic_index', 0.0)
        self.supply_chain_disrupted = simulation_context.get('supply_chain_disrupted', False)
        
        # Phase-specific processing
        phase_results = await self._process_phase_specific_logic(phase, simulation_context)
        
        # Generate comprehensive metrics
        metrics = await self._generate_hospital_metrics()
        
        # Save state to cloud services
        await self._save_hospital_state(metrics)
        
        # Log to BigQuery for analytics
        await self._log_hospital_event(phase, metrics)
        
        return {
            "agent_id": self.agent_id,
            "phase": phase.value,
            "hospital_metrics": metrics,
            "phase_actions": phase_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _process_phase_specific_logic(self, phase: SimulationPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute phase-specific hospital load logic"""
        
        if phase == SimulationPhase.IMPACT:
            return await self._process_impact_phase(context)
        elif phase == SimulationPhase.RESPONSE:
            return await self._process_response_phase(context)
        elif phase == SimulationPhase.RECOVERY:
            return await self._process_recovery_phase(context)
        
        return {}
    
    async def _process_impact_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process immediate impact on hospital systems"""
        
        # Calculate initial patient surge
        severity_factor = self.disaster_severity / 10
        panic_factor = 1 + (self.panic_index * 0.5)  # Panic increases patient visits
        
        # Disaster-specific surge patterns
        surge_multipliers = {
            'earthquake': 3.0,
            'tsunami': 4.5,
            'hurricane': 2.5,
            'flood': 2.0,
            'wildfire': 2.8,
            'pandemic': 1.5,
            'volcanic_eruption': 3.2,
            'severe_storm': 2.2,
            'epidemic': 1.8,
            'landslide': 3.5
        }
        
        base_surge = 50  # baseline surge patients
        surge_multiplier = surge_multipliers.get(self.disaster_type, 2.0)
        total_surge = int(base_surge * surge_multiplier * severity_factor * panic_factor)
        
        # Patient acuity distribution (more severe disasters = more critical patients)
        critical_pct = min(0.3, 0.1 + (severity_factor * 0.2))
        serious_pct = min(0.4, 0.2 + (severity_factor * 0.2))
        
        acuity_dist = {
            'critical': int(total_surge * critical_pct),
            'serious': int(total_surge * serious_pct),
            'moderate': int(total_surge * 0.3),
            'minor': int(total_surge * (1 - critical_pct - serious_pct - 0.3))
        }
        
        # FIXED: Replace ADK tool calls with direct implementations
        try:
            # Process surge through internal methods instead of ADK tools
            # Update patient load
            self.current_load.disaster_related += total_surge
            self.current_load.critical_patients += acuity_dist.get('critical', 0)
            self.current_load.serious_patients += acuity_dist.get('serious', 0)
            self.current_load.moderate_patients += acuity_dist.get('moderate', 0)
            self.current_load.minor_patients += acuity_dist.get('minor', 0)
            
            # Calculate bed allocation
            beds_needed = total_surge
            available_beds = self.current_capacity.total_beds - self.current_capacity.occupied_beds
            
            if beds_needed <= available_beds:
                self.current_capacity.occupied_beds += beds_needed
                status = "accommodated"
                overflow_count = 0
            else:
                # Overflow situation
                overflow_count = beds_needed - available_beds
                self.current_capacity.occupied_beds = self.current_capacity.total_beds
                status = f"overflow_{overflow_count}_patients"
                logger.warning(f"Hospital overflow: {overflow_count} patients cannot be accommodated")
            
            # Update ER load
            er_surge = min(total_surge, self.current_capacity.er_capacity - self.current_capacity.er_current)
            self.current_capacity.er_current += er_surge
            
            surge_result = {
                "status": status,
                "patients_processed": total_surge,
                "overflow_count": overflow_count,
                "er_surge": er_surge,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Staff response
            staff_callback = 75  # 75% callback rate in impact phase
            off_duty_staff = self.current_capacity.staff_total - self.current_capacity.staff_available
            callback_staff = int(off_duty_staff * (staff_callback / 100))
            availability_factor = 0.7  # 30% may not be able to report immediately in impact phase
            actual_callback = int(callback_staff * availability_factor)
            
            self.current_capacity.staff_available = min(
                self.current_capacity.staff_total,
                self.current_capacity.staff_available + actual_callback
            )
            
            staff_result = {
                "staff_called_back": actual_callback,
                "total_available": self.current_capacity.staff_available,
                "availability_percentage": round((self.current_capacity.staff_available / self.current_capacity.staff_total) * 100, 1),
                "staffing_adequacy": "strained" if self.current_capacity.staff_available < self.current_capacity.staff_total * 0.8 else "adequate",
                "phase": 'impact'
            }
            
            # Resource consumption increases
            consumption_multiplier = 1.5 + (severity_factor * 0.5)
            
            # Update oxygen supply
            base_oxygen_consumption = 2.5  # 2.5% per hour normal consumption
            hourly_oxygen_consumption = base_oxygen_consumption * consumption_multiplier
            new_oxygen_level = max(0, self.current_capacity.oxygen_supply - hourly_oxygen_consumption)
            if not self.supply_chain_disrupted:
                resupply_amount = min(10.0, 100.0 - new_oxygen_level)
                new_oxygen_level = min(100.0, new_oxygen_level + resupply_amount)
            self.current_capacity.oxygen_supply = new_oxygen_level
            
            oxygen_result = {
                "resource_type": 'oxygen',
                "current_level": round(new_oxygen_level, 1),
                "consumption_rate": consumption_multiplier,
                "hourly_depletion": round(hourly_oxygen_consumption, 2),
                "alerts": [f"WARNING: oxygen supply at {new_oxygen_level:.1f}%"] if new_oxygen_level < 50 else [],
                "resupply_status": "available" if not self.supply_chain_disrupted else "disrupted"
            }
            
            # Update medical supplies
            base_supplies_consumption = 1.8
            hourly_supplies_consumption = base_supplies_consumption * consumption_multiplier
            new_supplies_level = max(0, self.current_capacity.medical_supplies - hourly_supplies_consumption)
            if not self.supply_chain_disrupted:
                resupply_amount = min(10.0, 100.0 - new_supplies_level)
                new_supplies_level = min(100.0, new_supplies_level + resupply_amount)
            self.current_capacity.medical_supplies = new_supplies_level
            
            supplies_result = {
                "resource_type": 'medical_supplies',
                "current_level": round(new_supplies_level, 1),
                "consumption_rate": consumption_multiplier,
                "hourly_depletion": round(hourly_supplies_consumption, 2),
                "alerts": [f"WARNING: medical supplies at {new_supplies_level:.1f}%"] if new_supplies_level < 50 else [],
                "resupply_status": "available" if not self.supply_chain_disrupted else "disrupted"
            }
            
            # Update blood supply
            base_blood_consumption = 1.2
            hourly_blood_consumption = base_blood_consumption * consumption_multiplier * 1.2  # Blood consumption higher
            new_blood_level = max(0, self.current_capacity.blood_supply - hourly_blood_consumption)
            if not self.supply_chain_disrupted:
                resupply_amount = min(10.0, 100.0 - new_blood_level)
                new_blood_level = min(100.0, new_blood_level + resupply_amount)
            self.current_capacity.blood_supply = new_blood_level
            
            blood_result = {
                "resource_type": 'blood_supply',
                "current_level": round(new_blood_level, 1),
                "consumption_rate": consumption_multiplier * 1.2,
                "hourly_depletion": round(hourly_blood_consumption, 2),
                "alerts": [f"WARNING: blood supply at {new_blood_level:.1f}%"] if new_blood_level < 50 else [],
                "resupply_status": "available" if not self.supply_chain_disrupted else "disrupted"
            }
            
        except Exception as e:
            logger.warning(f"Hospital agent tool processing error: {e}")
            # Create fallback results
            surge_result = {"status": "processed", "patients_processed": total_surge}
            staff_result = {"availability_percentage": 75, "phase": 'impact'}
            oxygen_result = {"resource_type": 'oxygen', "current_level": 80}
            supplies_result = {"resource_type": 'medical_supplies', "current_level": 75}
            blood_result = {"resource_type": 'blood_supply', "current_level": 85}
        
        return {
            "patient_surge": surge_result,
            "staff_response": staff_result,
            "resource_management": {
                "oxygen": oxygen_result,
                "medical_supplies": supplies_result,
                "blood_supply": blood_result
            },
            "surge_patients": total_surge,
            "acuity_distribution": acuity_dist
        }
    
    async def _process_response_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process coordinated response phase hospital operations"""
        
        # Continued patient influx but at reduced rate
        ongoing_surge = max(5, int(20 * (self.disaster_severity / 10)))
        
        # FIXED: Replace ADK tool calls with direct implementations
        try:
            # Better staff coordination in response phase
            staff_callback = 90  # 90% callback rate in response phase
            off_duty_staff = self.current_capacity.staff_total - self.current_capacity.staff_available
            callback_staff = int(off_duty_staff * (staff_callback / 100))
            availability_factor = 0.85  # Better coordination in response phase
            actual_callback = int(callback_staff * availability_factor)
            
            self.current_capacity.staff_available = min(
                self.current_capacity.staff_total,
                self.current_capacity.staff_available + actual_callback
            )
            
            staff_result = {
                "staff_called_back": actual_callback,
                "total_available": self.current_capacity.staff_available,
                "availability_percentage": round((self.current_capacity.staff_available / self.current_capacity.staff_total) * 100, 1),
                "staffing_adequacy": "adequate" if self.current_capacity.staff_available > self.current_capacity.staff_total * 0.8 else "strained",
                "phase": 'response'
            }
            
            # Stabilized resource consumption
            consumption_multiplier = 1.2 + (self.disaster_severity / 20)
            
            # Update oxygen with stabilized consumption
            base_oxygen_consumption = 2.5
            hourly_oxygen_consumption = base_oxygen_consumption * consumption_multiplier
            new_oxygen_level = max(0, self.current_capacity.oxygen_supply - hourly_oxygen_consumption)
            resupply_amount = min(10.0, 100.0 - new_oxygen_level)
            new_oxygen_level = min(100.0, new_oxygen_level + resupply_amount)
            self.current_capacity.oxygen_supply = new_oxygen_level
            
            oxygen_result = {
                "resource_type": 'oxygen',
                "current_level": round(new_oxygen_level, 1),
                "consumption_rate": consumption_multiplier,
                "stabilization_status": "improving"
            }
            
            # Update medical supplies
            base_supplies_consumption = 1.8
            hourly_supplies_consumption = base_supplies_consumption * consumption_multiplier
            new_supplies_level = max(0, self.current_capacity.medical_supplies - hourly_supplies_consumption)
            resupply_amount = min(10.0, 100.0 - new_supplies_level)
            new_supplies_level = min(100.0, new_supplies_level + resupply_amount)
            self.current_capacity.medical_supplies = new_supplies_level
            
            supplies_result = {
                "resource_type": 'medical_supplies',
                "current_level": round(new_supplies_level, 1),
                "consumption_rate": consumption_multiplier,
                "stabilization_status": "improving"
            }
            
            # Process ongoing patients
            acuity_dist = {
                'critical': int(ongoing_surge * 0.15),
                'serious': int(ongoing_surge * 0.25),
                'moderate': int(ongoing_surge * 0.35),
                'minor': int(ongoing_surge * 0.25)
            }
            
            # Update patient load for ongoing surge
            self.current_load.disaster_related += ongoing_surge
            self.current_load.critical_patients += acuity_dist.get('critical', 0)
            self.current_load.serious_patients += acuity_dist.get('serious', 0)
            self.current_load.moderate_patients += acuity_dist.get('moderate', 0)
            self.current_load.minor_patients += acuity_dist.get('minor', 0)
            
            # Calculate bed allocation for ongoing surge
            beds_needed = ongoing_surge
            available_beds = self.current_capacity.total_beds - self.current_capacity.occupied_beds
            
            if beds_needed <= available_beds:
                self.current_capacity.occupied_beds += beds_needed
                status = "accommodated"
                overflow_count = 0
            else:
                overflow_count = beds_needed - available_beds
                self.current_capacity.occupied_beds = self.current_capacity.total_beds
                status = f"overflow_{overflow_count}_patients"
            
            surge_result = {
                "status": status,
                "patients_processed": ongoing_surge,
                "overflow_count": overflow_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Hospital agent response phase error: {e}")
            # Create fallback results
            surge_result = {"status": "processed", "patients_processed": ongoing_surge}
            staff_result = {"availability_percentage": 90, "phase": 'response'}
            oxygen_result = {"resource_type": 'oxygen', "current_level": 85}
            supplies_result = {"resource_type": 'medical_supplies', "current_level": 80}

        return {
            "ongoing_surge": surge_result,
            "staff_coordination": staff_result,
            "resource_stabilization": {
                "oxygen": oxygen_result,
                "medical_supplies": supplies_result
            },
            "phase_focus": "stabilization_and_coordination"
        }
    
    async def _process_recovery_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process recovery phase hospital operations"""
        
        # Reduced patient influx, focus on recovery
        recovery_patients = max(2, int(8 * (self.disaster_severity / 10)))
        
        # FIXED: Replace ADK tool calls with direct implementations
        try:
            # Full staff availability in recovery phase
            staff_callback = 95  # 95% callback rate in recovery phase
            off_duty_staff = self.current_capacity.staff_total - self.current_capacity.staff_available
            callback_staff = int(off_duty_staff * (staff_callback / 100))
            availability_factor = 0.9  # Most staff available in recovery
            actual_callback = int(callback_staff * availability_factor)
            
            self.current_capacity.staff_available = min(
                self.current_capacity.staff_total,
                self.current_capacity.staff_available + actual_callback
            )
            
            staff_result = {
                "staff_called_back": actual_callback,
                "total_available": self.current_capacity.staff_available,
                "availability_percentage": round((self.current_capacity.staff_available / self.current_capacity.staff_total) * 100, 1),
                "staffing_adequacy": "adequate",
                "phase": 'recovery'
            }
            
            # Normal resource consumption in recovery
            consumption_multiplier = 1.0
            
            # Update oxygen with normal consumption
            base_oxygen_consumption = 2.5
            hourly_oxygen_consumption = base_oxygen_consumption * consumption_multiplier
            new_oxygen_level = max(0, self.current_capacity.oxygen_supply - hourly_oxygen_consumption)
            resupply_amount = min(15.0, 100.0 - new_oxygen_level)  # Better resupply in recovery
            new_oxygen_level = min(100.0, new_oxygen_level + resupply_amount)
            self.current_capacity.oxygen_supply = new_oxygen_level
            
            oxygen_result = {
                "resource_type": 'oxygen',
                "current_level": round(new_oxygen_level, 1),
                "consumption_rate": consumption_multiplier,
                "recovery_status": "normalizing"
            }
            
            # Update medical supplies
            base_supplies_consumption = 1.8
            hourly_supplies_consumption = base_supplies_consumption * consumption_multiplier
            new_supplies_level = max(0, self.current_capacity.medical_supplies - hourly_supplies_consumption)
            resupply_amount = min(15.0, 100.0 - new_supplies_level)  # Better resupply in recovery
            new_supplies_level = min(100.0, new_supplies_level + resupply_amount)
            self.current_capacity.medical_supplies = new_supplies_level
            
            supplies_result = {
                "resource_type": 'medical_supplies',
                "current_level": round(new_supplies_level, 1),
                "consumption_rate": consumption_multiplier,
                "recovery_status": "normalizing"
            }
            
            # Process recovery patients
            acuity_dist = {
                'critical': int(recovery_patients * 0.1),
                'serious': int(recovery_patients * 0.2),
                'moderate': int(recovery_patients * 0.4),
                'minor': int(recovery_patients * 0.3)
            }
            
            # Update patient load for recovery phase
            self.current_load.disaster_related += recovery_patients
            self.current_load.critical_patients += acuity_dist.get('critical', 0)
            self.current_load.serious_patients += acuity_dist.get('serious', 0)
            self.current_load.moderate_patients += acuity_dist.get('moderate', 0)
            self.current_load.minor_patients += acuity_dist.get('minor', 0)
            
            # Calculate bed allocation
            beds_needed = recovery_patients
            available_beds = self.current_capacity.total_beds - self.current_capacity.occupied_beds
            
            if beds_needed <= available_beds:
                self.current_capacity.occupied_beds += beds_needed
                status = "accommodated"
                overflow_count = 0
            else:
                overflow_count = beds_needed - available_beds
                self.current_capacity.occupied_beds = self.current_capacity.total_beds
                status = f"overflow_{overflow_count}_patients"
            
            surge_result = {
                "status": status,
                "patients_processed": recovery_patients,
                "overflow_count": overflow_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Begin discharge planning for stable patients
            discharge_rate = 0.1  # 10% of stable patients discharged
            discharged = int(self.current_capacity.occupied_beds * discharge_rate)
            self.current_capacity.occupied_beds = max(0, self.current_capacity.occupied_beds - discharged)
            
        except Exception as e:
            logger.warning(f"Hospital agent recovery phase error: {e}")
            # Create fallback results
            surge_result = {"status": "processed", "patients_processed": recovery_patients}
            staff_result = {"availability_percentage": 95, "phase": 'recovery'}
            oxygen_result = {"resource_type": 'oxygen', "current_level": 90}
            supplies_result = {"resource_type": 'medical_supplies', "current_level": 85}
            discharged = 10  # fallback discharge count

        return {
            "recovery_patients": surge_result,
            "staff_normalization": staff_result,
            "resource_recovery": {
                "oxygen": oxygen_result,
                "medical_supplies": supplies_result
            },
            "patient_discharge": {
                "discharged_count": discharged,
                "remaining_capacity": self.current_capacity.total_beds - self.current_capacity.occupied_beds
            },
            "phase_focus": "recovery_and_normalization"
        }
    
    async def _generate_hospital_metrics(self) -> Dict[str, Any]:
        """Generate comprehensive hospital metrics"""
        
        bed_utilization = (self.current_capacity.occupied_beds / self.current_capacity.total_beds) * 100
        icu_utilization = (self.current_capacity.icu_occupied / self.current_capacity.icu_beds) * 100
        er_utilization = (self.current_capacity.er_current / self.current_capacity.er_capacity) * 100
        staff_utilization = (self.current_capacity.staff_available / self.current_capacity.staff_total) * 100
        
        return {
            "capacity_metrics": {
                "bed_utilization_percentage": round(bed_utilization, 1),
                "icu_utilization_percentage": round(icu_utilization, 1),
                "er_utilization_percentage": round(er_utilization, 1),
                "staff_utilization_percentage": round(staff_utilization, 1),
                "surge_capacity_remaining": max(0, self.current_capacity.total_beds - self.current_capacity.occupied_beds)
            },
            "resource_metrics": {
                "oxygen_supply_percentage": self.current_capacity.oxygen_supply,
                "medical_supplies_percentage": self.current_capacity.medical_supplies,
                "blood_supply_percentage": self.current_capacity.blood_supply
            },
            "patient_metrics": {
                "total_patients": self.current_capacity.occupied_beds,
                "disaster_related_patients": self.current_load.disaster_related,
                "critical_patients": self.current_load.critical_patients,
                "icu_patients": self.current_capacity.icu_occupied,
                "er_patients": self.current_capacity.er_current
            },
            "operational_metrics": {
                "staff_available": self.current_capacity.staff_available,
                "critical_shortages": self._identify_critical_shortages(),
                "overflow_status": "yes" if bed_utilization >= 100 else "no",
                "system_stress_level": self._calculate_system_stress()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate_system_stress(self) -> str:
        """Calculate overall hospital system stress level"""
        
        bed_stress = (self.current_capacity.occupied_beds / self.current_capacity.total_beds)
        icu_stress = (self.current_capacity.icu_occupied / self.current_capacity.icu_beds)
        resource_stress = (100 - min(self.current_capacity.oxygen_supply, 
                                   self.current_capacity.medical_supplies)) / 100
        
        overall_stress = (bed_stress + icu_stress + resource_stress) / 3
        
        if overall_stress < 0.6:
            return "low"
        elif overall_stress < 0.8:
            return "moderate"
        elif overall_stress < 0.95:
            return "high"
        else:
            return "critical"
    
    async def _save_hospital_state(self, metrics: Dict[str, Any]):
        """Save hospital state to Firestore"""
        try:
            state_data = {
                "agent_id": self.agent_id,
                "simulation_id": self.simulation_id,
                "current_capacity": {
                    "total_beds": self.current_capacity.total_beds,
                    "occupied_beds": self.current_capacity.occupied_beds,
                    "icu_beds": self.current_capacity.icu_beds,
                    "icu_occupied": self.current_capacity.icu_occupied,
                    "staff_available": self.current_capacity.staff_available,
                    "oxygen_supply": self.current_capacity.oxygen_supply,
                    "medical_supplies": self.current_capacity.medical_supplies,
                    "blood_supply": self.current_capacity.blood_supply
                },
                "patient_load": {
                    "disaster_related": self.current_load.disaster_related,
                    "critical_patients": self.current_load.critical_patients,
                    "total_patients": self.current_capacity.occupied_beds
                },
                "metrics": metrics,
                "phase": self.current_phase.value,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.cloud.firestore.save_agent_state(self.agent_id, self.simulation_id, state_data)
            
        except Exception as e:
            logger.error(f"Failed to save hospital state: {e}")
    
    async def _log_hospital_event(self, phase: SimulationPhase, metrics: Dict[str, Any]):
        """Log hospital events to BigQuery for analytics"""
        try:
            event_data = {
                "event_type": "hospital_capacity_update",
                "agent_id": self.agent_id,
                "phase": phase.value,
                "bed_utilization": metrics["capacity_metrics"]["bed_utilization_percentage"],
                "icu_utilization": metrics["capacity_metrics"]["icu_utilization_percentage"],
                "resource_status": {
                    "oxygen": metrics["resource_metrics"]["oxygen_supply_percentage"],
                    "supplies": metrics["resource_metrics"]["medical_supplies_percentage"]
                },
                "system_stress": metrics["operational_metrics"]["system_stress_level"],
                "disaster_patients": metrics["patient_metrics"]["disaster_related_patients"]
            }
            
            await self.cloud.bigquery.log_simulation_event(
                simulation_id=self.simulation_id,
                event_type="hospital_load_update",
                event_data=event_data,
                agent_id=self.agent_id,
                phase=phase.value
            )
            
        except Exception as e:
            logger.error(f"Failed to log hospital event: {e}")


def create_hospital_load_agent(cloud_services: CloudServices) -> HospitalLoadAgent:
    """Factory function to create a Hospital Load Agent"""
    return HospitalLoadAgent(cloud_services)