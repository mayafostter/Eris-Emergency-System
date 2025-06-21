"""
ERIS Orchestrator
Coordinates 6 ADK agents + 4 enhanced agents
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Google ADK imports
from google.adk.agents import Agent as LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# ERIS imports
from services import get_cloud_services
from utils.time_utils import SimulationTimeManager, SimulationPhase
from config import ERISConfig

# ADK Agent imports - Updated names
from agents.base_agent import (
    create_emergency_response_agent,
    create_public_health_agent,
    create_infrastructure_manager_agent,
    create_logistics_coordinator_agent,
    create_communications_director_agent,
    create_recovery_coordinator_agent
)

# Enhanced agent imports
from agents.hospital_load_agent import create_hospital_load_agent
from agents.public_behavior_agent import create_public_behavior_agent  
from agents.social_media_agent import create_social_media_agent
from agents.news_simulation_agent import create_news_simulation_agent

logger = logging.getLogger(__name__)

class ERISOrchestrator:
    """
    ERIS Orchestrator managing 10 agents total:
    - 6 ADK agents (Google ADK framework)
    - 4 Enhanced agents (specialized implementations)
    """
    
    def __init__(self, simulation_id: str, disaster_type: str, location: str, severity: int, duration: int = 72):
        # Simulation configuration
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.location = location
        self.severity = severity
        
        # ERIS infrastructure
        self.cloud = get_cloud_services()
        self.config = ERISConfig()
        self.time_manager = SimulationTimeManager(duration_hours=duration)
        
        # Orchestrator state
        self.current_phase = SimulationPhase.IMPACT
        self.simulation_start_time = datetime.utcnow()
        
        # ADK components
        self.session_service = InMemorySessionService()
        
        # Agent management
        self.adk_agents = {}  # 6 ADK agents
        self.enhanced_agents = {}  # 4 enhanced agents
        self.agent_statuses = {}
        self.simulation_metrics = {}
        
        # Simulation context (shared across all agents)
        self.simulation_context = {
            'disaster_type': disaster_type,
            'location': location,
            'severity': severity,
            'infrastructure_damage': 0,
            'hospital_capacity_utilization': 80,
            'panic_index': 0.0,
            'evacuation_compliance': 0.0,
            'official_communication_reach': 0.8,
            'supply_chain_disrupted': False,
            'total_population': 175000
        }
        
        logger.info(f"ERISOrchestrator initialized for simulation {simulation_id}")
    
    async def initialize_all_agents(self):
        """Initialize both ADK and enhanced agents"""
        try:
            logger.info("Initializing all agents...")
            
            # Initialize 6 ADK agents
            await self._initialize_adk_agents()
            
            # Initialize 4 enhanced agents  
            await self._initialize_enhanced_agents()
            
            logger.info(f"All agents initialized: {len(self.adk_agents)} ADK + {len(self.enhanced_agents)} enhanced")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise
    
    async def _initialize_adk_agents(self):
        """Initialize the 6 ADK agents"""
        adk_agent_creators = {
            'emergency_response': create_emergency_response_agent,
            'public_health': create_public_health_agent,
            'infrastructure_manager': create_infrastructure_manager_agent,
            'logistics_coordinator': create_logistics_coordinator_agent,
            'communications_director': create_communications_director_agent,
            'recovery_coordinator': create_recovery_coordinator_agent
        }
        
        for agent_name, creator_func in adk_agent_creators.items():
            try:
                # Create agent instance
                agent_instance = creator_func(self.cloud)
                
                # Initialize for simulation
                await agent_instance.initialize_for_simulation(
                    self.simulation_id,
                    self.disaster_type, 
                    self.severity,
                    self.location
                )
                
                self.adk_agents[agent_name] = agent_instance
                self.agent_statuses[agent_name] = "initialized"
                
                logger.info(f"ADK agent {agent_name} initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize ADK agent {agent_name}: {e}")
                self.agent_statuses[agent_name] = "failed"
    
    async def _initialize_enhanced_agents(self):
        """Initialize the 4 enhanced agents"""
        enhanced_agent_creators = {
            'hospital_load': create_hospital_load_agent,
            'public_behavior': create_public_behavior_agent,
            'social_media': create_social_media_agent,
            'news_simulation': create_news_simulation_agent
        }
        
        for agent_name, creator_func in enhanced_agent_creators.items():
            try:
                # Create agent instance
                agent_instance = creator_func(self.cloud)
                
                # Initialize for simulation
                await agent_instance.initialize_for_simulation(
                    self.simulation_id,
                    self.disaster_type,
                    self.severity,
                    self.location
                )
                
                self.enhanced_agents[agent_name] = agent_instance
                self.agent_statuses[agent_name] = "initialized"
                
                logger.info(f"Enhanced agent {agent_name} initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize enhanced agent {agent_name}: {e}")
                self.agent_statuses[agent_name] = "failed"
    
    async def start_simulation(self):
        """Start the complete simulation"""
        try:
            logger.info(f"Starting ERIS simulation {self.simulation_id}")
            
            # Initialize all agents
            await self.initialize_all_agents()
            
            # Save initial state
            await self.cloud.firestore.save_simulation_state(self.simulation_id, {
                "status": "running",
                "current_phase": self.current_phase.value,
                "adk_agents_count": len(self.adk_agents),
                "enhanced_agents_count": len(self.enhanced_agents),
                "total_agents": len(self.adk_agents) + len(self.enhanced_agents),
                "simulation_start": datetime.utcnow().isoformat()
            })
            
            # Execute simulation phases
            for phase in [SimulationPhase.IMPACT, SimulationPhase.RESPONSE, SimulationPhase.RECOVERY]:
                await self._execute_phase(phase)
                await asyncio.sleep(2)  # Brief pause between phases
            
            await self._complete_simulation()
            
        except Exception as e:
            logger.error(f"Simulation {self.simulation_id} failed: {e}")
            await self._handle_simulation_error(e)
            raise
    
    async def _execute_phase(self, phase: SimulationPhase):
        """Execute all agents for a specific phase"""
        self.current_phase = phase
        phase_start = datetime.utcnow()
        
        logger.info(f"Executing phase: {phase.value}")
        
        # 1. Execute enhanced agents first (they provide context)
        enhanced_results = await self._execute_enhanced_agents(phase)
        
        # 2. Update simulation context based on enhanced agent results
        self._update_simulation_context(enhanced_results)
        
        # 3. Execute ADK agents with updated context
        adk_results = await self._execute_adk_agents(phase)
        
        # 4. Log phase completion
        await self._log_phase_completion(phase, phase_start, enhanced_results, adk_results)
    
    async def _execute_enhanced_agents(self, phase: SimulationPhase) -> Dict[str, Any]:
        """Execute the 4 enhanced agents"""
        results = {}
        
        execution_order = ['hospital_load', 'public_behavior', 'social_media', 'news_simulation']
        
        for agent_name in execution_order:
            if agent_name in self.enhanced_agents:
                try:
                    agent = self.enhanced_agents[agent_name]
                    result = await agent.process_phase(phase, self.simulation_context)
                    results[agent_name] = result
                    
                    self.agent_statuses[agent_name] = f"completed_{phase.value}"
                    logger.info(f"Enhanced agent {agent_name} completed {phase.value}")
                    
                except Exception as e:
                    logger.error(f"Enhanced agent {agent_name} failed: {e}")
                    results[agent_name] = {"error": str(e), "status": "failed"}
                    self.agent_statuses[agent_name] = f"failed_{phase.value}"
        
        return results
    
    async def _execute_adk_agents(self, phase: SimulationPhase) -> Dict[str, Any]:
        """Execute the 6 ADK agents"""
        results = {}
        
        for agent_name, agent_instance in self.adk_agents.items():
            try:
                result = await agent_instance.process_phase(phase, self.simulation_context)
                results[agent_name] = result
                
                self.agent_statuses[agent_name] = f"completed_{phase.value}"
                logger.info(f"ADK agent {agent_name} completed {phase.value}")
                
            except Exception as e:
                logger.error(f"ADK agent {agent_name} failed: {e}")
                results[agent_name] = {"error": str(e), "status": "failed"}
                self.agent_statuses[agent_name] = f"failed_{phase.value}"
        
        return results
    
    def _update_simulation_context(self, enhanced_results: Dict[str, Any]):
        """Update simulation context based on enhanced agent outputs"""
        
        # Update from hospital load agent
        if 'hospital_load' in enhanced_results:
            hospital_data = enhanced_results['hospital_load']
            if 'metrics' in hospital_data:
                metrics = hospital_data['metrics']
                if 'resource_utilization' in metrics:
                    self.simulation_context['hospital_capacity_utilization'] = metrics['resource_utilization']
        
        # Update from public behavior agent
        if 'public_behavior' in enhanced_results:
            behavior_data = enhanced_results['public_behavior']
            if 'metrics' in behavior_data:
                metrics = behavior_data['metrics']
                if 'panic_index' in metrics:
                    self.simulation_context['panic_index'] = metrics['panic_index']
                if 'evacuation_compliance' in metrics:
                    self.simulation_context['evacuation_compliance'] = metrics['evacuation_compliance']
        
        # Update from social media agent
        if 'social_media' in enhanced_results:
            social_data = enhanced_results['social_media']
            if 'metrics' in social_data:
                metrics = social_data['metrics']
                # Update social media activity levels
                self.simulation_context['social_media_activity'] = metrics.get('activity_level', 0.6)
        
        # Update from news simulation agent
        if 'news_simulation' in enhanced_results:
            news_data = enhanced_results['news_simulation']
            if 'metrics' in news_data:
                metrics = news_data['metrics']
                # Update official communication reach
                self.simulation_context['official_communication_reach'] = metrics.get('public_trust', 0.8)
        
        # Update infrastructure damage based on phase and severity
        if self.current_phase == SimulationPhase.IMPACT:
            self.simulation_context['infrastructure_damage'] = min(90, self.severity * 8)
        elif self.current_phase == SimulationPhase.RESPONSE:
            self.simulation_context['infrastructure_damage'] = max(20, self.simulation_context['infrastructure_damage'] - 20)
        else:  # RECOVERY
            self.simulation_context['infrastructure_damage'] = max(10, self.simulation_context['infrastructure_damage'] - 30)
        
        logger.info(f"Context updated - Panic: {self.simulation_context['panic_index']:.2f}, "
                   f"Hospital: {self.simulation_context['hospital_capacity_utilization']:.1f}%, "
                   f"Infrastructure: {self.simulation_context['infrastructure_damage']}%")
    
    async def _log_phase_completion(self, phase: SimulationPhase, phase_start: datetime, 
                                   enhanced_results: Dict, adk_results: Dict):
        """Log completion of simulation phase"""
        duration = (datetime.utcnow() - phase_start).total_seconds()
        
        phase_data = {
            "phase": phase.value,
            "duration_seconds": duration,
            "enhanced_agents_completed": len([r for r in enhanced_results.values() if r.get('status') != 'failed']),
            "adk_agents_completed": len([r for r in adk_results.values() if r.get('status') != 'failed']),
            "total_agents_executed": len(enhanced_results) + len(adk_results),
            "simulation_context": self.simulation_context
        }
        
        try:
            await self.cloud.firestore.log_event(
                self.simulation_id,
                f"phase_{phase.value}_completed", 
                phase_data
            )
            
            await self.cloud.bigquery.log_simulation_event(
                simulation_id=self.simulation_id,
                event_type=f"phase_{phase.value}_completed",
                event_data=phase_data
            )
        except Exception as e:
            logger.warning(f"Failed to log phase completion: {e}")
    
    async def _complete_simulation(self):
        """Complete the simulation and generate final report"""
        completion_time = datetime.utcnow()
        duration = completion_time - self.simulation_start_time
        
        # Generate final metrics from all agents
        final_metrics = await self._collect_final_metrics()
        
        simulation_report = {
            "simulation_id": self.simulation_id,
            "status": "completed",
            "completion_time": completion_time.isoformat(),
            "total_duration_seconds": duration.total_seconds(),
            "adk_agents_count": len(self.adk_agents),
            "enhanced_agents_count": len(self.enhanced_agents),
            "agent_statuses": self.agent_statuses,
            "final_simulation_context": self.simulation_context,
            "final_metrics": final_metrics
        }
        
        # Save final state
        await self.cloud.firestore.save_simulation_state(self.simulation_id, simulation_report)
        
        logger.info(f"Simulation {self.simulation_id} completed successfully in {duration}")
    
    async def _collect_final_metrics(self) -> Dict[str, Any]:
        """Collect final metrics from all agents"""
        metrics = {
            "adk_agents": {},
            "enhanced_agents": {},
            "simulation_summary": {
                "total_agents": len(self.adk_agents) + len(self.enhanced_agents),
                "successful_agents": len([s for s in self.agent_statuses.values() if "completed" in s]),
                "failed_agents": len([s for s in self.agent_statuses.values() if "failed" in s])
            }
        }
        
        # Collect from ADK agents
        for agent_name, agent_instance in self.adk_agents.items():
            try:
                if hasattr(agent_instance, '_generate_metrics'):
                    agent_metrics = await agent_instance._generate_metrics()
                    metrics["adk_agents"][agent_name] = agent_metrics
                else:
                    metrics["adk_agents"][agent_name] = {"status": "completed"}
            except Exception as e:
                metrics["adk_agents"][agent_name] = {"error": str(e)}
        
        # Collect from enhanced agents
        for agent_name, agent_instance in self.enhanced_agents.items():
            try:
                # Try to get metrics (each enhanced agent has different method names)
                if hasattr(agent_instance, '_generate_hospital_metrics'):
                    agent_metrics = await agent_instance._generate_hospital_metrics()
                elif hasattr(agent_instance, '_generate_behavior_metrics'):
                    agent_metrics = await agent_instance._generate_behavior_metrics()
                elif hasattr(agent_instance, '_generate_metrics'):
                    agent_metrics = await agent_instance._generate_metrics()
                else:
                    agent_metrics = {"status": "completed"}
                
                metrics["enhanced_agents"][agent_name] = agent_metrics
            except Exception as e:
                metrics["enhanced_agents"][agent_name] = {"error": str(e)}
        
        return metrics
    
    async def _handle_simulation_error(self, error: Exception):
        """Handle simulation errors"""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "current_phase": self.current_phase.value,
            "agent_statuses": self.agent_statuses,
            "failed_at": datetime.utcnow().isoformat()
        }
        
        await self.cloud.firestore.save_simulation_state(self.simulation_id, {
            "status": "failed",
            "error": error_data
        })
        
        logger.error(f"Simulation {self.simulation_id} failed: {error}")
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status"""
        elapsed_time = datetime.utcnow() - self.simulation_start_time
        
        return {
            "simulation_id": self.simulation_id,
            "current_phase": self.current_phase.value,
            "elapsed_time_seconds": elapsed_time.total_seconds(),
            "adk_agents_count": len(self.adk_agents),
            "enhanced_agents_count": len(self.enhanced_agents),
            "total_agents": len(self.adk_agents) + len(self.enhanced_agents),
            "agent_statuses": self.agent_statuses,
            "simulation_context": self.simulation_context,
            "disaster_info": {
                "type": self.disaster_type,
                "location": self.location,
                "severity": self.severity
            }
        }
    
    def get_all_agent_info(self) -> Dict[str, Any]:
        """Get information about all agents"""
        return {
            "adk_agents": {
                name: {
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type,
                    "status": self.agent_statuses.get(name, "unknown")
                }
                for name, agent in self.adk_agents.items()
            },
            "enhanced_agents": {
                name: {
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type,
                    "status": self.agent_statuses.get(name, "unknown")
                }
                for name, agent in self.enhanced_agents.items()
            },
            "total_agents": len(self.adk_agents) + len(self.enhanced_agents),
            "simulation_context": self.simulation_context
        }
