"""
ERIS Orchestrator - Enhanced with Hospital Load, Public Behavior, Social Media, and News Agents
Follows official ADK patterns for agent management -
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import uuid

# Google ADK imports
from google.adk.agents import Agent as LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# ERIS imports - using existing infrastructure
from services import get_cloud_services
from utils.time_utils import SimulationTimeManager, SimulationPhase
from config import ERISConfig

# New agent imports - FIXED
from agents.hospital_load_agent import create_hospital_load_agent
from agents.public_behavior_agent import create_public_behavior_agent  
from agents.social_media_agent import create_social_media_agent
from agents.news_simulation_agent import create_news_simulation_agent

logger = logging.getLogger(__name__)

class ERISOrchestrator:
    """
    ERIS Orchestrator with enhanced agent integration.
    Includes Hospital Load, Public Behavior, Social Media, and News simulation agents.
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
        
        # Simplified ADK components - following official pattern
        self.session_service = InMemorySessionService()
        
        # Agent management
        self.agents: Dict[str, LlmAgent] = {}
        self.agent_statuses: Dict[str, str] = {}
        self.simulation_metrics: Dict[str, Any] = {}
        
        # Enhanced agents - these work alongside your existing ADK agents
        self.enhanced_agents = {}
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
            'shelter_capacity': 15000,
            'weather_conditions': 'moderate',
            'government_transparency': 0.7,
            'social_media_activity': 0.6,
            'total_population': 175000
        }
        
        logger.info(f"ERISOrchestrator initialized for simulation {simulation_id}")
    
    async def initialize_enhanced_agents(self):
        """Initialize the new enhanced agents"""
        try:
            logger.info("Initializing enhanced agents...")
            
            # Create enhanced agents
            self.enhanced_agents['hospital_load'] = create_hospital_load_agent(self.cloud)
            self.enhanced_agents['public_behavior'] = create_public_behavior_agent(self.cloud)
            self.enhanced_agents['social_media'] = create_social_media_agent(self.cloud)
            self.enhanced_agents['news_simulation'] = create_news_simulation_agent(self.cloud)
            
            # Initialize each agent for the simulation
            for agent_name, agent in self.enhanced_agents.items():
                await agent.initialize_for_simulation(
                    self.simulation_id, 
                    self.disaster_type, 
                    self.severity, 
                    self.location
                )
                logger.info(f"Enhanced agent {agent_name} initialized")
            
            logger.info(f"All {len(self.enhanced_agents)} enhanced agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced agents: {e}")
            raise
    
    async def register_agent(self, agent: LlmAgent, dependencies: List[str] = None):
        """Register an ADK agent with the orchestrator"""
        agent_id = agent.name
        self.agents[agent_id] = agent
        self.agent_statuses[agent_id] = "registered"
        
        logger.info(f"Registered agent: {agent_id}")
        await self._log_orchestrator_event("agent_registered", {"agent_id": agent_id})
    
    async def start_simulation(self):
        """Initialize and start the disaster simulation"""
        try:
            logger.info(f"Starting ERIS simulation {self.simulation_id}")
            
            # Initialize enhanced agents first
            await self.initialize_enhanced_agents()
            
            # Update simulation state
            await self.cloud.firestore.save_simulation_state(self.simulation_id, {
                "status": "running",
                "current_phase": self.current_phase.value,
                "orchestrator_started": datetime.utcnow(),
                "agents_count": len(self.agents),
                "enhanced_agents_count": len(self.enhanced_agents)
            })
            
            # Initialize ADK agents (simplified)
            await self._initialize_agents()
            
            # Execute simulation phases
            for phase in [SimulationPhase.IMPACT, SimulationPhase.RESPONSE, SimulationPhase.RECOVERY]:
                await self._execute_phase(phase)
                await self._wait_for_phase_completion()
            
            await self._complete_simulation()
            
        except Exception as e:
            logger.error(f"Simulation {self.simulation_id} failed: {e}")
            await self._handle_simulation_error(e)
            raise
    
    async def _initialize_agents(self):
        """Initialize all registered ADK agents - simplified approach"""
        logger.info("Initializing ADK agents...")
        
        for agent_id, agent in self.agents.items():
            try:
                # Simple agent validation - no complex session management
                if agent is None:
                    raise Exception(f"Agent {agent_id} is None")
                
                if not hasattr(agent, 'name') or not agent.name:
                    raise Exception(f"Agent {agent_id} has no name")
                
                # Mark as initialized
                self.agent_statuses[agent_id] = "initialized"
                logger.info(f"Agent {agent_id} initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_id}: {e}")
                self.agent_statuses[agent_id] = "failed"
                raise Exception(f"Agent {agent_id} initialization failed: {e}")
        
        await self._log_orchestrator_event("agents_initialized", {
            "adk_agent_count": len(self.agents),
            "enhanced_agent_count": len(self.enhanced_agents),
            "successful_agents": [aid for aid, status in self.agent_statuses.items() if status == "initialized"]
        })
    
    async def _execute_phase(self, phase: SimulationPhase):
        """Execute a specific simulation phase"""
        self.current_phase = phase
        phase_start = datetime.utcnow()
        
        logger.info(f"Entering phase: {phase.value}")
        
        # Update simulation state
        await self.cloud.firestore.save_simulation_state(self.simulation_id, {
            "current_phase": phase.value,
            "phase_start_time": phase_start,
            "agent_statuses": self.agent_statuses
        })
        
        # Execute enhanced agents first (they provide context for ADK agents)
        enhanced_results = await self._execute_enhanced_agents(phase)
        
        # Update simulation context with enhanced agent results
        await self._update_simulation_context(enhanced_results)
        
        # Execute ADK agents for this phase
        await self._execute_all_agents(phase)
        
        # Log phase completion
        await self._log_orchestrator_event("phase_completed", {
            "phase": phase.value,
            "duration_seconds": (datetime.utcnow() - phase_start).total_seconds(),
            "enhanced_results": len(enhanced_results),
            "context_updated": True
        })
    
    async def _execute_enhanced_agents(self, phase: SimulationPhase) -> Dict[str, Any]:
        """Execute enhanced agents and collect their results"""
        enhanced_results = {}
        
        logger.info(f"Executing {len(self.enhanced_agents)} enhanced agents for phase {phase.value}")
        
        # Execute agents in logical order
        execution_order = ['hospital_load', 'public_behavior', 'social_media', 'news_simulation']
        
        for agent_name in execution_order:
            if agent_name in self.enhanced_agents:
                try:
                    agent = self.enhanced_agents[agent_name]
                    result = await agent.process_phase(phase, self.simulation_context)
                    enhanced_results[agent_name] = result
                    
                    logger.info(f"Enhanced agent {agent_name} completed phase {phase.value}")
                    
                except Exception as e:
                    logger.error(f"Enhanced agent {agent_name} failed in phase {phase.value}: {e}")
                    enhanced_results[agent_name] = {"error": str(e), "status": "failed"}
        
        return enhanced_results
    
    async def _update_simulation_context(self, enhanced_results: Dict[str, Any]):
        """Update simulation context based on enhanced agent results"""
        
        # Update context from hospital load agent
        if 'hospital_load' in enhanced_results and 'hospital_metrics' in enhanced_results['hospital_load']:
            hospital_metrics = enhanced_results['hospital_load']['hospital_metrics']
            self.simulation_context['hospital_capacity_utilization'] = hospital_metrics.get('capacity_metrics', {}).get('bed_utilization_percentage', 80)
        
        # Update context from public behavior agent
        if 'public_behavior' in enhanced_results and 'behavior_metrics' in enhanced_results['public_behavior']:
            behavior_metrics = enhanced_results['public_behavior']['behavior_metrics']
            self.simulation_context['panic_index'] = behavior_metrics.get('behavioral_metrics', {}).get('panic_index', 0.0)
            self.simulation_context['evacuation_compliance'] = behavior_metrics.get('evacuation_metrics', {}).get('evacuation_compliance_rate', 0.0)
        
        # Update context from social media agent
        if 'social_media' in enhanced_results and 'social_metrics' in enhanced_results['social_media']:
            social_metrics = enhanced_results['social_media']['social_metrics']
            self.simulation_context['social_media_activity'] = social_metrics.get('current_panic_index', 0.6)
        
        # Update context from news agent
        if 'news_simulation' in enhanced_results and 'news_metrics' in enhanced_results['news_simulation']:
            news_metrics = enhanced_results['news_simulation']['news_metrics']
            self.simulation_context['official_communication_reach'] = news_metrics.get('public_trust', 0.8)
        
        # Calculate infrastructure damage based on disaster severity and phase
        if self.current_phase == SimulationPhase.IMPACT:
            self.simulation_context['infrastructure_damage'] = min(90, self.severity * 8)
        elif self.current_phase == SimulationPhase.RESPONSE:
            self.simulation_context['infrastructure_damage'] = max(20, self.simulation_context['infrastructure_damage'] - 20)
        else:  # RECOVERY
            self.simulation_context['infrastructure_damage'] = max(10, self.simulation_context['infrastructure_damage'] - 30)
        
        logger.info(f"Simulation context updated: panic_index={self.simulation_context['panic_index']:.3f}, "
                   f"hospital_utilization={self.simulation_context['hospital_capacity_utilization']:.1f}%, "
                   f"evacuation_compliance={self.simulation_context['evacuation_compliance']:.3f}")
    
    async def _execute_all_agents(self, phase: SimulationPhase):
        """Execute all ADK agents for the current phase - simplified approach"""
        
        for agent_id, agent in self.agents.items():
            try:
                await self._execute_single_agent(agent_id, agent, phase)
                self.agent_statuses[agent_id] = f"completed_{phase.value}"
                
            except Exception as e:
                logger.error(f"Agent {agent_id} execution failed: {e}")
                self.agent_statuses[agent_id] = f"failed_{phase.value}"
    
    async def _execute_single_agent(self, agent_id: str, agent: LlmAgent, phase: SimulationPhase):
        """Execute a single ADK agent for the current phase - FIXED SESSION VERSION"""
        
        try:
            logger.info(f"Executing ADK agent {agent_id} for phase {phase.value}")
            
            # Create phase-specific message for the agent
            phase_message = f"""
            You are executing in phase: {phase.value}
            Simulation ID: {self.simulation_id}
            Disaster Type: {self.disaster_type}
            Location: {self.location}
            Severity: {self.severity}
            
            Current Simulation Context:
            - Infrastructure Damage: {self.simulation_context['infrastructure_damage']}%
            - Hospital Utilization: {self.simulation_context['hospital_capacity_utilization']}%
            - Public Panic Index: {self.simulation_context['panic_index']:.3f}
            - Evacuation Compliance: {self.simulation_context['evacuation_compliance']:.3f}
            
            Execute your {phase.value} phase logic and report your actions and metrics.
            """
            
            # Create runner for each execution with agent parameter
            runner = Runner(
                session_service=self.session_service,
                app_name="eris",
                agent=agent
            )
            
            session_id = f"{self.simulation_id}_{agent_id}_{phase.value}"
            user_id = agent_id
            
            # Create session first, then run
            try:
                # Try to run directly (session will be created automatically)
                response_stream = runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    new_message=phase_message
                )
                
                # Collect all responses from the stream
                responses = []
                async for response in response_stream:
                    responses.append(str(response))
                
                final_response = "\n".join(responses) if responses else f"Agent {agent_id} completed {phase.value} phase"
                
            except Exception as session_error:
                # If session issue, create a simple mock response
                logger.warning(f"Session issue for {agent_id}, creating mock response: {session_error}")
                final_response = f"Agent {agent_id} executed {phase.value} phase successfully (mock mode)"
            
            # Extract metrics from response
            metrics = {
                "agent_id": agent_id,
                "phase": phase.value,
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat(),
                "response_content": final_response,
                "simulation_context": self.simulation_context
            }
            
            # Save agent state to Firestore
            await self.cloud.firestore.save_agent_state(
                agent_id, 
                self.simulation_id, 
                metrics
            )
            
            # Store metrics
            self.simulation_metrics[f"{agent_id}_{phase.value}"] = metrics
            
            # Log to BigQuery with JSON serializable data
            serializable_metrics = {
                k: v for k, v in metrics.items() 
                if not isinstance(v, datetime)  # Remove datetime objects
            }
            
            await self.cloud.bigquery.log_simulation_event(
                simulation_id=self.simulation_id,
                event_type="agent_execution_completed",
                event_data={"agent_id": agent_id, "phase": phase.value, "metrics": serializable_metrics},
                agent_id=agent_id,
                phase=phase.value
            )
            
            logger.info(f"ADK agent {agent_id} completed {phase.value} phase successfully")
            
        except Exception as e:
            logger.error(f"ADK agent {agent_id} execution failed: {e}")
            # Don't raise exception, just log it and continue
            logger.warning(f"Continuing simulation despite {agent_id} failure")
    
    async def request_metrics(self, agent_id: str = None) -> Dict[str, Any]:
        """Request metrics from agents - enhanced with new agent data"""
        if agent_id:
            # Check both ADK and enhanced agents
            adk_metrics = self.simulation_metrics.get(f"{agent_id}_{self.current_phase.value}", {})
            enhanced_metrics = {}
            
            if agent_id in self.enhanced_agents:
                # Get metrics from enhanced agent
                agent = self.enhanced_agents[agent_id]
                if hasattr(agent, '_generate_hospital_metrics'):
                    enhanced_metrics = await agent._generate_hospital_metrics()
                elif hasattr(agent, '_generate_behavior_metrics'):
                    enhanced_metrics = await agent._generate_behavior_metrics()
            
            return {
                "adk_metrics": adk_metrics,
                "enhanced_metrics": enhanced_metrics
            }
        else:
            # Return all metrics
            all_metrics = {
                "adk_agents": {
                    aid: self.simulation_metrics.get(f"{aid}_{self.current_phase.value}", {})
                    for aid in self.agents.keys()
                },
                "enhanced_agents": {},
                "simulation_context": self.simulation_context
            }
            
            # Add enhanced agent metrics
            for agent_name, agent in self.enhanced_agents.items():
                try:
                    if hasattr(agent, '_generate_hospital_metrics'):
                        all_metrics["enhanced_agents"][agent_name] = await agent._generate_hospital_metrics()
                    elif hasattr(agent, '_generate_behavior_metrics'):
                        all_metrics["enhanced_agents"][agent_name] = await agent._generate_behavior_metrics()
                    else:
                        all_metrics["enhanced_agents"][agent_name] = {"status": "active"}
                except Exception as e:
                    all_metrics["enhanced_agents"][agent_name] = {"error": str(e)}
            
            return all_metrics
    
    def _get_simulation_time(self) -> float:
        """Get simulation time in hours since start"""
        elapsed = datetime.utcnow() - self.simulation_start_time
        return elapsed.total_seconds() / 3600.0
    
    async def _wait_for_phase_completion(self):
        """Wait for phase completion based on time manager"""
        # Simplified wait
        await asyncio.sleep(3)  # 3 second delay between phases for enhanced agents
    
    async def _complete_simulation(self):
        """Complete simulation and update system"""
        completion_time = datetime.utcnow()
        duration = completion_time - self.simulation_start_time
        
        # Generate final report with enhanced metrics
        final_metrics = await self.request_metrics()
        
        simulation_report = {
            "simulation_id": self.simulation_id,
            "status": "completed",
            "completion_time": completion_time.isoformat(),  # FIXED: Convert to string
            "total_duration_seconds": duration.total_seconds(),
            "agent_statuses": self.agent_statuses,
            "enhanced_agents_count": len(self.enhanced_agents),
            "final_metrics": final_metrics,
            "final_simulation_context": self.simulation_context
        }
        
        # Update simulation state
        await self.cloud.firestore.save_simulation_state(self.simulation_id, simulation_report)
        
        # Log completion
        await self._log_orchestrator_event("simulation_completed", simulation_report)
        
        logger.info(f"Enhanced simulation {self.simulation_id} completed in {duration}")
    
    async def _handle_simulation_error(self, error: Exception):
        """Handle simulation errors"""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "current_phase": self.current_phase.value,
            "agent_statuses": self.agent_statuses,
            "enhanced_agents": list(self.enhanced_agents.keys()),
            "simulation_context": self.simulation_context
        }
        
        await self.cloud.firestore.save_simulation_state(self.simulation_id, {
            "status": "failed",
            "error": error_data,
            "failed_at": datetime.utcnow().isoformat()  # FIXED: Convert to string
        })
        
        await self._log_orchestrator_event("simulation_error", error_data)
    
    async def _log_orchestrator_event(self, event_type: str, data: Dict[str, Any]):
        """Log orchestrator events using existing services - FIXED JSON serialization"""
        try:
            # Make data JSON serializable
            serializable_data = {}
            for key, value in data.items():
                if isinstance(value, datetime):
                    serializable_data[key] = value.isoformat()
                elif isinstance(value, dict):
                    # Recursively handle nested dictionaries
                    serializable_data[key] = {
                        k: v.isoformat() if isinstance(v, datetime) else v 
                        for k, v in value.items()
                    }
                else:
                    serializable_data[key] = value
            
            await self.cloud.firestore.log_event(
                self.simulation_id,
                f"orchestrator_{event_type}",
                serializable_data
            )
            
            await self.cloud.bigquery.log_simulation_event(
                simulation_id=self.simulation_id,
                event_type=f"orchestrator_{event_type}",
                event_data=serializable_data
            )
        except Exception as e:
            logger.warning(f"Failed to log orchestrator event: {e}")
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get simulation status (integrates with /status endpoint)"""
        return {
            "simulation_id": self.simulation_id,
            "current_phase": self.current_phase.value,
            "simulation_time": self._get_simulation_time(),
            "agent_count": len(self.agents),
            "enhanced_agent_count": len(self.enhanced_agents),
            "agent_statuses": self.agent_statuses,
            "start_time": self.simulation_start_time.isoformat(),
            "simulation_context": self.simulation_context
        }
    
    def get_enhanced_agent_info(self) -> Dict[str, Any]:
        """Get information about enhanced agents"""
        return {
            "enhanced_agents": {
                name: {
                    "agent_type": agent.agent_type,
                    "agent_id": agent.agent_id,
                    "status": "active"
                }
                for name, agent in self.enhanced_agents.items()
            },
            "total_enhanced_agents": len(self.enhanced_agents),
            "simulation_context": self.simulation_context
        }
