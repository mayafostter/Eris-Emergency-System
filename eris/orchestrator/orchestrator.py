"""
ERIS Orchestrator
Coordinates 6 ADK agents + 4 enhanced agents with optimizations
"""

import asyncio
import logging
import gc
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# Google ADK imports
from google.adk.agents import Agent as LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# ERIS imports
from services import get_cloud_services
from utils.time_utils import SimulationTimeManager, SimulationPhase
from config import ERISConfig

# ADK Agent imports
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

@dataclass
class AgentExecutionResult:
    """Structured result from agent execution"""
    agent_name: str
    agent_type: str
    status: str
    execution_time: float
    result_data: Dict[str, Any]
    error: Optional[str] = None
    efficiency_score: float = 0.0

class OptimizedSimulationContext:
    """Memory-efficient simulation context with automatic cleanup"""
    
    def __init__(self, max_size: int = 50):
        self._data: Dict[str, Any] = {}
        self._access_times: Dict[str, datetime] = {}
        self._max_size = max_size
        self._cleanup_threshold = max_size * 0.8
    
    def __setitem__(self, key: str, value: Any):
        if len(self._data) > self._cleanup_threshold:
            self._cleanup_old_entries()
        self._data[key] = value
        self._access_times[key] = datetime.utcnow()
    
    def __getitem__(self, key: str):
        self._access_times[key] = datetime.utcnow()
        return self._data[key]
    
    def __contains__(self, key: str):
        return key in self._data
    
    def get(self, key: str, default=None):
        if key in self._data:
            self._access_times[key] = datetime.utcnow()
            return self._data[key]
        return default
    
    def update(self, other: Dict[str, Any]):
        current_time = datetime.utcnow()
        for key, value in other.items():
            self._data[key] = value
            self._access_times[key] = current_time
        if len(self._data) > self._cleanup_threshold:
            self._cleanup_old_entries()
    
    def _cleanup_old_entries(self):
        if len(self._data) <= self._max_size:
            return
        sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
        items_to_remove = len(self._data) - int(self._max_size * 0.7)
        for key, _ in sorted_items[:items_to_remove]:
            if key in self._data:
                del self._data[key]
            if key in self._access_times:
                del self._access_times[key]
    
    def clear(self):
        self._data.clear()
        self._access_times.clear()
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        return self._data.items()

class ERISOrchestrator:
    """
    Production-ready ERIS Orchestrator managing 10 agents total:
    - 6 ADK agents (Google ADK framework)
    - 4 Enhanced agents (specialized implementations)
    
    Features:
    - Concurrent agent execution
    - Memory optimization
    - Error recovery
    - Performance monitoring
    """
    
    def __init__(self, simulation_id: str, disaster_type: str, location: str, severity: int, duration: int = 72):
        # Simulation configuration
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.location = location
        self.disaster_severity = severity
        self.severity = severity
        
        # ERIS infrastructure with error handling
        try:
            self.cloud = get_cloud_services()
            self.config = ERISConfig()
        except Exception as e:
            logger.warning(f"Cloud services initialization failed: {e}, using fallback")
            self.cloud = self._create_fallback_services()
            self.config = self._create_fallback_config()
        
        self.time_manager = SimulationTimeManager(duration_hours=duration)
        
        # Orchestrator state
        self.current_phase = SimulationPhase.IMPACT
        self.simulation_start_time = datetime.utcnow()
        
        # ADK components
        self.session_service = InMemorySessionService()
        
        # Agent management
        self.adk_agents = {}
        self.enhanced_agents = {}
        self.agent_statuses = {}
        self.agent_restart_attempts = {}
        self.max_restart_attempts = 3
        
        # Performance optimization
        self.max_concurrent_agents = 6
        self.agent_timeout = 30
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.performance_metrics = {
            'total_execution_time': 0,
            'agent_execution_times': {},
            'concurrent_executions': 0,
            'efficiency_score': 0,
            'phase_times': {}
        }
        
        # Memory-optimized simulation context
        self.simulation_context = OptimizedSimulationContext()
        self.simulation_context.update({
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
        })
        
        logger.info(f"ERISOrchestrator initialized for simulation {simulation_id} with optimizations")
    
    def _create_fallback_services(self):
        """Create fallback cloud services"""
        return type('FallbackCloud', (), {
            'firestore': type('MockFirestore', (), {
                'save_simulation_state': lambda self, sim_id, data: asyncio.sleep(0),
                'get_simulation_state': lambda self, sim_id: {"status": "active"},
                'log_event': lambda self, sim_id, event_type, data: asyncio.sleep(0)
            })(),
            'vertex_ai': type('MockVertexAI', (), {
                'generate_official_statements': lambda self, context, stage, dept, type: "Mock emergency statement"
            })(),
            'bigquery': type('MockBigQuery', (), {
                'log_simulation_event': lambda self, **kwargs: asyncio.sleep(0)
            })()
        })()
    
    def _create_fallback_config(self):
        """Create fallback configuration"""
        return type('FallbackConfig', (), {
            'get_disaster_config': lambda self, disaster_type: {"severity_multiplier": 1.0, "duration": 24}
        })()
    
    async def initialize_all_agents(self):
        """Initialize both ADK and enhanced agents concurrently"""
        try:
            logger.info("Initializing all agents with concurrent optimization...")
            
            # Initialize ADK and enhanced agents concurrently
            adk_task = asyncio.create_task(self._initialize_adk_agents())
            enhanced_task = asyncio.create_task(self._initialize_enhanced_agents())
            
            # Wait for both to complete
            await asyncio.gather(adk_task, enhanced_task, return_exceptions=True)
            
            total_agents = len(self.adk_agents) + len(self.enhanced_agents)
            successful_agents = len([s for s in self.agent_statuses.values() if s == "initialized"])
            
            logger.info(f"Agent initialization complete: {successful_agents}/{total_agents} agents ready")
            
            if successful_agents < 6:
                raise Exception(f"Insufficient agents initialized: {successful_agents}/10")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise
    
    async def _initialize_adk_agents(self):
        """Initialize the 6 ADK agents concurrently"""
        adk_agent_creators = {
            'emergency_response': create_emergency_response_agent,
            'public_health': create_public_health_agent,
            'infrastructure_manager': create_infrastructure_manager_agent,
            'logistics_coordinator': create_logistics_coordinator_agent,
            'communications_director': create_communications_director_agent,
            'recovery_coordinator': create_recovery_coordinator_agent
        }
        
        tasks = []
        for agent_name, creator_func in adk_agent_creators.items():
            task = asyncio.create_task(self._initialize_single_adk_agent(agent_name, creator_func))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _initialize_single_adk_agent(self, agent_name: str, creator_func):
        """Initialize a single ADK agent"""
        try:
            agent_instance = creator_func(self.cloud)
            await agent_instance.initialize_for_simulation(
                self.simulation_id, self.disaster_type, self.severity, self.location
            )
            self.adk_agents[agent_name] = agent_instance
            self.agent_statuses[agent_name] = "initialized"
            self.agent_restart_attempts[agent_name] = 0
            logger.info(f"ADK agent {agent_name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ADK agent {agent_name}: {e}")
            self.agent_statuses[agent_name] = "failed"
    
    async def _initialize_enhanced_agents(self):
        """Initialize the 4 enhanced agents concurrently"""
        enhanced_agent_creators = {
            'hospital_load': create_hospital_load_agent,
            'public_behavior': create_public_behavior_agent,
            'social_media': create_social_media_agent,
            'news_simulation': create_news_simulation_agent
        }
        
        tasks = []
        for agent_name, creator_func in enhanced_agent_creators.items():
            task = asyncio.create_task(self._initialize_single_enhanced_agent(agent_name, creator_func))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _initialize_single_enhanced_agent(self, agent_name: str, creator_func):
        """Initialize a single enhanced agent"""
        try:
            agent_instance = creator_func(self.cloud)
            await agent_instance.initialize_for_simulation(
                self.simulation_id, self.disaster_type, self.severity, self.location
            )
            self.enhanced_agents[agent_name] = agent_instance
            self.agent_statuses[agent_name] = "initialized"
            self.agent_restart_attempts[agent_name] = 0
            logger.info(f"Enhanced agent {agent_name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced agent {agent_name}: {e}")
            self.agent_statuses[agent_name] = "failed"
    
    async def start_simulation(self):
        """Start the complete optimized simulation"""
        try:
            logger.info(f"Starting optimized ERIS simulation {self.simulation_id}")
            
            await self.initialize_all_agents()
            
            await self._save_simulation_state({
                "status": "running",
                "current_phase": self.current_phase.value,
                "adk_agents_count": len(self.adk_agents),
                "enhanced_agents_count": len(self.enhanced_agents),
                "total_agents": len(self.adk_agents) + len(self.enhanced_agents),
                "simulation_start": datetime.utcnow().isoformat(),
                "optimizations_enabled": True
            })
            
            # Execute simulation phases
            for phase in [SimulationPhase.IMPACT, SimulationPhase.RESPONSE, SimulationPhase.RECOVERY]:
                phase_start = datetime.utcnow()
                await self._execute_phase_optimized(phase)
                phase_duration = (datetime.utcnow() - phase_start).total_seconds()
                self.performance_metrics['phase_times'][phase.value] = phase_duration
                await asyncio.sleep(1)
            
            await self._complete_simulation()
            
        except Exception as e:
            logger.error(f"Optimized simulation {self.simulation_id} failed: {e}")
            await self._handle_simulation_error(e)
            raise
    
    async def _execute_phase_optimized(self, phase: SimulationPhase):
        """Execute all agents for a specific phase with optimization"""
        self.current_phase = phase
        phase_start = datetime.utcnow()
        
        logger.info(f"Executing phase: {phase.value} (optimized concurrent execution)")
        
        # Execute enhanced agents first (they provide context)
        enhanced_results = await self._execute_enhanced_agents_concurrent(phase)
        
        # Update simulation context
        self._update_simulation_context_optimized(enhanced_results)
        
        # Execute ADK agents with updated context
        adk_results = await self._execute_adk_agents_concurrent(phase)
        
        # Update agent statuses
        all_results = {**enhanced_results, **adk_results}
        for name, result in all_results.items():
            self.agent_statuses[name] = f"{result.status}_{phase.value}"
            if result.execution_time > 0:
                self.performance_metrics['agent_execution_times'][name] = result.execution_time
        
        # Log performance
        phase_duration = (datetime.utcnow() - phase_start).total_seconds()
        efficiency = self.calculate_efficiency_score()
        
        logger.info(f"Phase {phase.value} completed in {phase_duration:.2f}s, efficiency: {efficiency:.1f}%")
        
        await self._log_phase_completion_optimized(phase, phase_start, enhanced_results, adk_results)
    
    async def _execute_enhanced_agents_concurrent(self, phase: SimulationPhase) -> Dict[str, AgentExecutionResult]:
        """Execute enhanced agents concurrently with intelligent batching"""
        start_time = time.time()
        
        priority_agents = ['hospital_load', 'public_behavior']
        secondary_agents = ['social_media', 'news_simulation']
        
        results = {}
        
        # Execute priority agents first
        for agent_name in priority_agents:
            if agent_name in self.enhanced_agents:
                try:
                    task = asyncio.create_task(
                        self._execute_single_agent_resilient(
                            agent_name, self.enhanced_agents[agent_name], phase, 'enhanced'
                        )
                    )
                    result = await asyncio.wait_for(task, timeout=self.agent_timeout)
                    results[agent_name] = result
                except Exception as e:
                    logger.error(f"Enhanced agent {agent_name} failed: {e}")
                    results[agent_name] = AgentExecutionResult(
                        agent_name, 'enhanced', 'failed', 0, {}, str(e)
                    )
        
        # Update context and execute secondary agents
        self._update_simulation_context_optimized(results)
        
        for agent_name in secondary_agents:
            if agent_name in self.enhanced_agents:
                try:
                    task = asyncio.create_task(
                        self._execute_single_agent_resilient(
                            agent_name, self.enhanced_agents[agent_name], phase, 'enhanced'
                        )
                    )
                    result = await asyncio.wait_for(task, timeout=self.agent_timeout)
                    results[agent_name] = result
                except Exception as e:
                    logger.error(f"Enhanced agent {agent_name} failed: {e}")
                    results[agent_name] = AgentExecutionResult(
                        agent_name, 'enhanced', 'failed', 0, {}, str(e)
                    )
        
        execution_time = time.time() - start_time
        self.performance_metrics['total_execution_time'] += execution_time
        self.performance_metrics['concurrent_executions'] += 1
        
        logger.info(f"Enhanced agents completed in {execution_time:.2f}s")
        return results
    
    async def _execute_adk_agents_concurrent(self, phase: SimulationPhase) -> Dict[str, AgentExecutionResult]:
        """Execute ADK agents with intelligent batching"""
        start_time = time.time()
        
        agent_batches = [
            ['emergency_response', 'public_health'],
            ['infrastructure_manager', 'logistics_coordinator'],  
            ['communications_director', 'recovery_coordinator']
        ]
        
        all_results = {}
        
        for batch in agent_batches:
            batch_tasks = []
            for agent_name in batch:
                if agent_name in self.adk_agents:
                    task = asyncio.create_task(
                        self._execute_single_agent_resilient(
                            agent_name, self.adk_agents[agent_name], phase, 'adk'
                        )
                    )
                    batch_tasks.append((agent_name, task))
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(
                *[task for _, task in batch_tasks],
                return_exceptions=True
            )
            
            # Process results
            for i, (agent_name, _) in enumerate(batch_tasks):
                result = batch_results[i]
                if isinstance(result, Exception):
                    logger.error(f"ADK agent {agent_name} failed: {result}")
                    all_results[agent_name] = AgentExecutionResult(
                        agent_name, 'adk', 'failed', 0, {}, str(result)
                    )
                else:
                    all_results[agent_name] = result
            
            await asyncio.sleep(0.5)  # Small delay between batches
        
        execution_time = time.time() - start_time
        self.performance_metrics['total_execution_time'] += execution_time
        
        logger.info(f"ADK agents completed in {execution_time:.2f}s")
        return all_results
    
    async def _execute_single_agent_resilient(self, agent_name: str, agent_instance, 
                                            phase: SimulationPhase, agent_type: str) -> AgentExecutionResult:
        """Execute single agent with error handling and performance tracking"""
        start_time = time.time()
        
        try:
            # Check if agent needs restart
            if self.agent_statuses.get(agent_name) == "restart_pending":
                await self._restart_agent(agent_name, agent_instance)
            
            # Execute agent
            context_copy = dict(self.simulation_context.items())
            result = await agent_instance.process_phase(phase, context_copy)
            
            execution_time = time.time() - start_time
            
            # Calculate efficiency
            target_time = 15.0 if agent_type == 'enhanced' else 20.0
            efficiency = max(0, min(100, 100 - ((execution_time - target_time) / target_time * 50)))
            
            self.agent_restart_attempts[agent_name] = 0
            
            return AgentExecutionResult(
                agent_name=agent_name,
                agent_type=agent_type,
                status='completed',
                execution_time=execution_time,
                result_data=result,
                efficiency_score=efficiency
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Agent {agent_name} execution failed: {e}")
            
            self.agent_statuses[agent_name] = "restart_pending"
            
            return AgentExecutionResult(
                agent_name=agent_name,
                agent_type=agent_type,
                status='failed',
                execution_time=execution_time,
                result_data={},
                error=str(e),
                efficiency_score=0
            )
    
    async def _restart_agent(self, agent_name: str, agent_instance):
        """Restart a failed agent"""
        restart_count = self.agent_restart_attempts.get(agent_name, 0)
        
        if restart_count >= self.max_restart_attempts:
            logger.error(f"Agent {agent_name} exceeded maximum restart attempts")
            self.agent_statuses[agent_name] = "permanently_failed"
            return
        
        try:
            logger.info(f"Restarting agent {agent_name} (attempt {restart_count + 1})")
            
            await agent_instance.initialize_for_simulation(
                self.simulation_id, self.disaster_type, self.severity, self.location
            )
            
            self.agent_statuses[agent_name] = "restarted"
            self.agent_restart_attempts[agent_name] = restart_count + 1
            
            logger.info(f"Successfully restarted agent {agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to restart agent {agent_name}: {e}")
            self.agent_statuses[agent_name] = "restart_failed"
    
    def _update_simulation_context_optimized(self, results: Dict[str, AgentExecutionResult]):
        """Optimized context update with batch processing"""
        updates = {}
        
        for agent_name, result in results.items():
            if result.status == 'completed' and result.result_data:
                metrics = result.result_data.get('metrics', {})
                
                if agent_name == 'hospital_load' and 'resource_utilization' in metrics:
                    updates['hospital_capacity_utilization'] = metrics['resource_utilization']
                
                elif agent_name == 'public_behavior':
                    if 'panic_index' in metrics:
                        updates['panic_index'] = metrics['panic_index']
                    if 'evacuation_compliance' in metrics:
                        updates['evacuation_compliance'] = metrics['evacuation_compliance']
                
                elif agent_name == 'social_media' and 'activity_level' in metrics:
                    updates['social_media_activity'] = metrics['activity_level']
                
                elif agent_name == 'news_simulation' and 'public_trust' in metrics:
                    updates['official_communication_reach'] = metrics['public_trust']
        
        # Apply updates
        if updates:
            self.simulation_context.update(updates)
            
            # Update infrastructure damage based on phase
            if self.current_phase == SimulationPhase.IMPACT:
                updates['infrastructure_damage'] = min(90, self.severity * 8)
            elif self.current_phase == SimulationPhase.RESPONSE:
                current_damage = self.simulation_context.get('infrastructure_damage', 40)
                updates['infrastructure_damage'] = max(15, current_damage - 20)
            else:  # RECOVERY
                current_damage = self.simulation_context.get('infrastructure_damage', 30)
                updates['infrastructure_damage'] = max(5, current_damage - 25)
            
            self.simulation_context.update(updates)
    
    def calculate_efficiency_score(self) -> float:
        """Calculate overall orchestrator efficiency"""
        if not self.performance_metrics['agent_execution_times']:
            return 95.0
        
        execution_times = list(self.performance_metrics['agent_execution_times'].values())
        avg_execution_time = sum(execution_times) / len(execution_times)
        
        # Speed score (target: <10s average)
        target_time = 10.0
        speed_score = max(0, 100 - ((avg_execution_time - target_time) / target_time * 50))
        
        # Success rate
        total_agents = len(self.agent_statuses)
        successful_agents = len([s for s in self.agent_statuses.values() if 'completed' in s])
        success_rate = (successful_agents / total_agents * 100) if total_agents > 0 else 100
        
        # Weighted efficiency
        efficiency = (speed_score * 0.4) + (success_rate * 0.6)
        
        self.performance_metrics['efficiency_score'] = efficiency
        return min(100, max(0, efficiency))
    
    async def _log_phase_completion_optimized(self, phase: SimulationPhase, phase_start: datetime, 
                                            enhanced_results: Dict, adk_results: Dict):
        """Log completion of simulation phase"""
        duration = (datetime.utcnow() - phase_start).total_seconds()
        
        enhanced_successful = len([r for r in enhanced_results.values() if r.status == 'completed'])
        adk_successful = len([r for r in adk_results.values() if r.status == 'completed'])
        total_agents = len(enhanced_results) + len(adk_results)
        
        phase_data = {
            "phase": phase.value,
            "duration_seconds": duration,
            "enhanced_agents_completed": enhanced_successful,
            "adk_agents_completed": adk_successful,
            "total_agents_executed": total_agents,
            "success_rate": (enhanced_successful + adk_successful) / total_agents * 100 if total_agents > 0 else 0,
            "simulation_context": dict(self.simulation_context.items()),
            "performance_optimizations": True
        }
        
        try:
            await self._save_event(f"phase_{phase.value}_completed", phase_data)
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
        
        final_metrics = await self._collect_final_metrics()
        overall_efficiency = self.calculate_efficiency_score()
        
        simulation_report = {
            "simulation_id": self.simulation_id,
            "status": "completed",
            "completion_time": completion_time.isoformat(),
            "total_duration_seconds": duration.total_seconds(),
            "disaster_info": {
                "type": self.disaster_type,
                "location": self.location,
                "severity": self.severity
            },
            "agent_summary": {
                "adk_agents_count": len(self.adk_agents),
                "enhanced_agents_count": len(self.enhanced_agents),
                "total_agents": len(self.adk_agents) + len(self.enhanced_agents),
                "successful_agents": len([s for s in self.agent_statuses.values() if "completed" in s]),
                "failed_agents": len([s for s in self.agent_statuses.values() if "failed" in s])
            },
            "performance_metrics": {
                "overall_efficiency": overall_efficiency,
                "total_execution_time": self.performance_metrics['total_execution_time'],
                "concurrent_executions": self.performance_metrics['concurrent_executions'],
                "agent_execution_times": self.performance_metrics['agent_execution_times'],
                "phase_times": self.performance_metrics['phase_times']
            },
            "agent_statuses": self.agent_statuses,
            "final_simulation_context": dict(self.simulation_context.items()),
            "final_metrics": final_metrics,
            "optimizations_used": True
        }
        
        await self._save_simulation_state(simulation_report)
        await self._cleanup_resources()
        
        logger.info(f"Optimized simulation {self.simulation_id} completed in {duration.total_seconds():.1f}s with {overall_efficiency:.1f}% efficiency")
    
    async def _collect_final_metrics(self) -> Dict[str, Any]:
        """Collect final metrics from all agents"""
        metrics = {
            "adk_agents": {},
            "enhanced_agents": {},
            "simulation_summary": {
                "total_agents": len(self.adk_agents) + len(self.enhanced_agents),
                "successful_agents": len([s for s in self.agent_statuses.values() if "completed" in s]),
                "failed_agents": len([s for s in self.agent_statuses.values() if "failed" in s]),
                "efficiency_score": self.calculate_efficiency_score()
            }
        }
        
        # Collect from ADK agents
        for agent_name, agent_instance in self.adk_agents.items():
            try:
                if hasattr(agent_instance, '_generate_metrics'):
                    agent_metrics = await asyncio.wait_for(agent_instance._generate_metrics(), timeout=10)
                    metrics["adk_agents"][agent_name] = agent_metrics
                else:
                    metrics["adk_agents"][agent_name] = {
                        "status": self.agent_statuses.get(agent_name, "unknown"),
                        "execution_time": self.performance_metrics['agent_execution_times'].get(agent_name, 0)
                    }
            except Exception as e:
                metrics["adk_agents"][agent_name] = {"error": str(e)}
        
        # Collect from enhanced agents
        for agent_name, agent_instance in self.enhanced_agents.items():
            try:
                agent_metrics = {}
                if hasattr(agent_instance, '_generate_hospital_metrics'):
                    agent_metrics = await asyncio.wait_for(agent_instance._generate_hospital_metrics(), timeout=10)
                elif hasattr(agent_instance, '_generate_behavior_metrics'):
                    agent_metrics = await asyncio.wait_for(agent_instance._generate_behavior_metrics(), timeout=10)
                elif hasattr(agent_instance, '_generate_metrics'):
                    agent_metrics = await asyncio.wait_for(agent_instance._generate_metrics(), timeout=10)
                else:
                    agent_metrics = {
                        "status": self.agent_statuses.get(agent_name, "unknown"),
                        "execution_time": self.performance_metrics['agent_execution_times'].get(agent_name, 0)
                    }
                
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
        
        await self._save_simulation_state({
            "status": "failed",
            "error": error_data
        })
        
        logger.error(f"Simulation {self.simulation_id} failed: {error}")
    
    async def _save_simulation_state(self, data: Dict[str, Any]):
        """Save simulation state to cloud storage"""
        try:
            await self.cloud.firestore.save_simulation_state(self.simulation_id, data)
        except Exception as e:
            logger.warning(f"Failed to save simulation state: {e}")
    
    async def _save_event(self, event_type: str, event_data: Dict[str, Any]):
        """Save event to cloud storage"""
        try:
            await self.cloud.firestore.log_event(self.simulation_id, event_type, event_data)
        except Exception as e:
            logger.warning(f"Failed to save event: {e}")
    
    async def _cleanup_resources(self):
        """Clean up resources and perform garbage collection"""
        try:
            # Clear agent references
            self.adk_agents.clear()
            self.enhanced_agents.clear()
            
            # Clear context
            self.simulation_context.clear()
            
            # Shutdown executor
            if self.executor:
                self.executor.shutdown(wait=False)
            
            # Force garbage collection
            gc.collect()
            
            logger.info(f"Resources cleaned up for simulation {self.simulation_id}")
            
        except Exception as e:
            logger.warning(f"Error during resource cleanup: {e}")
    
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
            "simulation_context": dict(self.simulation_context.items()),
            "performance_metrics": self.performance_metrics,
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
                    "agent_id": getattr(agent, 'agent_id', f"adk_{name}"),
                    "agent_type": getattr(agent, 'agent_type', 'adk'),
                    "status": self.agent_statuses.get(name, "unknown"),
                    "restart_attempts": self.agent_restart_attempts.get(name, 0),
                    "execution_time": self.performance_metrics['agent_execution_times'].get(name, 0)
                }
                for name, agent in self.adk_agents.items()
            },
            "enhanced_agents": {
                name: {
                    "agent_id": getattr(agent, 'agent_id', f"enhanced_{name}"),
                    "agent_type": getattr(agent, 'agent_type', 'enhanced'),
                    "status": self.agent_statuses.get(name, "unknown"),
                    "restart_attempts": self.agent_restart_attempts.get(name, 0),
                    "execution_time": self.performance_metrics['agent_execution_times'].get(name, 0)
                }
                for name, agent in self.enhanced_agents.items()
            },
            "total_agents": len(self.adk_agents) + len(self.enhanced_agents),
            "simulation_context": dict(self.simulation_context.items()),
            "performance_summary": {
                "efficiency_score": self.calculate_efficiency_score(),
                "total_execution_time": self.performance_metrics['total_execution_time'],
                "concurrent_executions": self.performance_metrics['concurrent_executions'],
                "phase_times": self.performance_metrics['phase_times']
            }
        }
