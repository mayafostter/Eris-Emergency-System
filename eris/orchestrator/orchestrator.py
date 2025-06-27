"""
ERIS Orchestrator
"""

import asyncio
import logging
import gc
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

# Google ADK imports
try:
    from google.adk.agents import Agent as LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    logging.warning("Google ADK not available, using mock implementations")

# ERIS imports
from services import get_cloud_services
from config import ERISConfig

# Agent imports with fallback handling
try:
    from agents.base_agent import (
        create_emergency_response_agent,
        create_public_health_agent,
        create_infrastructure_manager_agent,
        create_logistics_coordinator_agent,
        create_communications_director_agent,
        create_recovery_coordinator_agent
    )
    from agents.hospital_load_agent import create_hospital_load_agent
    from agents.public_behavior_agent import create_public_behavior_agent  
    from agents.social_media_agent import create_social_media_agent
    from agents.news_simulation_agent import create_news_simulation_agent
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    logging.warning("Agent modules not available, using mock implementations")

# Time utilities
class SimulationPhase(Enum):
    IMPACT = "impact"
    RESPONSE = "response" 
    RECOVERY = "recovery"

class SimulationTimeManager:
    def __init__(self, duration_hours: int = 72):
        self.duration_hours = duration_hours
        self.start_time = datetime.utcnow()
        self.phase_duration = duration_hours / 3  # Equal phases
    
    def get_current_phase(self) -> SimulationPhase:
        elapsed = (datetime.utcnow() - self.start_time).total_seconds() / 3600
        if elapsed < self.phase_duration:
            return SimulationPhase.IMPACT
        elif elapsed < self.phase_duration * 2:
            return SimulationPhase.RESPONSE
        else:
            return SimulationPhase.RECOVERY

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
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SimulationMetrics:
    """Real-time simulation metrics"""
    timestamp: datetime
    current_phase: str
    infrastructure_damage: float
    hospital_capacity_utilization: float
    panic_index: float
    evacuation_compliance: float
    social_media_activity: float
    news_coverage_intensity: float
    official_communication_reach: float
    resource_allocation_efficiency: float
    public_safety_score: float
    recovery_progress: float
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class OptimizedSimulationContext:
    """Memory-efficient simulation context with automatic cleanup"""
    
    def __init__(self, max_size: int = 100):
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
        if key in self._data:
            self._access_times[key] = datetime.utcnow()
            return self._data[key]
        raise KeyError(key)
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

class MockAgent:
    """Mock agent for fallback when real agents aren't available"""
    
    def __init__(self, agent_name: str, agent_type: str):
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.agent_id = f"{agent_type}_{agent_name}"
    
    async def initialize_for_simulation(self, simulation_id: str, disaster_type: str, 
                                      severity: int, location: str):
        await asyncio.sleep(0.1)  # Simulate initialization
        return True
    
    async def process_phase(self, phase: SimulationPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.5)  # Simulate processing
        
        # Generate realistic mock data based on agent type
        if self.agent_name == 'hospital_load':
            return {
                'metrics': {
                    'resource_utilization': min(95, context.get('hospital_capacity_utilization', 80) + phase.value.__hash__() % 20),
                    'patient_volume': 150 + phase.value.__hash__() % 100,
                    'staff_availability': max(60, 90 - phase.value.__hash__() % 30)
                },
                'recommendations': ['Increase emergency staffing', 'Prepare overflow facilities']
            }
        elif self.agent_name == 'public_behavior':
            return {
                'metrics': {
                    'panic_index': max(0, min(1, 0.3 + (phase.value.__hash__() % 100) / 100 * 0.4)),
                    'evacuation_compliance': max(0.4, min(1, 0.8 - (phase.value.__hash__() % 100) / 100 * 0.3)),
                    'social_cohesion': max(0.3, min(1, 0.7 - (phase.value.__hash__() % 100) / 100 * 0.2))
                },
                'behavioral_patterns': ['Increased information seeking', 'Community support activation']
            }
        elif self.agent_name == 'social_media':
            return {
                'metrics': {
                    'activity_level': min(10, 3 + (phase.value.__hash__() % 100) / 100 * 5),
                    'misinformation_spread': max(0, min(1, 0.2 + (phase.value.__hash__() % 100) / 100 * 0.3)),
                    'sentiment_score': max(-1, min(1, -0.2 + (phase.value.__hash__() % 100) / 100 * 0.6))
                },
                'trending_topics': ['#DisasterResponse', '#SafetyFirst', '#CommunitySupport']
            }
        elif self.agent_name == 'news_simulation':
            return {
                'metrics': {
                    'public_trust': max(0.4, min(1, 0.7 + (phase.value.__hash__() % 100) / 100 * 0.2)),
                    'coverage_intensity': min(10, 4 + (phase.value.__hash__() % 100) / 100 * 4),
                    'accuracy_score': max(0.6, min(1, 0.8 + (phase.value.__hash__() % 100) / 100 * 0.1))
                },
                'breaking_news': ['Emergency services deployed', 'Evacuation centers opened']
            }
        else:
            # Generic ADK agent response
            return {
                'metrics': {
                    'response_effectiveness': max(0.5, min(1, 0.8 + (phase.value.__hash__() % 100) / 100 * 0.15)),
                    'resource_efficiency': max(0.4, min(1, 0.75 + (phase.value.__hash__() % 100) / 100 * 0.2))
                },
                'actions_taken': [f'{self.agent_name} deployed standard protocols', 'Monitoring situation']
            }

class ERISOrchestrator:
    """
    Production-ready ERIS Orchestrator managing 10 agents total:
    - 6 ADK agents (Google ADK framework)
    - 4 Enhanced agents (specialized implementations)
    
    Features:
    - Concurrent agent execution with intelligent batching
    - Memory optimization and automatic cleanup
    - Comprehensive error recovery and agent restart
    - Real-time performance monitoring and metrics
    - WebSocket streaming support
    - Fallback mode for development/testing
    """
    
    def __init__(self, simulation_id: str, disaster_type: str, location: str, 
                 severity: int, duration: int = 72):
        # Core simulation configuration
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.location = location
        self.disaster_severity = severity
        self.severity = severity
        self.duration_hours = duration
        
        # Infrastructure initialization with error handling
        try:
            self.cloud = get_cloud_services()
            self.config = ERISConfig()
            self.cloud_available = True
        except Exception as e:
            logger.warning(f"Cloud services initialization failed: {e}, using fallback mode")
            self.cloud = self._create_fallback_services()
            self.config = self._create_fallback_config()
            self.cloud_available = False
        
        # Time management
        self.time_manager = SimulationTimeManager(duration_hours=duration)
        self.current_phase = SimulationPhase.IMPACT
        self.simulation_start_time = datetime.utcnow()
        self.last_phase_change = datetime.utcnow()
        
        # ADK infrastructure
        if ADK_AVAILABLE:
            self.session_service = InMemorySessionService()
        else:
            self.session_service = None
        
        # Agent management
        self.adk_agents: Dict[str, Any] = {}
        self.enhanced_agents: Dict[str, Any] = {}
        self.agent_statuses: Dict[str, str] = {}
        self.agent_restart_attempts: Dict[str, int] = {}
        self.max_restart_attempts = 3
        
        # Performance optimization
        self.max_concurrent_agents = 8
        self.agent_timeout = 35
        self.executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="ERIS-Agent")
        
        # Performance tracking
        self.performance_metrics = {
            'total_execution_time': 0,
            'agent_execution_times': {},
            'concurrent_executions': 0,
            'efficiency_score': 95.0,
            'phase_times': {},
            'successful_operations': 0,
            'failed_operations': 0,
            'restart_count': 0
        }
        
        # Memory-optimized simulation context
        self.simulation_context = OptimizedSimulationContext(max_size=150)
        self._initialize_simulation_context()
        
        # Real-time metrics for WebSocket streaming
        self.current_metrics = None
        self.metrics_history: List[SimulationMetrics] = []
        self.max_metrics_history = 1000
        
        # State management
        self.is_running = False
        self.is_paused = False
        self.is_completed = False
        self.error_state = None
        
        # WebSocket callbacks
        self.metrics_callbacks: List[callable] = []
        self.status_callbacks: List[callable] = []
        
        logger.info(f"ERISOrchestrator initialized for simulation {simulation_id}")
        logger.info(f"Cloud services: {'Available' if self.cloud_available else 'Fallback mode'}")
        logger.info(f"ADK services: {'Available' if ADK_AVAILABLE else 'Mock mode'}")
    
    def _create_fallback_services(self):
        """Create fallback cloud services for development/testing"""
        class MockFirestore:
            async def save_simulation_state(self, sim_id: str, data: Dict[str, Any]):
                logger.debug(f"Mock: Saving simulation state for {sim_id}")
                await asyncio.sleep(0.01)
            
            async def get_simulation_state(self, sim_id: str) -> Dict[str, Any]:
                return {"status": "active", "phase": "impact"}
            
            async def log_event(self, sim_id: str, event_type: str, data: Dict[str, Any]):
                logger.debug(f"Mock: Logging event {event_type} for {sim_id}")
                await asyncio.sleep(0.01)
        
        class MockVertexAI:
            async def generate_official_statements(self, context: Dict, stage: str, 
                                                 dept: str, type: str) -> str:
                return f"Mock emergency statement from {dept} during {stage}"
        
        class MockBigQuery:
            async def log_simulation_event(self, **kwargs):
                logger.debug(f"Mock: Logging BigQuery event with {len(kwargs)} fields")
                await asyncio.sleep(0.01)
        
        return type('MockCloudServices', (), {
            'firestore': MockFirestore(),
            'vertex_ai': MockVertexAI(),
            'bigquery': MockBigQuery()
        })()
    
    def _create_fallback_config(self):
        """Create fallback configuration"""
        class MockConfig:
            def get_disaster_config(self, disaster_type: str) -> Dict[str, Any]:
                return {
                    "severity_multiplier": 1.0,
                    "duration": 24,
                    "impact_radius": 50,
                    "population_density": "high"
                }
        
        return MockConfig()
    
    def _initialize_simulation_context(self):
        """Initialize simulation context with baseline values"""
        baseline_context = {
            # Core disaster info
            'disaster_type': self.disaster_type,
            'location': self.location,
            'severity': self.severity,
            'simulation_start': self.simulation_start_time.isoformat(),
            
            # Infrastructure metrics
            'infrastructure_damage': self._calculate_initial_damage(),
            'power_grid_status': max(20, 100 - self.severity * 8),
            'transportation_disruption': min(90, self.severity * 10),
            'communication_networks': max(40, 100 - self.severity * 6),
            
            # Healthcare system
            'hospital_capacity_utilization': 75 + self.severity * 2,
            'medical_supply_status': max(30, 100 - self.severity * 7),
            'ambulance_availability': max(40, 100 - self.severity * 8),
            
            # Population dynamics
            'total_population': self._estimate_population(),
            'affected_population': int(self._estimate_population() * (self.severity / 10)),
            'evacuation_compliance': 0.7,
            'panic_index': 0.2 + (self.severity / 10) * 0.3,
            
            # Communication and media
            'social_media_activity': 3.0 + self.severity * 0.5,
            'news_coverage_intensity': 2.0 + self.severity * 0.8,
            'official_communication_reach': 0.8,
            'misinformation_spread': 0.15 + (self.severity / 10) * 0.2,
            
            # Resource management
            'emergency_supplies_available': max(40, 100 - self.severity * 5),
            'shelter_capacity_utilization': 20 + self.severity * 3,
            'supply_chain_disrupted': self.severity >= 6,
            
            # Response effectiveness
            'first_responder_deployment': 85,
            'inter_agency_coordination': 0.75,
            'public_safety_score': max(30, 90 - self.severity * 6),
            'recovery_progress': 0.0
        }
        
        self.simulation_context.update(baseline_context)
    
    def _calculate_initial_damage(self) -> float:
        """Calculate initial infrastructure damage based on severity"""
        base_damage = self.severity * 8
        location_modifier = 1.0
        
        # Location-based modifiers (simplified)
        if 'coast' in self.location.lower() or 'beach' in self.location.lower():
            location_modifier = 1.2
        elif 'mountain' in self.location.lower() or 'rural' in self.location.lower():
            location_modifier = 0.8
        
        return min(95, base_damage * location_modifier)
    
    def _estimate_population(self) -> int:
        """Estimate population based on location"""
        # Simplified population estimation
        base_population = 100000
        
        if 'city' in self.location.lower() or 'metro' in self.location.lower():
            return base_population * 5
        elif 'town' in self.location.lower():
            return base_population * 2
        else:
            return base_population
    
    # Core orchestration methods
    
    async def start_simulation(self) -> Dict[str, Any]:
        """Start the complete simulation with all phases"""
        try:
            logger.info(f"Starting ERIS simulation {self.simulation_id}")
            
            if self.is_running:
                raise Exception("Simulation is already running")
            
            self.is_running = True
            self.simulation_start_time = datetime.utcnow()
            
            # Initialize all agents
            await self.initialize_all_agents()
            
            # Save initial state
            await self._save_simulation_state({
                "status": "running",
                "current_phase": self.current_phase.value,
                "adk_agents_count": len(self.adk_agents),
                "enhanced_agents_count": len(self.enhanced_agents),
                "total_agents": len(self.adk_agents) + len(self.enhanced_agents),
                "simulation_start": self.simulation_start_time.isoformat(),
                "cloud_mode": "live" if self.cloud_available else "fallback",
                "adk_mode": "live" if ADK_AVAILABLE else "mock"
            })
            
            # Execute all simulation phases
            for phase in [SimulationPhase.IMPACT, SimulationPhase.RESPONSE, SimulationPhase.RECOVERY]:
                if not self.is_running:
                    break
                    
                phase_start = datetime.utcnow()
                await self._execute_simulation_phase(phase)
                
                phase_duration = (datetime.utcnow() - phase_start).total_seconds()
                self.performance_metrics['phase_times'][phase.value] = phase_duration
                
                # Brief pause between phases
                await asyncio.sleep(2)
            
            # Complete simulation
            if self.is_running:
                await self._complete_simulation()
            
            return self.get_simulation_status()
            
        except Exception as e:
            logger.error(f"Simulation {self.simulation_id} failed: {e}")
            await self._handle_simulation_error(e)
            raise
    
    async def initialize_all_agents(self):
        """Initialize both ADK and enhanced agents with error handling"""
        try:
            logger.info("Initializing all agents...")
            
            # Initialize agents concurrently
            init_tasks = []
            
            # ADK agents initialization
            if AGENTS_AVAILABLE and ADK_AVAILABLE:
                init_tasks.append(self._initialize_adk_agents())
            else:
                init_tasks.append(self._initialize_mock_adk_agents())
            
            # Enhanced agents initialization
            if AGENTS_AVAILABLE:
                init_tasks.append(self._initialize_enhanced_agents())
            else:
                init_tasks.append(self._initialize_mock_enhanced_agents())
            
            # Execute initialization tasks
            results = await asyncio.gather(*init_tasks, return_exceptions=True)
            
            # Check results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Agent initialization task {i} failed: {result}")
            
            total_agents = len(self.adk_agents) + len(self.enhanced_agents)
            successful_agents = len([s for s in self.agent_statuses.values() if s == "initialized"])
            
            logger.info(f"Agent initialization complete: {successful_agents}/{total_agents} agents ready")
            
            if total_agents < 6:  # Minimum viable agents
                raise Exception(f"Insufficient agents initialized: {total_agents}/10 minimum")
                
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
        
        tasks = []
        for agent_name, creator_func in adk_agent_creators.items():
            task = self._initialize_single_adk_agent(agent_name, creator_func)
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
            # Create mock agent as fallback
            self.adk_agents[agent_name] = MockAgent(agent_name, "adk")
            await self.adk_agents[agent_name].initialize_for_simulation(
                self.simulation_id, self.disaster_type, self.severity, self.location
            )
            self.agent_statuses[agent_name] = "initialized_mock"
    
    async def _initialize_enhanced_agents(self):
        """Initialize the 4 enhanced agents"""
        enhanced_agent_creators = {
            'hospital_load': create_hospital_load_agent,
            'public_behavior': create_public_behavior_agent,
            'social_media': create_social_media_agent,
            'news_simulation': create_news_simulation_agent
        }
        
        tasks = []
        for agent_name, creator_func in enhanced_agent_creators.items():
            task = self._initialize_single_enhanced_agent(agent_name, creator_func)
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
            # Create mock agent as fallback
            self.enhanced_agents[agent_name] = MockAgent(agent_name, "enhanced")
            await self.enhanced_agents[agent_name].initialize_for_simulation(
                self.simulation_id, self.disaster_type, self.severity, self.location
            )
            self.agent_statuses[agent_name] = "initialized_mock"
    
    async def _initialize_mock_adk_agents(self):
        """Initialize mock ADK agents when real ones aren't available"""
        adk_agent_names = [
            'emergency_response', 'public_health', 'infrastructure_manager',
            'logistics_coordinator', 'communications_director', 'recovery_coordinator'
        ]
        
        for agent_name in adk_agent_names:
            try:
                mock_agent = MockAgent(agent_name, "adk")
                await mock_agent.initialize_for_simulation(
                    self.simulation_id, self.disaster_type, self.severity, self.location
                )
                self.adk_agents[agent_name] = mock_agent
                self.agent_statuses[agent_name] = "initialized_mock"
                self.agent_restart_attempts[agent_name] = 0
                logger.info(f"Mock ADK agent {agent_name} initialized")
            except Exception as e:
                logger.error(f"Failed to initialize mock ADK agent {agent_name}: {e}")
                self.agent_statuses[agent_name] = "failed"
    
    async def _initialize_mock_enhanced_agents(self):
        """Initialize mock enhanced agents when real ones aren't available"""
        enhanced_agent_names = ['hospital_load', 'public_behavior', 'social_media', 'news_simulation']
        
        for agent_name in enhanced_agent_names:
            try:
                mock_agent = MockAgent(agent_name, "enhanced")
                await mock_agent.initialize_for_simulation(
                    self.simulation_id, self.disaster_type, self.severity, self.location
                )
                self.enhanced_agents[agent_name] = mock_agent
                self.agent_statuses[agent_name] = "initialized_mock"
                self.agent_restart_attempts[agent_name] = 0
                logger.info(f"Mock enhanced agent {agent_name} initialized")
            except Exception as e:
                logger.error(f"Failed to initialize mock enhanced agent {agent_name}: {e}")
                self.agent_statuses[agent_name] = "failed"
    
    async def _execute_simulation_phase(self, phase: SimulationPhase):
        """Execute all agents for a specific simulation phase"""
        self.current_phase = phase
        self.last_phase_change = datetime.utcnow()
        phase_start = datetime.utcnow()
        
        logger.info(f"Executing simulation phase: {phase.value}")
        
        try:
            # Execute enhanced agents first (they provide critical context)
            enhanced_results = await self._execute_enhanced_agents_batch(phase)
            
            # Update simulation context with enhanced agent results
            self._update_simulation_context_from_results(enhanced_results)
            
            # Execute ADK agents with updated context
            adk_results = await self._execute_adk_agents_batch(phase)
            
            # Final context update
            self._update_simulation_context_from_results(adk_results)
            
            # Update agent statuses
            all_results = {**enhanced_results, **adk_results}
            for name, result in all_results.items():
                if result.status == 'completed':
                    self.agent_statuses[name] = f"completed_{phase.value}"
                    self.performance_metrics['successful_operations'] += 1
                else:
                    self.agent_statuses[name] = f"failed_{phase.value}"
                    self.performance_metrics['failed_operations'] += 1
                
                if result.execution_time > 0:
                    self.performance_metrics['agent_execution_times'][name] = result.execution_time
            
            # Calculate and update metrics
            await self._calculate_real_time_metrics()
            
            # Log phase completion
            phase_duration = (datetime.utcnow() - phase_start).total_seconds()
            efficiency = self.calculate_efficiency_score()
            
            logger.info(f"Phase {phase.value} completed in {phase_duration:.2f}s, efficiency: {efficiency:.1f}%")
            
            await self._log_phase_completion(phase, phase_start, enhanced_results, adk_results)
            
            # Notify callbacks about phase completion
            await self._notify_status_callbacks()
            
        except Exception as e:
            logger.error(f"Phase {phase.value} execution failed: {e}")
            raise
    
    async def _execute_enhanced_agents_batch(self, phase: SimulationPhase) -> Dict[str, AgentExecutionResult]:
        """Execute enhanced agents in optimized batches"""
        start_time = time.time()
        results = {}
        
        # Priority batch: Hospital and behavior agents (provide critical context)
        priority_agents = ['hospital_load', 'public_behavior']
        priority_tasks = []
        
        for agent_name in priority_agents:
            if agent_name in self.enhanced_agents:
                task = self._execute_single_agent_with_retry(
                    agent_name, self.enhanced_agents[agent_name], phase, 'enhanced'
                )
                priority_tasks.append((agent_name, task))
        
        # Execute priority agents
        if priority_tasks:
            priority_results = await asyncio.gather(
                *[task for _, task in priority_tasks],
                return_exceptions=True
            )
            
            for i, (agent_name, _) in enumerate(priority_tasks):
                result = priority_results[i]
                if isinstance(result, Exception):
                    logger.error(f"Priority enhanced agent {agent_name} failed: {result}")
                    results[agent_name] = AgentExecutionResult(
                        agent_name, 'enhanced', 'failed', 0, {}, str(result)
                    )
                else:
                    results[agent_name] = result
        
        # Update context with priority results
        self._update_simulation_context_from_results(results)
        
        # Secondary batch: Media and news agents
        secondary_agents = ['social_media', 'news_simulation']
        secondary_tasks = []
        
        for agent_name in secondary_agents:
            if agent_name in self.enhanced_agents:
                task = self._execute_single_agent_with_retry(
                    agent_name, self.enhanced_agents[agent_name], phase, 'enhanced'
                )
                secondary_tasks.append((agent_name, task))
        
        # Execute secondary agents
        if secondary_tasks:
            secondary_results = await asyncio.gather(
                *[task for _, task in secondary_tasks],
                return_exceptions=True
            )
            
            for i, (agent_name, _) in enumerate(secondary_tasks):
                result = secondary_results[i]
                if isinstance(result, Exception):
                    logger.error(f"Secondary enhanced agent {agent_name} failed: {result}")
                    results[agent_name] = AgentExecutionResult(
                        agent_name, 'enhanced', 'failed', 0, {}, str(result)
                    )
                else:
                    results[agent_name] = result
        
        execution_time = time.time() - start_time
        self.performance_metrics['total_execution_time'] += execution_time
        self.performance_metrics['concurrent_executions'] += 1
        
        logger.info(f"Enhanced agents batch completed in {execution_time:.2f}s")
        return results
    
    async def _execute_adk_agents_batch(self, phase: SimulationPhase) -> Dict[str, AgentExecutionResult]:
        """Execute ADK agents in intelligent batches"""
        start_time = time.time()
        
        # Organize ADK agents into logical batches
        agent_batches = [
            ['emergency_response', 'public_health'],          # First responders
            ['infrastructure_manager', 'logistics_coordinator'], # Operations
            ['communications_director', 'recovery_coordinator']   # Coordination & recovery
        ]
        
        all_results = {}
        
        for batch_idx, batch in enumerate(agent_batches):
            batch_tasks = []
            
            for agent_name in batch:
                if agent_name in self.adk_agents:
                    task = self._execute_single_agent_with_retry(
                        agent_name, self.adk_agents[agent_name], phase, 'adk'
                    )
                    batch_tasks.append((agent_name, task))
            
            # Execute batch concurrently
            if batch_tasks:
                batch_results = await asyncio.gather(
                    *[task for _, task in batch_tasks],
                    return_exceptions=True
                )
                
                # Process batch results
                for i, (agent_name, _) in enumerate(batch_tasks):
                    result = batch_results[i]
                    if isinstance(result, Exception):
                        logger.error(f"ADK agent {agent_name} failed: {result}")
                        all_results[agent_name] = AgentExecutionResult(
                            agent_name, 'adk', 'failed', 0, {}, str(result)
                        )
                    else:
                        all_results[agent_name] = result
                
                # Small delay between batches for system stability
                if batch_idx < len(agent_batches) - 1:
                    await asyncio.sleep(0.5)
        
        execution_time = time.time() - start_time
        self.performance_metrics['total_execution_time'] += execution_time
        
        logger.info(f"ADK agents batch completed in {execution_time:.2f}s")
        return all_results
    
    async def _execute_single_agent_with_retry(self, agent_name: str, agent_instance, 
                                             phase: SimulationPhase, agent_type: str) -> AgentExecutionResult:
        """Execute single agent with retry logic and performance tracking"""
        start_time = time.time()
        
        for attempt in range(self.max_restart_attempts):
            try:
                # Check if agent needs restart
                if self.agent_statuses.get(agent_name) == "restart_pending":
                    await self._restart_agent(agent_name, agent_instance)
                
                # Execute agent with timeout
                context_copy = self.simulation_context.to_dict()
                
                # Add phase-specific context
                context_copy.update({
                    'current_phase': phase.value,
                    'simulation_time': (datetime.utcnow() - self.simulation_start_time).total_seconds(),
                    'phase_time': (datetime.utcnow() - self.last_phase_change).total_seconds()
                })
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    agent_instance.process_phase(phase, context_copy),
                    timeout=self.agent_timeout
                )
                
                execution_time = time.time() - start_time
                
                # Calculate efficiency score
                target_time = 15.0 if agent_type == 'enhanced' else 20.0
                efficiency = self._calculate_agent_efficiency(execution_time, target_time)
                
                # Reset restart attempts on success
                self.agent_restart_attempts[agent_name] = 0
                
                return AgentExecutionResult(
                    agent_name=agent_name,
                    agent_type=agent_type,
                    status='completed',
                    execution_time=execution_time,
                    result_data=result,
                    efficiency_score=efficiency
                )
                
            except asyncio.TimeoutError:
                logger.warning(f"Agent {agent_name} timed out (attempt {attempt + 1})")
                if attempt < self.max_restart_attempts - 1:
                    self.agent_statuses[agent_name] = "restart_pending"
                    await asyncio.sleep(1)
                    continue
                else:
                    execution_time = time.time() - start_time
                    return AgentExecutionResult(
                        agent_name, agent_type, 'timeout', execution_time, {}, 
                        "Agent execution timed out", 0
                    )
                    
            except Exception as e:
                logger.error(f"Agent {agent_name} execution failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_restart_attempts - 1:
                    self.agent_statuses[agent_name] = "restart_pending"
                    await asyncio.sleep(1)
                    continue
                else:
                    execution_time = time.time() - start_time
                    return AgentExecutionResult(
                        agent_name, agent_type, 'failed', execution_time, {}, str(e), 0
                    )
        
        # Should not reach here, but just in case
        execution_time = time.time() - start_time
        return AgentExecutionResult(
            agent_name, agent_type, 'failed', execution_time, {}, 
            "Maximum retry attempts exceeded", 0
        )
    
    def _calculate_agent_efficiency(self, execution_time: float, target_time: float) -> float:
        """Calculate efficiency score for an agent"""
        if execution_time <= target_time:
            return 100.0
        else:
            # Penalty for exceeding target time
            penalty = ((execution_time - target_time) / target_time) * 50
            return max(0, 100 - penalty)
    
    async def _restart_agent(self, agent_name: str, agent_instance):
        """Restart a failed agent with exponential backoff"""
        restart_count = self.agent_restart_attempts.get(agent_name, 0)
        
        if restart_count >= self.max_restart_attempts:
            logger.error(f"Agent {agent_name} exceeded maximum restart attempts")
            self.agent_statuses[agent_name] = "permanently_failed"
            return
        
        try:
            # Exponential backoff
            delay = min(5, 2 ** restart_count)
            await asyncio.sleep(delay)
            
            logger.info(f"Restarting agent {agent_name} (attempt {restart_count + 1})")
            
            # Reinitialize agent
            await agent_instance.initialize_for_simulation(
                self.simulation_id, self.disaster_type, self.severity, self.location
            )
            
            self.agent_statuses[agent_name] = "restarted"
            self.agent_restart_attempts[agent_name] = restart_count + 1
            self.performance_metrics['restart_count'] += 1
            
            logger.info(f"Successfully restarted agent {agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to restart agent {agent_name}: {e}")
            self.agent_statuses[agent_name] = "restart_failed"
            self.agent_restart_attempts[agent_name] = restart_count + 1
    
    def _update_simulation_context_from_results(self, results: Dict[str, AgentExecutionResult]):
        """Update simulation context based on agent execution results"""
        updates = {}
        
        for agent_name, result in results.items():
            if result.status != 'completed' or not result.result_data:
                continue
                
            metrics = result.result_data.get('metrics', {})
            
            # Hospital Load Agent updates
            if agent_name == 'hospital_load':
                if 'resource_utilization' in metrics:
                    updates['hospital_capacity_utilization'] = metrics['resource_utilization']
                if 'patient_volume' in metrics:
                    updates['patient_volume'] = metrics['patient_volume']
                if 'staff_availability' in metrics:
                    updates['medical_staff_availability'] = metrics['staff_availability']
            
            # Public Behavior Agent updates
            elif agent_name == 'public_behavior':
                if 'panic_index' in metrics:
                    updates['panic_index'] = max(0, min(1, metrics['panic_index']))
                if 'evacuation_compliance' in metrics:
                    updates['evacuation_compliance'] = max(0, min(1, metrics['evacuation_compliance']))
                if 'social_cohesion' in metrics:
                    updates['social_cohesion'] = max(0, min(1, metrics['social_cohesion']))
            
            # Social Media Agent updates
            elif agent_name == 'social_media':
                if 'activity_level' in metrics:
                    updates['social_media_activity'] = metrics['activity_level']
                if 'misinformation_spread' in metrics:
                    updates['misinformation_spread'] = max(0, min(1, metrics['misinformation_spread']))
                if 'sentiment_score' in metrics:
                    updates['social_media_sentiment'] = metrics['sentiment_score']
            
            # News Simulation Agent updates
            elif agent_name == 'news_simulation':
                if 'public_trust' in metrics:
                    updates['official_communication_reach'] = max(0, min(1, metrics['public_trust']))
                if 'coverage_intensity' in metrics:
                    updates['news_coverage_intensity'] = metrics['coverage_intensity']
                if 'accuracy_score' in metrics:
                    updates['news_accuracy'] = metrics['accuracy_score']
            
            # ADK Agent updates
            elif agent_name in self.adk_agents:
                if 'response_effectiveness' in metrics:
                    updates[f'{agent_name}_effectiveness'] = metrics['response_effectiveness']
                if 'resource_efficiency' in metrics:
                    updates[f'{agent_name}_efficiency'] = metrics['resource_efficiency']
        
        # Apply phase-specific context updates
        self._apply_phase_specific_updates(updates)
        
        # Update the context
        if updates:
            self.simulation_context.update(updates)
    
    def _apply_phase_specific_updates(self, updates: Dict[str, Any]):
        """Apply phase-specific updates to simulation context"""
        phase_multiplier = {
            SimulationPhase.IMPACT: 1.0,
            SimulationPhase.RESPONSE: 0.8,
            SimulationPhase.RECOVERY: 0.6
        }.get(self.current_phase, 1.0)
        
        # Infrastructure damage evolution
        if self.current_phase == SimulationPhase.IMPACT:
            updates['infrastructure_damage'] = min(95, self.severity * 8 * phase_multiplier)
        elif self.current_phase == SimulationPhase.RESPONSE:
            current_damage = self.simulation_context.get('infrastructure_damage', 40)
            updates['infrastructure_damage'] = max(10, current_damage - 15)
        else:  # RECOVERY
            current_damage = self.simulation_context.get('infrastructure_damage', 30)
            updates['infrastructure_damage'] = max(5, current_damage - 20)
        
        # Recovery progress
        if self.current_phase == SimulationPhase.RECOVERY:
            elapsed_hours = (datetime.utcnow() - self.simulation_start_time).total_seconds() / 3600
            progress = min(0.8, elapsed_hours / self.duration_hours)
            updates['recovery_progress'] = progress
        
        # Resource allocation efficiency
        successful_agents = len([s for s in self.agent_statuses.values() if 'completed' in s])
        total_agents = len(self.agent_statuses)
        if total_agents > 0:
            updates['resource_allocation_efficiency'] = successful_agents / total_agents
        
        # Public safety score calculation
        panic = self.simulation_context.get('panic_index', 0.3)
        evacuation = self.simulation_context.get('evacuation_compliance', 0.7)
        infrastructure = self.simulation_context.get('infrastructure_damage', 50)
        
        safety_score = max(0, min(100, 
            (100 - infrastructure) * 0.4 + 
            (1 - panic) * 100 * 0.3 + 
            evacuation * 100 * 0.3
        ))
        updates['public_safety_score'] = safety_score
    
    async def _calculate_real_time_metrics(self):
        """Calculate and store real-time simulation metrics"""
        current_time = datetime.utcnow()
        
        # Gather current context values
        context = self.simulation_context.to_dict()
        
        # Create metrics object
        metrics = SimulationMetrics(
            timestamp=current_time,
            current_phase=self.current_phase.value,
            infrastructure_damage=context.get('infrastructure_damage', 0),
            hospital_capacity_utilization=context.get('hospital_capacity_utilization', 80),
            panic_index=context.get('panic_index', 0.2),
            evacuation_compliance=context.get('evacuation_compliance', 0.7),
            social_media_activity=context.get('social_media_activity', 3.0),
            news_coverage_intensity=context.get('news_coverage_intensity', 2.0),
            official_communication_reach=context.get('official_communication_reach', 0.8),
            resource_allocation_efficiency=context.get('resource_allocation_efficiency', 0.8),
            public_safety_score=context.get('public_safety_score', 70),
            recovery_progress=context.get('recovery_progress', 0.0)
        )
        
        # Store current metrics
        self.current_metrics = metrics
        
        # Add to history with size limit
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_metrics_history:
            self.metrics_history = self.metrics_history[-self.max_metrics_history:]
        
        # Notify metrics callbacks (for WebSocket streaming)
        await self._notify_metrics_callbacks(metrics)
    
    def calculate_efficiency_score(self) -> float:
        """Calculate overall orchestrator efficiency score"""
        if not self.performance_metrics['agent_execution_times']:
            return 95.0
        
        # Speed metrics
        execution_times = list(self.performance_metrics['agent_execution_times'].values())
        avg_execution_time = sum(execution_times) / len(execution_times)
        target_time = 17.5  # Average target between enhanced (15s) and ADK (20s)
        speed_score = self._calculate_agent_efficiency(avg_execution_time, target_time)
        
        # Success rate
        total_operations = (self.performance_metrics['successful_operations'] + 
                          self.performance_metrics['failed_operations'])
        if total_operations > 0:
            success_rate = (self.performance_metrics['successful_operations'] / total_operations) * 100
        else:
            success_rate = 100
        
        # Restart penalty
        restart_penalty = min(20, self.performance_metrics['restart_count'] * 5)
        
        # Weighted efficiency calculation
        efficiency = (
            speed_score * 0.4 +
            success_rate * 0.5 +
            max(0, 100 - restart_penalty) * 0.1
        )
        
        self.performance_metrics['efficiency_score'] = efficiency
        return min(100, max(0, efficiency))
    
    async def _complete_simulation(self):
        """Complete the simulation and generate comprehensive final report"""
        if self.is_completed:
            return
            
        completion_time = datetime.utcnow()
        total_duration = completion_time - self.simulation_start_time
        
        logger.info(f"Completing simulation {self.simulation_id}")
        
        # Collect final metrics from all agents
        final_metrics = await self._collect_comprehensive_final_metrics()
        
        # Calculate final efficiency
        overall_efficiency = self.calculate_efficiency_score()
        
        # Generate comprehensive simulation report
        simulation_report = {
            "simulation_id": self.simulation_id,
            "status": "completed",
            "completion_time": completion_time.isoformat(),
            "total_duration_seconds": total_duration.total_seconds(),
            "total_duration_formatted": str(total_duration),
            
            # Disaster information
            "disaster_info": {
                "type": self.disaster_type,
                "location": self.location,
                "severity": self.severity,
                "duration_hours": self.duration_hours
            },
            
            # Agent summary
            "agent_summary": {
                "adk_agents_count": len(self.adk_agents),
                "enhanced_agents_count": len(self.enhanced_agents),
                "total_agents": len(self.adk_agents) + len(self.enhanced_agents),
                "successful_agents": len([s for s in self.agent_statuses.values() if "completed" in s]),
                "failed_agents": len([s for s in self.agent_statuses.values() if "failed" in s]),
                "mock_agents": len([s for s in self.agent_statuses.values() if "mock" in s])
            },
            
            # Performance metrics
            "performance_metrics": {
                "overall_efficiency": overall_efficiency,
                "total_execution_time": self.performance_metrics['total_execution_time'],
                "concurrent_executions": self.performance_metrics['concurrent_executions'],
                "successful_operations": self.performance_metrics['successful_operations'],
                "failed_operations": self.performance_metrics['failed_operations'],
                "restart_count": self.performance_metrics['restart_count'],
                "agent_execution_times": self.performance_metrics['agent_execution_times'],
                "phase_times": self.performance_metrics['phase_times']
            },
            
            # Final state
            "agent_statuses": self.agent_statuses,
            "final_simulation_context": self.simulation_context.to_dict(),
            "final_metrics": final_metrics,
            "current_metrics": self.current_metrics.to_dict() if self.current_metrics else {},
            
            # System info
            "system_info": {
                "cloud_services": "live" if self.cloud_available else "fallback",
                "adk_framework": "live" if ADK_AVAILABLE else "mock",
                "agents_module": "live" if AGENTS_AVAILABLE else "mock",
                "optimizations_enabled": True,
                "metrics_history_size": len(self.metrics_history)
            }
        }
        
        # Save final state
        await self._save_simulation_state(simulation_report)
        
        # Final metrics calculation
        await self._calculate_real_time_metrics()
        
        # Cleanup resources
        await self._cleanup_resources()
        
        # Update flags
        self.is_running = False
        self.is_completed = True
        
        # Notify callbacks
        await self._notify_status_callbacks()
        
        logger.info(f"Simulation {self.simulation_id} completed successfully in {total_duration.total_seconds():.1f}s with {overall_efficiency:.1f}% efficiency")
    
    async def _collect_comprehensive_final_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive final metrics from all agents"""
        metrics = {
            "adk_agents": {},
            "enhanced_agents": {},
            "simulation_summary": {
                "total_agents": len(self.adk_agents) + len(self.enhanced_agents),
                "successful_agents": len([s for s in self.agent_statuses.values() if "completed" in s]),
                "failed_agents": len([s for s in self.agent_statuses.values() if "failed" in s]),
                "efficiency_score": self.calculate_efficiency_score(),
                "phases_completed": len(self.performance_metrics.get('phase_times', {})),
                "total_operations": (self.performance_metrics['successful_operations'] + 
                                   self.performance_metrics['failed_operations'])
            }
        }
        
        # Collect from ADK agents
        for agent_name, agent_instance in self.adk_agents.items():
            try:
                agent_metrics = {}
                
                # Try different metric collection methods
                if hasattr(agent_instance, '_generate_metrics'):
                    agent_metrics = await asyncio.wait_for(
                        agent_instance._generate_metrics(), timeout=10
                    )
                elif hasattr(agent_instance, 'get_final_metrics'):
                    agent_metrics = await asyncio.wait_for(
                        agent_instance.get_final_metrics(), timeout=10
                    )
                else:
                    # Fallback metrics
                    agent_metrics = {
                        "agent_type": "adk",
                        "status": self.agent_statuses.get(agent_name, "unknown"),
                        "execution_time": self.performance_metrics['agent_execution_times'].get(agent_name, 0),
                        "restart_attempts": self.agent_restart_attempts.get(agent_name, 0)
                    }
                
                metrics["adk_agents"][agent_name] = agent_metrics
                
            except Exception as e:
                logger.warning(f"Failed to collect metrics from ADK agent {agent_name}: {e}")
                metrics["adk_agents"][agent_name] = {
                    "error": str(e),
                    "status": self.agent_statuses.get(agent_name, "unknown")
                }
        
        # Collect from enhanced agents
        for agent_name, agent_instance in self.enhanced_agents.items():
            try:
                agent_metrics = {}
                
                # Try different metric collection methods based on agent type
                if hasattr(agent_instance, '_generate_hospital_metrics'):
                    agent_metrics = await asyncio.wait_for(
                        agent_instance._generate_hospital_metrics(), timeout=10
                    )
                elif hasattr(agent_instance, '_generate_behavior_metrics'):
                    agent_metrics = await asyncio.wait_for(
                        agent_instance._generate_behavior_metrics(), timeout=10
                    )
                elif hasattr(agent_instance, '_generate_social_metrics'):
                    agent_metrics = await asyncio.wait_for(
                        agent_instance._generate_social_metrics(), timeout=10
                    )
                elif hasattr(agent_instance, '_generate_news_metrics'):
                    agent_metrics = await asyncio.wait_for(
                        agent_instance._generate_news_metrics(), timeout=10
                    )
                elif hasattr(agent_instance, '_generate_metrics'):
                    agent_metrics = await asyncio.wait_for(
                        agent_instance._generate_metrics(), timeout=10
                    )
                else:
                    # Fallback metrics
                    agent_metrics = {
                        "agent_type": "enhanced",
                        "status": self.agent_statuses.get(agent_name, "unknown"),
                        "execution_time": self.performance_metrics['agent_execution_times'].get(agent_name, 0),
                        "restart_attempts": self.agent_restart_attempts.get(agent_name, 0)
                    }
                
                metrics["enhanced_agents"][agent_name] = agent_metrics
                
            except Exception as e:
                logger.warning(f"Failed to collect metrics from enhanced agent {agent_name}: {e}")
                metrics["enhanced_agents"][agent_name] = {
                    "error": str(e),
                    "status": self.agent_statuses.get(agent_name, "unknown")
                }
        
        return metrics
    
    async def _log_phase_completion(self, phase: SimulationPhase, phase_start: datetime, 
                                   enhanced_results: Dict[str, AgentExecutionResult], 
                                   adk_results: Dict[str, AgentExecutionResult]):
        """Log comprehensive phase completion data"""
        duration = (datetime.utcnow() - phase_start).total_seconds()
        
        enhanced_successful = len([r for r in enhanced_results.values() if r.status == 'completed'])
        adk_successful = len([r for r in adk_results.values() if r.status == 'completed'])
        total_agents = len(enhanced_results) + len(adk_results)
        
        phase_data = {
            "phase": phase.value,
            "duration_seconds": duration,
            "enhanced_agents": {
                "executed": len(enhanced_results),
                "completed": enhanced_successful,
                "failed": len(enhanced_results) - enhanced_successful,
                "agents": {name: result.to_dict() for name, result in enhanced_results.items()}
            },
            "adk_agents": {
                "executed": len(adk_results),
                "completed": adk_successful,
                "failed": len(adk_results) - adk_successful,
                "agents": {name: result.to_dict() for name, result in adk_results.items()}
            },
            "summary": {
                "total_agents_executed": total_agents,
                "total_successful": enhanced_successful + adk_successful,
                "success_rate": ((enhanced_successful + adk_successful) / total_agents * 100) if total_agents > 0 else 0,
                "phase_efficiency": self.calculate_efficiency_score()
            },
            "simulation_context_snapshot": self.simulation_context.to_dict(),
            "current_metrics": self.current_metrics.to_dict() if self.current_metrics else {}
        }
        
        try:
            # Save to cloud services
            await self._save_event(f"phase_{phase.value}_completed", phase_data)
            
            if hasattr(self.cloud, 'bigquery'):
                await self.cloud.bigquery.log_simulation_event(
                    simulation_id=self.simulation_id,
                    event_type=f"phase_{phase.value}_completed",
                    event_data=phase_data
                )
                
        except Exception as e:
            logger.warning(f"Failed to log phase completion to cloud services: {e}")
    
    async def _handle_simulation_error(self, error: Exception):
        """Handle simulation errors with comprehensive logging"""
        self.error_state = str(error)
        
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "current_phase": self.current_phase.value,
            "agent_statuses": self.agent_statuses,
            "performance_metrics": self.performance_metrics,
            "simulation_context": self.simulation_context.to_dict(),
            "failed_at": datetime.utcnow().isoformat(),
            "elapsed_time": (datetime.utcnow() - self.simulation_start_time).total_seconds()
        }
        
        # Save error state
        await self._save_simulation_state({
            "status": "failed",
            "error": error_data,
            "partial_results": {
                "completed_phases": list(self.performance_metrics.get('phase_times', {}).keys()),
                "agent_results": self.agent_statuses
            }
        })
        
        # Update flags
        self.is_running = False
        
        # Notify callbacks
        await self._notify_status_callbacks()
        
        logger.error(f"Simulation {self.simulation_id} failed: {error}")
    
    # State management and monitoring methods
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get comprehensive current simulation status"""
        elapsed_time = datetime.utcnow() - self.simulation_start_time
        
        return {
            "simulation_id": self.simulation_id,
            "status": "completed" if self.is_completed else ("running" if self.is_running else "stopped"),
            "current_phase": self.current_phase.value,
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "is_completed": self.is_completed,
            "error_state": self.error_state,
            
            # Timing
            "elapsed_time_seconds": elapsed_time.total_seconds(),
            "elapsed_time_formatted": str(elapsed_time),
            "simulation_start": self.simulation_start_time.isoformat(),
            "last_phase_change": self.last_phase_change.isoformat(),
            
            # Agent information
            "agents": {
                "adk_agents_count": len(self.adk_agents),
                "enhanced_agents_count": len(self.enhanced_agents),
                "total_agents": len(self.adk_agents) + len(self.enhanced_agents),
                "successful_agents": len([s for s in self.agent_statuses.values() if "completed" in s]),
                "failed_agents": len([s for s in self.agent_statuses.values() if "failed" in s]),
                "agent_statuses": self.agent_statuses
            },
            
            # Performance
            "performance": {
                "efficiency_score": self.calculate_efficiency_score(),
                "total_execution_time": self.performance_metrics['total_execution_time'],
                "successful_operations": self.performance_metrics['successful_operations'],
                "failed_operations": self.performance_metrics['failed_operations'],
                "restart_count": self.performance_metrics['restart_count']
            },
            
            # Disaster information
            "disaster_info": {
                "type": self.disaster_type,
                "location": self.location,
                "severity": self.severity,
                "duration_hours": self.duration_hours
            },
            
            # Current context
            "simulation_context": self.simulation_context.to_dict(),
            "current_metrics": self.current_metrics.to_dict() if self.current_metrics else {},
            
            # System status
            "system_info": {
                "cloud_services": "live" if self.cloud_available else "fallback",
                "adk_framework": "live" if ADK_AVAILABLE else "mock",
                "agents_module": "live" if AGENTS_AVAILABLE else "mock"
            }
        }
    
    def get_all_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive information about all agents"""
        return {
            "adk_agents": {
                name: {
                    "agent_id": getattr(agent, 'agent_id', f"adk_{name}"),
                    "agent_type": getattr(agent, 'agent_type', 'adk'),
                    "agent_name": name,
                    "status": self.agent_statuses.get(name, "unknown"),
                    "restart_attempts": self.agent_restart_attempts.get(name, 0),
                    "execution_time": self.performance_metrics['agent_execution_times'].get(name, 0),
                    "is_mock": isinstance(agent, MockAgent),
                    "last_executed": self._get_agent_last_execution(name)
                }
                for name, agent in self.adk_agents.items()
            },
            "enhanced_agents": {
                name: {
                    "agent_id": getattr(agent, 'agent_id', f"enhanced_{name}"),
                    "agent_type": getattr(agent, 'agent_type', 'enhanced'),
                    "agent_name": name,
                    "status": self.agent_statuses.get(name, "unknown"),
                    "restart_attempts": self.agent_restart_attempts.get(name, 0),
                    "execution_time": self.performance_metrics['agent_execution_times'].get(name, 0),
                    "is_mock": isinstance(agent, MockAgent),
                    "last_executed": self._get_agent_last_execution(name)
                }
                for name, agent in self.enhanced_agents.items()
            },
            "summary": {
                "total_agents": len(self.adk_agents) + len(self.enhanced_agents),
                "adk_count": len(self.adk_agents),
                "enhanced_count": len(self.enhanced_agents),
                "mock_count": len([a for a in list(self.adk_agents.values()) + list(self.enhanced_agents.values()) if isinstance(a, MockAgent)]),
                "successful_count": len([s for s in self.agent_statuses.values() if "completed" in s]),
                "failed_count": len([s for s in self.agent_statuses.values() if "failed" in s])
            },
            "simulation_context": self.simulation_context.to_dict(),
            "performance_summary": {
                "efficiency_score": self.calculate_efficiency_score(),
                "total_execution_time": self.performance_metrics['total_execution_time'],
                "concurrent_executions": self.performance_metrics['concurrent_executions'],
                "phase_times": self.performance_metrics['phase_times'],
                "restart_summary": self.agent_restart_attempts
            }
        }
    
    def _get_agent_last_execution(self, agent_name: str) -> Optional[str]:
        """Get the last execution phase for an agent"""
        status = self.agent_statuses.get(agent_name, "")
        if "_" in status:
            parts = status.split("_")
            if len(parts) >= 2 and parts[-1] in ['impact', 'response', 'recovery']:
                return parts[-1]
        return None
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics for WebSocket streaming"""
        if not self.current_metrics:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": "No metrics available"
            }
        
        return {
            "metrics": self.current_metrics.to_dict(),
            "agent_statuses": self.agent_statuses,
            "performance": {
                "efficiency_score": self.calculate_efficiency_score(),
                "active_agents": len([s for s in self.agent_statuses.values() if "completed" in s or "running" in s]),
                "total_execution_time": self.performance_metrics['total_execution_time']
            },
            "simulation_status": {
                "is_running": self.is_running,
                "current_phase": self.current_phase.value,
                "elapsed_time": (datetime.utcnow() - self.simulation_start_time).total_seconds()
            }
        }
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get historical metrics data"""
        history = self.metrics_history
        if limit:
            history = history[-limit:]
        
        return [metrics.to_dict() for metrics in history]
    
    # WebSocket callback management
    
    def add_metrics_callback(self, callback: callable):
        """Add callback for real-time metrics updates"""
        self.metrics_callbacks.append(callback)
    
    def remove_metrics_callback(self, callback: callable):
        """Remove metrics callback"""
        if callback in self.metrics_callbacks:
            self.metrics_callbacks.remove(callback)
    
    def add_status_callback(self, callback: callable):
        """Add callback for status updates"""
        self.status_callbacks.append(callback)
    
    def remove_status_callback(self, callback: callable):
        """Remove status callback"""
        if callback in self.status_callbacks:
            self.status_callbacks.remove(callback)
    
    async def _notify_metrics_callbacks(self, metrics: SimulationMetrics):
        """Notify all metrics callbacks"""
        for callback in self.metrics_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics.to_dict())
                else:
                    callback(metrics.to_dict())
            except Exception as e:
                logger.warning(f"Metrics callback failed: {e}")
    
    async def _notify_status_callbacks(self):
        """Notify all status callbacks"""
        status = self.get_simulation_status()
        for callback in self.status_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(status)
                else:
                    callback(status)
            except Exception as e:
                logger.warning(f"Status callback failed: {e}")
    
    # Control methods
    
    async def pause_simulation(self) -> bool:
        """Pause the simulation"""
        if not self.is_running or self.is_completed:
            return False
        
        self.is_paused = True
        logger.info(f"Simulation {self.simulation_id} paused")
        await self._notify_status_callbacks()
        return True
    
    async def resume_simulation(self) -> bool:
        """Resume the simulation"""
        if not self.is_running or not self.is_paused or self.is_completed:
            return False
        
        self.is_paused = False
        logger.info(f"Simulation {self.simulation_id} resumed")
        await self._notify_status_callbacks()
        return True
    
    async def stop_simulation(self) -> bool:
        """Stop the simulation"""
        if self.is_completed:
            return False
        
        self.is_running = False
        self.is_paused = False
        logger.info(f"Simulation {self.simulation_id} stopped")
        
        # Save final state
        await self._save_simulation_state({
            "status": "stopped",
            "stopped_at": datetime.utcnow().isoformat(),
            "final_context": self.simulation_context.to_dict(),
            "agent_statuses": self.agent_statuses
        })
        
        await self._notify_status_callbacks()
        return True
    
    # Utility and cleanup methods
    
    async def _save_simulation_state(self, data: Dict[str, Any]):
        """Save simulation state to cloud storage with error handling"""
        try:
            await self.cloud.firestore.save_simulation_state(self.simulation_id, data)
        except Exception as e:
            logger.warning(f"Failed to save simulation state: {e}")
    
    async def _save_event(self, event_type: str, event_data: Dict[str, Any]):
        """Save event to cloud storage with error handling"""
        try:
            await self.cloud.firestore.log_event(self.simulation_id, event_type, event_data)
        except Exception as e:
            logger.warning(f"Failed to save event {event_type}: {e}")
    
    async def _cleanup_resources(self):
        """Clean up resources and perform garbage collection"""
        try:
            logger.info(f"Cleaning up resources for simulation {self.simulation_id}")
            
            # Clear callbacks
            self.metrics_callbacks.clear()
            self.status_callbacks.clear()
            
            # Clear agent references
            self.adk_agents.clear()
            self.enhanced_agents.clear()
            
            # Clear context and metrics
            self.simulation_context.clear()
            self.metrics_history.clear()
            self.current_metrics = None
            
            # Shutdown executor
            if self.executor:
                self.executor.shutdown(wait=False)
                self.executor = None
            
            # Clear session service
            if self.session_service:
                self.session_service = None
            
            # Force garbage collection
            gc.collect()
            
            logger.info(f"Resource cleanup completed for simulation {self.simulation_id}")
            
        except Exception as e:
            logger.warning(f"Error during resource cleanup: {e}")
    
    # Diagnostic and debugging methods
    
    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information"""
        return {
            "simulation_info": {
                "id": self.simulation_id,
                "disaster_type": self.disaster_type,
                "location": self.location,
                "severity": self.severity,
                "duration_hours": self.duration_hours
            },
            "runtime_info": {
                "is_running": self.is_running,
                "is_paused": self.is_paused,
                "is_completed": self.is_completed,
                "current_phase": self.current_phase.value,
                "elapsed_time": (datetime.utcnow() - self.simulation_start_time).total_seconds(),
                "error_state": self.error_state
            },
            "agent_info": {
                "adk_agents": list(self.adk_agents.keys()),
                "enhanced_agents": list(self.enhanced_agents.keys()),
                "agent_statuses": self.agent_statuses,
                "restart_attempts": self.agent_restart_attempts,
                "mock_agents": [name for name, agent in {**self.adk_agents, **self.enhanced_agents}.items() if isinstance(agent, MockAgent)]
            },
            "performance_info": self.performance_metrics,
            "system_info": {
                "cloud_available": self.cloud_available,
                "adk_available": ADK_AVAILABLE,
                "agents_available": AGENTS_AVAILABLE,
                "context_size": len(self.simulation_context.keys()),
                "metrics_history_size": len(self.metrics_history),
                "callbacks_count": {
                    "metrics": len(self.metrics_callbacks),
                    "status": len(self.status_callbacks)
                }
            },
            "memory_info": {
                "max_concurrent_agents": self.max_concurrent_agents,
                "agent_timeout": self.agent_timeout,
                "max_restart_attempts": self.max_restart_attempts,
                "max_metrics_history": self.max_metrics_history
            }
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a detailed performance report"""
        total_operations = (self.performance_metrics['successful_operations'] + 
                          self.performance_metrics['failed_operations'])
        
        return {
            "overall_metrics": {
                "efficiency_score": self.calculate_efficiency_score(),
                "total_execution_time": self.performance_metrics['total_execution_time'],
                "average_agent_time": (
                    sum(self.performance_metrics['agent_execution_times'].values()) / 
                    len(self.performance_metrics['agent_execution_times'])
                ) if self.performance_metrics['agent_execution_times'] else 0,
                "concurrent_executions": self.performance_metrics['concurrent_executions']
            },
            "operation_metrics": {
                "total_operations": total_operations,
                "successful_operations": self.performance_metrics['successful_operations'],
                "failed_operations": self.performance_metrics['failed_operations'],
                "success_rate": (self.performance_metrics['successful_operations'] / total_operations * 100) if total_operations > 0 else 0,
                "restart_count": self.performance_metrics['restart_count']
            },
            "timing_metrics": {
                "phase_times": self.performance_metrics['phase_times'],
                "agent_execution_times": self.performance_metrics['agent_execution_times'],
                "slowest_agent": max(self.performance_metrics['agent_execution_times'].items(), key=lambda x: x[1]) if self.performance_metrics['agent_execution_times'] else None,
                "fastest_agent": min(self.performance_metrics['agent_execution_times'].items(), key=lambda x: x[1]) if self.performance_metrics['agent_execution_times'] else None
            },
            "agent_reliability": {
                agent_name: {
                    "restart_attempts": self.agent_restart_attempts.get(agent_name, 0),
                    "current_status": self.agent_statuses.get(agent_name, "unknown"),
                    "execution_time": self.performance_metrics['agent_execution_times'].get(agent_name, 0)
                }
                for agent_name in {**self.adk_agents, **self.enhanced_agents}.keys()
            }
        }


# Factory function for FastAPI integration
def create_orchestrator(simulation_id: str, disaster_type: str, location: str, 
                       severity: int, duration: int = 72) -> ERISOrchestrator:
    """Factory function to create a new ERIS orchestrator instance"""
    return ERISOrchestrator(simulation_id, disaster_type, location, severity, duration)


# Async context manager for proper resource management
class ERISOrchestrationContext:
    """Async context manager for ERIS orchestrator lifecycle management"""
    
    def __init__(self, simulation_id: str, disaster_type: str, location: str, 
                 severity: int, duration: int = 72):
        self.orchestrator = None
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.location = location
        self.severity = severity
        self.duration = duration
    
    async def __aenter__(self) -> ERISOrchestrator:
        self.orchestrator = ERISOrchestrator(
            self.simulation_id, self.disaster_type, self.location, 
            self.severity, self.duration
        )
        return self.orchestrator
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.orchestrator:
            try:
                await self.orchestrator.stop_simulation()
                await self.orchestrator._cleanup_resources()
            except Exception as e:
                logger.warning(f"Error during orchestrator cleanup: {e}")


# Additional utility classes for advanced features

class SimulationManager:
    """High-level simulation manager for multiple concurrent simulations"""
    
    def __init__(self):
        self.active_simulations: Dict[str, ERISOrchestrator] = {}
        self.simulation_history: List[str] = []
        self.max_concurrent_simulations = 5
    
    async def start_simulation(self, simulation_id: str, disaster_type: str, 
                             location: str, severity: int, duration: int = 72) -> ERISOrchestrator:
        """Start a new simulation"""
        if len(self.active_simulations) >= self.max_concurrent_simulations:
            raise Exception(f"Maximum concurrent simulations ({self.max_concurrent_simulations}) reached")
        
        if simulation_id in self.active_simulations:
            raise Exception(f"Simulation {simulation_id} already exists")
        
        orchestrator = ERISOrchestrator(simulation_id, disaster_type, location, severity, duration)
        self.active_simulations[simulation_id] = orchestrator
        
        # Start simulation in background
        asyncio.create_task(self._run_simulation(simulation_id))
        
        return orchestrator
    
    async def _run_simulation(self, simulation_id: str):
        """Run simulation and handle completion"""
        try:
            orchestrator = self.active_simulations[simulation_id]
            await orchestrator.start_simulation()
        except Exception as e:
            logger.error(f"Simulation {simulation_id} failed: {e}")
        finally:
            # Move to history
            if simulation_id in self.active_simulations:
                del self.active_simulations[simulation_id]
            if simulation_id not in self.simulation_history:
                self.simulation_history.append(simulation_id)
    
    def get_simulation(self, simulation_id: str) -> Optional[ERISOrchestrator]:
        """Get active simulation"""
        return self.active_simulations.get(simulation_id)
    
    def get_all_simulations(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all simulations"""
        return {
            sim_id: orchestrator.get_simulation_status() 
            for sim_id, orchestrator in self.active_simulations.items()
        }
    
    async def stop_simulation(self, simulation_id: str) -> bool:
        """Stop a specific simulation"""
        if simulation_id in self.active_simulations:
            return await self.active_simulations[simulation_id].stop_simulation()
        return False
    
    async def cleanup_all(self):
        """Stop and cleanup all simulations"""
        for orchestrator in self.active_simulations.values():
            try:
                await orchestrator.stop_simulation()
            except Exception as e:
                logger.warning(f"Error stopping simulation: {e}")
        
        self.active_simulations.clear()


# Global simulation manager instance
simulation_manager = SimulationManager()

# Export main classes and functions
__all__ = [
    'ERISOrchestrator',
    'ERISOrchestrationContext', 
    'SimulationManager',
    'SimulationPhase',
    'SimulationMetrics',
    'AgentExecutionResult',
    'create_orchestrator',
    'simulation_manager'
]
