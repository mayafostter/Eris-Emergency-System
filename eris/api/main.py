import uuid
import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, status, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
import os

from services import get_cloud_services
from utils.time_utils import SimulationTimeManager, SimulationPhase
from config import ERISConfig

# Import orchestrator
from orchestrator.orchestrator import ERISOrchestrator

# Import metrics collector
from services.metrics_collector import ERISMetricsCollector

logger = logging.getLogger(__name__)

# Initialize config and cloud services with error handling
try:
    config = ERISConfig()
    logger.info("✅ Config initialized successfully")
except Exception as e:
    logger.warning(f"Config initialization failed: {e}, using defaults")
    config = type('MinimalConfig', (), {
        'get_disaster_config': lambda self, disaster_type: {"severity_multiplier": 1.0, "duration": 24}
    })()

try:
    cloud = get_cloud_services()
    logger.info("✅ Cloud services initialized successfully")
except Exception as e:
    logger.warning(f"Cloud services initialization failed: {e}, using mock mode")
    # Create a minimal cloud fallback
    cloud = type('MinimalCloud', (), {
        'firestore': type('MockFirestore', (), {
            'save_simulation_state': lambda self, sim_id, data: asyncio.sleep(0),
            'get_simulation_state': lambda self, sim_id: {"status": "active"}
        })(),
        'vertex_ai': type('MockVertexAI', (), {
            'generate_official_statements': lambda self, context, stage, dept, type: "Mock emergency statement",
            'generate_social_media_content': lambda self, prompt, context: "Mock social media content"
        })(),
        'bigquery': type('MockBigQuery', (), {
            'log_simulation_event': lambda self, **kwargs: asyncio.sleep(0)
        })()
    })()

# Create FastAPI app
app = FastAPI(
    title="ERIS API", 
    version="0.5.0",  
    description="ERIS Disaster Simulation API - 10 Agent Orchestrator with Gemini 2.0 Flash"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        # Keep these exact Vercel domains
        "https://eris-emergency-system.vercel.app",
        "https://eris-emergency-system-4roj91x8e-mayafostters-projects.vercel.app",
        "https://eris-emergency-system-git-main-mayafostters-projects.vercel.app",
        # Allow all Vercel subdomains for safety
        "https://eris-emergency-system-*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator storage and WebSocket connections
active_orchestrators: Dict[str, ERISOrchestrator] = {}
websocket_connections: Dict[str, List[WebSocket]] = {}

# Pydantic Models
class SimulationRequest(BaseModel):
    disaster_type: str
    location: str
    severity: int = Field(..., ge=1, le=10)
    duration: int = Field(72, ge=24, le=168) 

class AgentUpdateRequest(BaseModel):
    simulation_id: str
    agent_id: str
    agent_type: str
    state_data: Dict[str, Any]

# Dynamic metrics calculator
class DynamicMetricsCalculator:
    def __init__(self, simulation_id: str, orchestrator: ERISOrchestrator):
        self.simulation_id = simulation_id
        self.orchestrator = orchestrator
        self.last_update = datetime.utcnow()
        
    def calculate_real_time_metrics(self) -> Dict[str, Any]:
        """Calculate real-time metrics based on current simulation state"""
        try:
            context = self.orchestrator.simulation_context
            current_time = datetime.utcnow()
            time_elapsed = (current_time - self.last_update).total_seconds() / 60  # minutes
            
            # Base metrics from simulation context
            base_panic = context.get('panic_index', 0.2)
            base_hospital = context.get('hospital_capacity_utilization', 70)
            base_population = context.get('total_population', 175000)
            infrastructure_damage = context.get('infrastructure_damage', 15)
            
            # Apply time-based evolution
            panic_index = self._evolve_panic_index(base_panic, time_elapsed)
            hospital_capacity = self._evolve_hospital_capacity(base_hospital, panic_index, time_elapsed)
            emergency_response = self._calculate_emergency_response(context)
            population_affected = self._calculate_population_affected(base_population, panic_index)
            infrastructure_failures = self._calculate_infrastructure_failures(infrastructure_damage, time_elapsed)
            
            # Calculate alert level
            alert_level = self._determine_alert_level(panic_index, hospital_capacity, infrastructure_failures)
            
            # Get social media influence
            social_influence = self._get_social_media_influence()
            
            return {
                "alert_level": alert_level,
                "panic_index": int(panic_index * 100),
                "hospital_capacity": int(hospital_capacity),
                "population_affected": int(population_affected),
                "infrastructure_failures": infrastructure_failures,
                "emergency_response": emergency_response,
                "public_trust": self._calculate_public_trust(context, social_influence),
                "evacuation_compliance": self._calculate_evacuation_compliance(panic_index, context),
                "timestamp": current_time.isoformat(),
                "social_media_activity": social_influence.get("activity_level", "moderate"),
                "misinformation_spread": social_influence.get("misinformation_level", 0.2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating real-time metrics: {e}")
            return self._get_fallback_metrics()
    
    def _evolve_panic_index(self, base_panic: float, time_elapsed: float) -> float:
        """Evolve panic index over time based on phase and events"""
        phase = self.orchestrator.current_phase
        
        # Panic evolution based on phase
        if phase == SimulationPhase.IMPACT:
            # Panic increases rapidly during impact
            evolution = min(0.9, base_panic + (time_elapsed * 0.05))
        elif phase == SimulationPhase.RESPONSE:
            # Panic stabilizes or slowly decreases during response
            evolution = max(0.1, base_panic - (time_elapsed * 0.02))
        else:  # RECOVERY
            # Panic decreases during recovery
            evolution = max(0.05, base_panic - (time_elapsed * 0.03))
        
        # Add some randomness for realism
        import random
        evolution += random.uniform(-0.05, 0.05)
        return max(0.0, min(1.0, evolution))
    
    def _evolve_hospital_capacity(self, base_capacity: float, panic_index: float, time_elapsed: float) -> float:
        """Evolve hospital capacity based on disaster progression"""
        # Hospital capacity increases with panic and time
        capacity_increase = (panic_index * 30) + (time_elapsed * 2)
        
        # Apply phase-specific modifiers
        phase = self.orchestrator.current_phase
        if phase == SimulationPhase.IMPACT:
            capacity_increase *= 1.5  # Rapid increase during impact
        elif phase == SimulationPhase.RECOVERY:
            capacity_increase *= 0.7  # Slower increase during recovery
        
        new_capacity = base_capacity + capacity_increase
        return min(98, max(45, new_capacity))  # Cap between 45% and 98%
    
    def _calculate_emergency_response(self, context: Dict[str, Any]) -> int:
        """Calculate emergency response efficiency"""
        base_response = 85
        
        # Get emergency response agent data if available
        try:
            if hasattr(self.orchestrator, 'adk_agents') and 'emergency_response' in self.orchestrator.adk_agents:
                emergency_agent = self.orchestrator.adk_agents['emergency_response']
                if hasattr(emergency_agent, 'response_effectiveness'):
                    base_response = int(emergency_agent.response_effectiveness * 100)
        except:
            pass
        
        # Adjust based on infrastructure damage
        infrastructure_damage = context.get('infrastructure_damage', 20)
        response_efficiency = base_response - (infrastructure_damage / 2)
        
        # Add randomness
        import random
        response_efficiency += random.randint(-5, 5)
        
        return max(60, min(98, int(response_efficiency)))
    
    def _calculate_population_affected(self, total_population: int, panic_index: float) -> int:
        """Calculate affected population based on panic and disaster scope"""
        base_affected_rate = 0.15  # 15% base affected rate
        panic_multiplier = 1 + (panic_index * 2)  # Panic can triple the affected rate
        
        affected = int(total_population * base_affected_rate * panic_multiplier)
        return min(int(total_population * 0.8), affected)  # Cap at 80% of population
    
    def _calculate_infrastructure_failures(self, base_damage: float, time_elapsed: float) -> int:
        """Calculate infrastructure failures"""
        # Infrastructure failures accumulate over time
        additional_failures = int(time_elapsed / 30)  # 1 failure every 30 minutes
        
        import random
        base_failures = int(base_damage / 10) + additional_failures + random.randint(0, 2)
        return max(0, min(15, base_failures))  # Cap at 15 failures
    
    def _determine_alert_level(self, panic_index: float, hospital_capacity: float, infrastructure_failures: int) -> str:
        """Determine overall alert level"""
        # Calculate composite risk score
        panic_score = panic_index * 40
        hospital_score = max(0, (hospital_capacity - 80) * 2) if hospital_capacity > 80 else 0
        infrastructure_score = infrastructure_failures * 3
        
        total_score = panic_score + hospital_score + infrastructure_score
        
        if total_score >= 60:
            return "RED"
        elif total_score >= 35:
            return "ORANGE"
        elif total_score >= 15:
            return "YELLOW"
        else:
            return "GREEN"
    
    def _get_social_media_influence(self) -> Dict[str, Any]:
        """Get social media influence metrics"""
        try:
            if hasattr(self.orchestrator, 'enhanced_agents') and 'social_media' in self.orchestrator.enhanced_agents:
                social_agent = self.orchestrator.enhanced_agents['social_media']
                return {
                    "activity_level": "high" if len(getattr(social_agent, 'posts_generated', [])) > 20 else "moderate",
                    "misinformation_level": getattr(social_agent, 'misinformation_level', 0.2),
                    "panic_influence": getattr(social_agent, 'panic_index', 0.3),
                    "viral_topics": getattr(social_agent, 'viral_topics', [])
                }
        except:
            pass
        
        return {
            "activity_level": "moderate",
            "misinformation_level": 0.2,
            "panic_influence": 0.3,
            "viral_topics": []
        }
    
    def _calculate_public_trust(self, context: Dict[str, Any], social_influence: Dict[str, Any]) -> int:
        """Calculate public trust level"""
        base_trust = 75
        
        # Reduce trust based on misinformation
        misinformation_penalty = int(social_influence.get("misinformation_level", 0.2) * 30)
        
        # Adjust based on official communication reach
        communication_reach = context.get('official_communication_reach', 0.7)
        communication_bonus = int((communication_reach - 0.5) * 40)
        
        trust_level = base_trust - misinformation_penalty + communication_bonus
        
        import random
        trust_level += random.randint(-5, 5)
        
        return max(30, min(95, trust_level))
    
    def _calculate_evacuation_compliance(self, panic_index: float, context: Dict[str, Any]) -> int:
        """Calculate evacuation compliance rate"""
        base_compliance = 70
        
        # Higher panic can increase compliance (fear) but also reduce it (chaos)
        if panic_index < 0.3:
            panic_effect = panic_index * 50  # Low panic = low compliance
        elif panic_index < 0.7:
            panic_effect = 30 + (panic_index - 0.3) * 25  # Medium panic = good compliance
        else:
            panic_effect = 40 - (panic_index - 0.7) * 30  # High panic = chaos reduces compliance
        
        compliance = base_compliance + panic_effect
        
        # Adjust based on infrastructure damage
        infrastructure_damage = context.get('infrastructure_damage', 20)
        compliance -= infrastructure_damage / 2  # Damaged infrastructure reduces compliance
        
        import random
        compliance += random.randint(-8, 8)
        
        return max(20, min(95, int(compliance)))
    
    def _get_fallback_metrics(self) -> Dict[str, Any]:
        """Return fallback metrics if calculation fails"""
        import random
        return {
            "alert_level": "YELLOW",
            "panic_index": random.randint(15, 45),
            "hospital_capacity": random.randint(70, 85),
            "population_affected": random.randint(8000, 15000),
            "infrastructure_failures": random.randint(1, 5),
            "emergency_response": random.randint(85, 95),
            "public_trust": random.randint(65, 85),
            "evacuation_compliance": random.randint(70, 90),
            "timestamp": datetime.utcnow().isoformat(),
            "social_media_activity": "moderate",
            "misinformation_spread": 0.2
        }

# WebSocket manager for real-time updates
class WebSocketManager:
    def __init__(self):
        self.connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, simulation_id: str):
        await websocket.accept()
        if simulation_id not in self.connections:
            self.connections[simulation_id] = []
        self.connections[simulation_id].append(websocket)
        logger.info(f"WebSocket connected for simulation {simulation_id}")
    
    def disconnect(self, websocket: WebSocket, simulation_id: str):
        if simulation_id in self.connections:
            if websocket in self.connections[simulation_id]:
                self.connections[simulation_id].remove(websocket)
            if not self.connections[simulation_id]:
                del self.connections[simulation_id]
        logger.info(f"WebSocket disconnected for simulation {simulation_id}")
    
    async def broadcast_to_simulation(self, simulation_id: str, data: dict):
        if simulation_id in self.connections:
            disconnected = []
            for websocket in self.connections[simulation_id]:
                try:
                    await websocket.send_json(data)
                except:
                    disconnected.append(websocket)
            
            # Remove disconnected websockets
            for ws in disconnected:
                self.disconnect(ws, simulation_id)

websocket_manager = WebSocketManager()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow(),
        "version": "0.5.0",  
        "orchestrator_status": "active",
        "total_agents": 10,  
        "adk_agents": 6,
        "enhanced_agents": 4,
        "ai_model": "Gemini 2.0 Flash",  
        "cors_enabled": True,
        "websockets_active": len(websocket_manager.connections)
    }

@app.get("/")
async def root():
    return {
        "message": "ERIS Disaster Simulation API - 10 Agent Orchestrator",
        "version": "0.5.0",  
        "ai_orchestrator": "Gemini 2.0 Flash",
        "agent_architecture": {
            "adk_agents": 6,
            "enhanced_agents": 4,
            "total_agents": 10
        },
        "cors_enabled": True,
        "real_time_features": {
            "dynamic_metrics": True,
            "ai_content_generation": True,
            "websocket_streaming": True,
            "live_social_feed": True
        }
    }

@app.post("/simulate")
async def start_simulation(request: SimulationRequest, background_tasks: BackgroundTasks):
    try:
        simulation_id = str(uuid.uuid4())
        
        logger.info(f"Starting ERIS simulation {simulation_id}: {request.disaster_type} in {request.location}, severity {request.severity}")

        # Validate disaster type
        valid_disasters = ["earthquake", "hurricane", "flood", "tsunami", "wildfire", "volcanic_eruption", "severe_storm", "epidemic", "pandemic", "landslide"]
        
        if hasattr(config, 'get_disaster_config') and callable(getattr(config, 'get_disaster_config')):
            disaster_config = config.get_disaster_config(request.disaster_type)
            if not disaster_config:
                raise HTTPException(status_code=400, detail=f"Unknown disaster type: {request.disaster_type}")
        else:
            if request.disaster_type not in valid_disasters:
                raise HTTPException(status_code=400, detail=f"Unknown disaster type: {request.disaster_type}")

        # Create orchestrator
        orchestrator = ERISOrchestrator(
            simulation_id=simulation_id,
            disaster_type=request.disaster_type,
            location=request.location,
            severity=request.severity,
            duration=request.duration
        )

        # Store orchestrator
        active_orchestrators[simulation_id] = orchestrator

        # Initialize simulation data
        simulation_data = {
            "simulation_id": simulation_id,
            "disaster_type": request.disaster_type,
            "location": request.location,
            "severity": request.severity,
            "duration": request.duration,
            "status": "initializing", 
            "created_at": datetime.utcnow(),
            "current_phase": SimulationPhase.IMPACT.value,
            "orchestrator_version": "0.5.0",
            "total_agents": 10,
            "adk_agents": 6,
            "enhanced_agents": 4,
            "real_time_metrics": True
        }

        # Save simulation state
        try:
            await cloud.firestore.save_simulation_state(simulation_id, simulation_data)
        except Exception as e:
            logger.warning(f"Failed to save to Firestore: {e}")
        
        # Generate initial scenario
        disaster_context = {
            "type": request.disaster_type,
            "location": request.location,
            "severity": request.severity
        }
        
        try:
            scenario = await cloud.vertex_ai.generate_official_statements(
                disaster_context, 
                "initial", 
                "emergency_management", 
                "public_advisory"
            )
        except Exception as e:
            logger.warning(f"Failed to generate scenario: {e}")
            scenario = f"Emergency response activated for {request.disaster_type} in {request.location}. Severity level {request.severity}. ERIS 10-agent orchestrator deployed with real-time AI content generation."

        # Start orchestrated simulation in background
        background_tasks.add_task(run_orchestrated_simulation, orchestrator, simulation_id)
        
        # Start real-time metrics streaming
        background_tasks.add_task(start_real_time_metrics_streaming, simulation_id)
        
        # Start social media content generation
        background_tasks.add_task(start_social_media_generation, simulation_id)

        # Response with information
        response_data = {
            "simulation_id": simulation_id,
            "status": "initializing",
            "message": "ERIS 10-agent orchestrator deployed with real-time AI generation",
            "orchestrator_info": {
                "version": "0.5.0",
                "ai_model": "Gemini 2.0 Flash",
                "total_agents": 10,
                "adk_agents": 6,
                "enhanced_agents": 4,
                "coordination_system": "cross-agent context sharing",
                "real_time_features": {
                    "dynamic_metrics": True,
                    "ai_social_media": True,
                    "live_emergency_feed": True,
                    "websocket_streaming": True
                }
            },
            "frontend_ready": True,
            "data": {
                "disaster_type": request.disaster_type,
                "location": request.location,
                "severity": request.severity,
                "duration": request.duration,
                "scenario": scenario,
                "orchestrator_status": "initializing",
                "expected_metrics_delay_seconds": 10,
                "websocket_endpoint": f"/ws/metrics/{simulation_id}",
                "live_feed_endpoint": f"/live-feed/{simulation_id}"
            }
        }
        
        logger.info(f"ERIS simulation {simulation_id} deployed successfully with real-time features")
        return response_data
        
    except Exception as e:
        logger.error(f"Simulation deployment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Real-time metrics streaming background task
async def start_real_time_metrics_streaming(simulation_id: str):
    """Background task for real-time metrics calculation and streaming"""
    try:
        await asyncio.sleep(5)  # Initial delay for orchestrator setup
        
        while simulation_id in active_orchestrators:
            try:
                orchestrator = active_orchestrators[simulation_id]
                
                # Calculate dynamic metrics
                metrics_calculator = DynamicMetricsCalculator(simulation_id, orchestrator)
                real_time_metrics = metrics_calculator.calculate_real_time_metrics()
                
                # Broadcast to WebSocket connections
                await websocket_manager.broadcast_to_simulation(simulation_id, {
                    "type": "metrics_update",
                    "simulation_id": simulation_id,
                    "dashboard_metrics": real_time_metrics,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Save to Firestore for API access
                try:
                    await cloud.firestore.save_simulation_state(simulation_id, {
                        "last_metrics": real_time_metrics,
                        "last_metrics_update": datetime.utcnow().isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Failed to save metrics to Firestore: {e}")
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Real-time metrics error for {simulation_id}: {e}")
                await asyncio.sleep(10)
                
        logger.info(f"Real-time metrics streaming completed for simulation {simulation_id}")
        
    except Exception as e:
        logger.error(f"Real-time metrics streaming initialization failed: {e}")

# Social media content generation background task
async def start_social_media_generation(simulation_id: str):
    """Background task for continuous social media content generation"""
    try:
        await asyncio.sleep(10)  # Wait for orchestrator initialization
        
        while simulation_id in active_orchestrators:
            try:
                orchestrator = active_orchestrators[simulation_id]
                
                # Generate new social media posts
                if hasattr(orchestrator, 'enhanced_agents') and 'social_media' in orchestrator.enhanced_agents:
                    social_agent = orchestrator.enhanced_agents['social_media']
                    
                    # Generate new posts based on current context
                    new_posts = await social_agent.generate_live_posts(orchestrator.simulation_context)
                    
                    if new_posts:
                        # Broadcast new posts to WebSocket connections
                        await websocket_manager.broadcast_to_simulation(simulation_id, {
                            "type": "social_media_update",
                            "simulation_id": simulation_id,
                            "new_posts": new_posts,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                
                # Generate at random intervals for realism
                import random
                await asyncio.sleep(random.randint(15, 45))
                
            except Exception as e:
                logger.error(f"Social media generation error for {simulation_id}: {e}")
                await asyncio.sleep(30)
                
        logger.info(f"Social media generation completed for simulation {simulation_id}")
        
    except Exception as e:
        logger.error(f"Social media generation initialization failed: {e}")

# WebSocket endpoint with real-time updates
@app.websocket("/ws/metrics/{simulation_id}")
async def websocket_metrics(websocket: WebSocket, simulation_id: str):
    """WebSocket endpoint for real-time metrics and content"""
    await websocket_manager.connect(websocket, simulation_id)
    
    if simulation_id not in active_orchestrators:
        await websocket.send_json({
            "error": "Active simulation not found", 
            "simulation_id": simulation_id,
            "message": "Simulation may be completed or not yet started"
        })
        websocket_manager.disconnect(websocket, simulation_id)
        return
    
    try:
        orchestrator = active_orchestrators[simulation_id]
        
        # Send initial state
        metrics_calculator = DynamicMetricsCalculator(simulation_id, orchestrator)
        initial_metrics = metrics_calculator.calculate_real_time_metrics()
        
        await websocket.send_json({
            "type": "initial_state",
            "simulation_id": simulation_id,
            "dashboard_metrics": initial_metrics,
            "orchestrator_info": {
                "current_phase": orchestrator.current_phase.value,
                "total_agents": len(orchestrator.adk_agents) + len(orchestrator.enhanced_agents),
                "ai_model": "Gemini 2.0 Flash",
                "real_time_enabled": True
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and handle incoming messages
        while simulation_id in active_orchestrators:
            try:
                # Wait for WebSocket message or timeout
                message = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                
                # Handle client requests
                if message.get("type") == "request_update":
                    current_metrics = metrics_calculator.calculate_real_time_metrics()
                    await websocket.send_json({
                        "type": "metrics_update",
                        "simulation_id": simulation_id,
                        "dashboard_metrics": current_metrics,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat()
                })
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        logger.error(f"WebSocket error for {simulation_id}: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        websocket_manager.disconnect(websocket, simulation_id)

# Dashboard metrics endpoint
@app.get("/metrics/dashboard/{simulation_id}")
async def get_dashboard_metrics(simulation_id: str):
    """Get real-time dashboard metrics"""
    if simulation_id not in active_orchestrators:
        # Try to get from Firestore for completed simulations
        try:
            simulation_data = await cloud.firestore.get_simulation_state(simulation_id)
            if simulation_data and "last_metrics" in simulation_data:
                return {
                    "simulation_id": simulation_id,
                    "status": "completed",
                    "dashboard_data": simulation_data["last_metrics"],
                    "timestamp": simulation_data.get("last_metrics_update", datetime.utcnow().isoformat()),
                    "frontend_ready": True
                }
        except Exception as e:
            logger.warning(f"Failed to get stored metrics: {e}")
            
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    orchestrator = active_orchestrators[simulation_id]
    
    try:
        # Calculate real-time metrics
        metrics_calculator = DynamicMetricsCalculator(simulation_id, orchestrator)
        dashboard_data = metrics_calculator.calculate_real_time_metrics()
        
        return {
            "simulation_id": simulation_id,
            "status": "active",
            "dashboard_data": dashboard_data,
            "orchestrator_info": {
                "current_phase": orchestrator.current_phase.value,
                "total_agents": len(orchestrator.adk_agents) + len(orchestrator.enhanced_agents),
                "ai_model": "Gemini 2.0 Flash",
                "coordination_active": True,
                "real_time_enabled": True
            },
            "timestamp": datetime.utcnow().isoformat(),
            "frontend_ready": True
        }
        
    except Exception as e:
        logger.error(f"Dashboard metrics error: {e}")
        # Return safe defaults
        return {
            "simulation_id": simulation_id,
            "status": "active",
            "dashboard_data": {
                "alert_level": "YELLOW",
                "panic_index": 25,
                "hospital_capacity": 78,
                "population_affected": 12000,
                "infrastructure_failures": 2,
                "emergency_response": 88,
                "public_trust": 75,
                "evacuation_compliance": 82
            },
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "frontend_ready": True
        }

# Live emergency feed endpoint
@app.get("/live-feed/{simulation_id}")
async def get_live_emergency_feed(simulation_id: str, limit: int = 20):
    """Get live emergency feed with AI-generated content"""
    if simulation_id not in active_orchestrators:
        raise HTTPException(status_code=404, detail="Active simulation not found")
    
    try:
        orchestrator = active_orchestrators[simulation_id]
        feed_items = []
        
        # Get social media posts
        if hasattr(orchestrator, 'enhanced_agents') and 'social_media' in orchestrator.enhanced_agents:
            social_agent = orchestrator.enhanced_agents['social_media']
            recent_posts = await social_agent.get_recent_posts(limit=limit//2)
            
            for post in recent_posts:
                feed_items.append({
                    "type": "social_media",
                    "source": f"@{post['user_type'].title()}User{hash(post['content']) % 1000}",
                    "content": post['content'],
                    "timestamp": post['timestamp'],
                    "engagement": {
                        "likes": post.get('likes', 0),
                        "shares": post.get('shares', 0),
                        "comments": post.get('comments', 0)
                    },
                    "sentiment": post['sentiment'],
                    "hashtags": post.get('hashtags', [])
                })
        
        # Generate official updates
        official_updates = await generate_official_updates(orchestrator, limit//2)
        feed_items.extend(official_updates)
        
        # Sort by timestamp
        feed_items.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return {
            "simulation_id": simulation_id,
            "feed_items": feed_items[:limit],
            "total_items": len(feed_items),
            "last_updated": datetime.utcnow().isoformat(),
            "ai_generated": True
        }
        
    except Exception as e:
        logger.error(f"Live feed error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_official_updates(orchestrator: ERISOrchestrator, limit: int) -> List[Dict[str, Any]]:
    """Generate official emergency updates based on current simulation state"""
    updates = []
    
    try:
        context = orchestrator.simulation_context
        phase = orchestrator.current_phase.value
        
        # Template official updates based on phase and context
        templates = {
            'impact': [
                {
                    "source": "@EmergencyPhuket",
                    "content": f"🚨 EMERGENCY ALERT: {orchestrator.disaster_type.replace('_', ' ').title()} affecting {orchestrator.location}. Severity Level {orchestrator.disaster_severity}. Follow evacuation orders immediately.",
                    "type": "official_alert"
                },
                {
                    "source": "@PhuketHospital",
                    "content": f"Hospital capacity at {context.get('hospital_capacity_utilization', 75):.0f}%. Emergency medical services active. Non-urgent cases please avoid ER.",
                    "type": "medical_update"
                }
            ],
            'response': [
                {
                    "source": "@ERISSystem",
                    "content": f"Gemini 2.0 Flash orchestrator coordinating {len(orchestrator.adk_agents) + len(orchestrator.enhanced_agents)} AI agents for optimal response. Real-time metrics active.",
                    "type": "system_update"
                },
                {
                    "source": "@PublicHealthTH",
                    "content": f"Public panic index at {context.get('panic_index', 0.3) * 100:.0f}%. Coordinating crowd management. Stay calm and follow official guidance.",
                    "type": "public_health"
                }
            ],
            'recovery': [
                {
                    "source": "@RecoveryCoord",
                    "content": f"Recovery operations underway. Infrastructure damage assessment: {context.get('infrastructure_damage', 20):.0f}% affected. Community support centers open.",
                    "type": "recovery_update"
                }
            ]
        }
        
        phase_templates = templates.get(phase, templates['impact'])
        
        for i, template in enumerate(phase_templates[:limit]):
            if i >= limit:
                break
                
            updates.append({
                "type": "official_update",
                "source": template["source"],
                "content": template["content"],
                "timestamp": (datetime.utcnow() - timedelta(minutes=i*5)).isoformat(),
                "engagement": {
                    "likes": 0,
                    "shares": 0,
                    "comments": 0
                },
                "sentiment": "official",
                "hashtags": [f"#{orchestrator.disaster_type.replace('_', '').title()}", "#Emergency", "#Official"],
                "update_type": template.get("type", "general")
            })
            
    except Exception as e:
        logger.error(f"Error generating official updates: {e}")
    
    return updates

# Background task for orchestrated simulation
async def run_orchestrated_simulation(orchestrator: ERISOrchestrator, simulation_id: str):
    """Background task to run the orchestrated simulation"""
    try:
        logger.info(f"Starting enhanced ERIS orchestrator for simulation {simulation_id}")
        
        # Update status to running
        try:
            await cloud.firestore.save_simulation_state(simulation_id, {
                "status": "running",
                "orchestrator_started": datetime.utcnow(),
                "message": "10-agent orchestrator active with real-time AI generation",
                "real_time_features": True
            })
        except Exception as e:
            logger.warning(f"Failed to update status: {e}")
        
        # Run the orchestrated simulation
        await orchestrator.start_simulation()
        
        logger.info(f"Enhanced ERIS simulation {simulation_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Enhanced ERIS simulation {simulation_id} failed: {e}")
        
        # Update simulation state on failure
        try:
            await cloud.firestore.save_simulation_state(simulation_id, {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.utcnow(),
                "orchestrator_status": "failed"
            })
        except Exception as save_error:
            logger.error(f"Failed to save error state: {save_error}")
    
    finally:
        # Clean up orchestrator and connections
        if simulation_id in active_orchestrators:
            del active_orchestrators[simulation_id]
        if simulation_id in websocket_manager.connections:
            del websocket_manager.connections[simulation_id]
        logger.info(f"Cleaned up enhanced orchestrator for simulation {simulation_id}")

# Status endpoint
@app.get("/status/{simulation_id}")
async def get_simulation_status(simulation_id: str):
    try:
        # Get base simulation data
        try:
            simulation_data = await cloud.firestore.get_simulation_state(simulation_id)
        except Exception as e:
            logger.warning(f"Failed to get simulation data: {e}")
            simulation_data = None
            
        if not simulation_data:
            raise HTTPException(status_code=404, detail="Simulation not found")

        # Get orchestrator status if active
        orchestrator_status = None
        agent_summary = None
        real_time_status = False
        
        if simulation_id in active_orchestrators:
            orchestrator = active_orchestrators[simulation_id]
            real_time_status = True
            try:
                orchestrator_status = orchestrator.get_simulation_status()
                agent_summary = {
                    "total_agents": len(orchestrator.adk_agents) + len(orchestrator.enhanced_agents),
                    "adk_agents": len(orchestrator.adk_agents),
                    "enhanced_agents": len(orchestrator.enhanced_agents),
                    "active_agents": len([s for s in orchestrator.agent_statuses.values() if "completed" in s or "active" in s]),
                    "failed_agents": len([s for s in orchestrator.agent_statuses.values() if "failed" in s])
                }
            except Exception as e:
                logger.warning(f"Failed to get orchestrator status: {e}")

        return {
            "simulation_id": simulation_id,
            "status": simulation_data.get("status", "unknown"),
            "current_phase": simulation_data.get("current_phase", "impact"),
            "last_updated": datetime.utcnow(),
            "orchestrator": {
                "version": "0.5.0",
                "ai_model": "Gemini 2.0 Flash",
                "status": orchestrator_status,
                "agent_summary": agent_summary,
                "real_time_active": real_time_status,
                "websocket_connections": len(websocket_manager.connections.get(simulation_id, []))
            },
            "simulation_data": simulation_data,
            "enhanced_simulation": True,
            "real_time_features": simulation_data.get("real_time_features", False),
            "frontend_accessible": True
        }
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/{simulation_id}")
async def get_simulation_metrics(simulation_id: str):
    try:
        # Get base simulation data
        try:
            simulation_data = await cloud.firestore.get_simulation_state(simulation_id)
        except Exception as e:
            logger.warning(f"Failed to get simulation data: {e}")
            simulation_data = {"status": "active", "current_phase": "impact"}
            
        if not simulation_data:
            raise HTTPException(status_code=404, detail="Simulation not found")

        # Get orchestrator metrics if available
        orchestrator_metrics = {}
        agent_metrics = {"adk_agents": {}, "enhanced_agents": {}}
        real_time_data = None
        
        if simulation_id in active_orchestrators:
            orchestrator = active_orchestrators[simulation_id]
            try:
                # Get real-time metrics
                metrics_calculator = DynamicMetricsCalculator(simulation_id, orchestrator)
                real_time_data = metrics_calculator.calculate_real_time_metrics()
                
                # Collect final metrics from orchestrator
                final_metrics = await orchestrator._collect_final_metrics()
                agent_metrics = final_metrics
                
                orchestrator_metrics = {
                    "current_phase": orchestrator.current_phase.value,
                    "simulation_context": orchestrator.simulation_context,
                    "agent_statuses": orchestrator.agent_statuses,
                    "total_agents": len(orchestrator.adk_agents) + len(orchestrator.enhanced_agents),
                    "coordination_events": len(orchestrator.agent_statuses),
                    "real_time_metrics": real_time_data
                }
            except Exception as e:
                logger.warning(f"Failed to get orchestrator metrics: {e}")

        return {
            "simulation_id": simulation_id,
            "current_phase": simulation_data.get("current_phase", "impact"),
            "orchestrator_metrics": orchestrator_metrics,
            "agent_metrics": agent_metrics,
            "real_time_data": real_time_data,
            "ai_model": "Gemini 2.0 Flash",
            "total_agents": 10,
            "enhanced_metrics_available": True,
            "real_time_enabled": simulation_id in active_orchestrators
        }
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System info endpoint
@app.get("/system/info")
async def get_system_info():
    """Get enhanced system information and capabilities"""
    return {
        "eris_version": "0.5.0",
        "orchestrator": {
            "ai_model": "Gemini 2.0 Flash",
            "architecture": "10-agent coordination system",
            "coordination_method": "cross-agent context sharing"
        },
        "agent_system": {
            "total_agents": 10,
            "adk_agents": 6,
            "enhanced_agents": 4,
            "ai_powered": True
        },
        "real_time_features": {
            "dynamic_metrics_calculation": True,
            "ai_content_generation": True,
            "websocket_streaming": True,
            "live_social_media_simulation": True,
            "contextual_emergency_feed": True,
            "real_time_panic_tracking": True,
            "hospital_capacity_modeling": True,
            "infrastructure_damage_evolution": True
        },
        "frontend_compatible": True,
        "cors_enabled": True,
        "active_simulations": len(active_orchestrators),
        "websocket_connections": sum(len(conns) for conns in websocket_manager.connections.values()),
        "capabilities": {
            "disaster_simulation": True,
            "multi_agent_coordination": True,
            "phase_based_execution": True,
            "real_time_metrics": True,
            "cloud_integration": True,
            "adk_integration": True,
            "hospital_load_modeling": True,
            "public_behavior_simulation": True,
            "social_media_simulation": True,
            "news_coverage_simulation": True,
            "cross_agent_context_sharing": True,
            "ai_orchestration": True,
            "dashboard_metrics": True,
            "websocket_streaming": True,
            "dynamic_content_generation": True
        }
    }

# Ping endpoint
@app.get("/ping")
async def ping():
    """Ping endpoint with real-time status"""
    return {
        "message": "pong",
        "timestamp": datetime.utcnow().isoformat(),
        "orchestrator": "Gemini 2.0 Flash",
        "agents": 10,
        "version": "0.5.0",
        "cors_working": True,
        "server_ready": True,
        "real_time_features": True,
        "active_simulations": len(active_orchestrators),
        "websocket_connections": len(websocket_manager.connections)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
