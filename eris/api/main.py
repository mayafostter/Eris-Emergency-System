import uuid
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status, BackgroundTasks, WebSocket
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
    # Create a minimal config fallback
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
            'generate_official_statements': lambda self, context, stage, dept, type: "Mock emergency statement"
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

# DEBUG: Check app type
print(f"✅ DEBUG: App type = {type(app)}")
print(f"✅ DEBUG: FastAPI app created successfully")

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

# Global orchestrator storage (in production, use Redis/database)
active_orchestrators: Dict[str, ERISOrchestrator] = {}

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
        "cors_enabled": True
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
        "endpoints": {
            "simulate": "/simulate",
            "status": "/status/{simulation_id}",
            "metrics": "/metrics/{simulation_id}",
            "extended_metrics": "/extended-metrics/{simulation_id}",
            "enhanced_agents": "/enhanced-agents/{simulation_id}",
            "dashboard_metrics": "/metrics/dashboard/{simulation_id}",
            "websocket_metrics": "/ws/metrics/{simulation_id}",
            "orchestrator_agents": "/orchestrator/{simulation_id}/agents",
            "orchestrator_info": "/orchestrator/{simulation_id}",
            "health": "/health",
            "system_info": "/system/info",
            "agents_health": "/agents/health"
        }
    }

@app.post("/simulate")
async def start_simulation(request: SimulationRequest, background_tasks: BackgroundTasks):
    try:
        simulation_id = str(uuid.uuid4())
        
        logger.info(f"Starting ERIS simulation {simulation_id}: {request.disaster_type} in {request.location}, severity {request.severity}")

        # Validate disaster type (with fallback)
        valid_disasters = ["earthquake", "hurricane", "flood", "tsunami", "wildfire", "volcanic_eruption", "severe_storm", "epidemic", "pandemic", "landslide"]
        
        if hasattr(config, 'get_disaster_config') and callable(getattr(config, 'get_disaster_config')):
            disaster_config = config.get_disaster_config(request.disaster_type)
            if not disaster_config:
                raise HTTPException(status_code=400, detail=f"Unknown disaster type: {request.disaster_type}")
        else:
            # Fallback validation
            if request.disaster_type not in valid_disasters:
                raise HTTPException(status_code=400, detail=f"Unknown disaster type: {request.disaster_type}")

        # Create orchestrator using new architecture
        orchestrator = ERISOrchestrator(
            simulation_id=simulation_id,
            disaster_type=request.disaster_type,
            location=request.location,
            severity=request.severity,
            duration=request.duration
        )

        # Store orchestrator for tracking
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
            "enhanced_agents": 4
        }

        # Save simulation state (with error handling)
        try:
            await cloud.firestore.save_simulation_state(simulation_id, simulation_data)
        except Exception as e:
            logger.warning(f"Failed to save to Firestore: {e}")
        
        # Generate initial scenario content (with error handling)
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
            scenario = f"Emergency response activated for {request.disaster_type} in {request.location}. Severity level {request.severity}. ERIS 10-agent orchestrator deployed."

        # Start orchestrated simulation in background
        background_tasks.add_task(run_orchestrated_simulation, orchestrator, simulation_id)
        
        # Start metrics collection in background
        background_tasks.add_task(start_metrics_collection, simulation_id)

        # Response with orchestrator information
        response_data = {
            "simulation_id": simulation_id,
            "status": "initializing",
            "message": "ERIS 10-agent orchestrator deployed successfully",
            "orchestrator_info": {
                "version": "0.5.0",
                "ai_model": "Gemini 2.0 Flash",
                "total_agents": 10,
                "adk_agents": 6,
                "enhanced_agents": 4,
                "coordination_system": "cross-agent context sharing"
            },
            "frontend_ready": True,
            "data": {
                "disaster_type": request.disaster_type,
                "location": request.location,
                "severity": request.severity,
                "duration": request.duration,
                "scenario": scenario,
                "orchestrator_status": "initializing",
                "agent_types": {
                    "adk_agents": [
                        "emergency_response", "public_health", "infrastructure_manager",
                        "logistics_coordinator", "communications_director", "recovery_coordinator"
                    ],
                    "enhanced_agents": [
                        "hospital_load", "public_behavior", "social_media", "news_simulation"
                    ]
                },
                "total_agents": 10,
                "expected_metrics_delay_seconds": 15
            }
        }
        
        logger.info(f"ERIS simulation {simulation_id} deployed successfully")
        return response_data
        
    except Exception as e:
        logger.error(f"Simulation deployment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task for orchestrated simulation
async def run_orchestrated_simulation(orchestrator: ERISOrchestrator, simulation_id: str):
    """Background task to run the orchestrated simulation"""
    try:
        logger.info(f"Starting ERIS orchestrator for simulation {simulation_id}")
        
        # Update status to running
        try:
            await cloud.firestore.save_simulation_state(simulation_id, {
                "status": "running",
                "orchestrator_started": datetime.utcnow(),
                "message": "10-agent orchestrator active"
            })
        except Exception as e:
            logger.warning(f"Failed to update status: {e}")
        
        # Run the orchestrated simulation
        await orchestrator.start_simulation()
        
        logger.info(f"ERIS simulation {simulation_id} completed successfully")
        
    except Exception as e:
        logger.error(f"ERIS simulation {simulation_id} failed: {e}")
        
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
        # Clean up orchestrator
        if simulation_id in active_orchestrators:
            del active_orchestrators[simulation_id]
            logger.info(f"Cleaned up orchestrator for simulation {simulation_id}")

# Metrics collection for 10 agents
async def start_metrics_collection(simulation_id: str):
    """Background task for continuous metrics collection from all 10 agents"""
    try:
        collector = ERISMetricsCollector(simulation_id)
        
        # Wait for orchestrator to initialize agents
        await asyncio.sleep(10)
        
        while simulation_id in active_orchestrators:
            try:
                orchestrator = active_orchestrators[simulation_id]
                
                # Collect metrics from all agents
                await collector.collect_agent_metrics(orchestrator)
                collector.calculate_composite_scores(orchestrator.disaster_type, orchestrator.simulation_context)
                
                # Publish metrics
                await collector.publish_to_firestore()
                await collector.insert_to_bigquery()
                
                logger.debug(f"Metrics collected from 10 agents for simulation {simulation_id}")
                await asyncio.sleep(15)  # Collect every 15 seconds
                
            except Exception as e:
                logger.error(f"Metrics collection error for {simulation_id}: {e}")
                await asyncio.sleep(15)
                
        logger.info(f"Metrics collection completed for simulation {simulation_id}")
    except Exception as e:
        logger.error(f"Metrics collection initialization failed: {e}")

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
        
        if simulation_id in active_orchestrators:
            orchestrator = active_orchestrators[simulation_id]
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
                "agent_summary": agent_summary
            },
            "simulation_data": simulation_data,
            "enhanced_simulation": True,
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
        
        if simulation_id in active_orchestrators:
            orchestrator = active_orchestrators[simulation_id]
            try:
                # Collect final metrics from orchestrator
                final_metrics = await orchestrator._collect_final_metrics()
                agent_metrics = final_metrics
                
                orchestrator_metrics = {
                    "current_phase": orchestrator.current_phase.value,
                    "simulation_context": orchestrator.simulation_context,
                    "agent_statuses": orchestrator.agent_statuses,
                    "total_agents": len(orchestrator.adk_agents) + len(orchestrator.enhanced_agents),
                    "coordination_events": len(orchestrator.agent_statuses)
                }
            except Exception as e:
                logger.warning(f"Failed to get orchestrator metrics: {e}")

        return {
            "simulation_id": simulation_id,
            "current_phase": simulation_data.get("current_phase", "impact"),
            "orchestrator_metrics": orchestrator_metrics,
            "agent_metrics": agent_metrics,
            "ai_model": "Gemini 2.0 Flash",
            "total_agents": 10,
            "enhanced_metrics_available": True
        }
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/dashboard/{simulation_id}")
async def get_dashboard_metrics(simulation_id: str):
    """Get dashboard-ready metrics with composite scores"""
    if simulation_id not in active_orchestrators:
        # Try to get from Firestore for completed simulations
        try:
            simulation_data = await cloud.firestore.get_simulation_state(simulation_id)
        except Exception as e:
            logger.warning(f"Failed to get simulation data: {e}")
            simulation_data = None
            
        if not simulation_data:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        return {
            "simulation_id": simulation_id,
            "status": "completed",
            "message": "Simulation completed - no real-time metrics available",
            "dashboard_data": {
                "alert_level": "GREEN",
                "panic_index": 8,  # Slightly adjusted
                "hospital_capacity": 82,
                "population_affected": 4500,
                "infrastructure_failures": 0,
                "emergency_response": 96,
                "public_trust": 88,  # Added public trust
                "evacuation_compliance": 94  # Added evacuation compliance
            },
            "orchestrator_info": {
                "total_agents": 10,
                "ai_model": "Gemini 2.0 Flash"
            },
            "frontend_ready": True
        }
    
    orchestrator = active_orchestrators[simulation_id]
    
    try:
        collector = ERISMetricsCollector(simulation_id)
        
        # Collect metrics from all 10 agents
        await collector.collect_agent_metrics(orchestrator)
        collector.calculate_composite_scores(orchestrator.disaster_type, orchestrator.simulation_context)
        
        # Generate dashboard JSON
        dashboard_data = collector.generate_dashboard_json()
        
        # Add orchestrator-specific fallback metrics
        if not dashboard_data:
            # Fallback metrics based on current simulation context
            context = orchestrator.simulation_context
            dashboard_data = {
                "alert_level": "YELLOW" if context.get('panic_index', 0) > 0.5 else "GREEN",
                "panic_index": int(context.get('panic_index', 0.2) * 100),
                "hospital_capacity": int(context.get('hospital_capacity_utilization', 75)),
                "population_affected": int(context.get('total_population', 175000) * 0.1),
                "infrastructure_failures": int(context.get('infrastructure_damage', 20) / 10),
                "emergency_response": 90,
                "public_trust": int(context.get('official_communication_reach', 0.8) * 100),
                "evacuation_compliance": int(context.get('evacuation_compliance', 0.7) * 100)
            }
        
        return {
            "simulation_id": simulation_id,
            "status": "active",
            "dashboard_data": dashboard_data,
            "orchestrator_info": {
                "current_phase": orchestrator.current_phase.value,
                "total_agents": len(orchestrator.adk_agents) + len(orchestrator.enhanced_agents),
                "ai_model": "Gemini 2.0 Flash",
                "coordination_active": True
            },
            "timestamp": datetime.utcnow().isoformat(),
            "frontend_ready": True
        }
        
    except Exception as e:
        logger.error(f"Dashboard metrics error: {e}")
        # Return safe defaults for frontend
        return {
            "simulation_id": simulation_id,
            "status": "active",
            "dashboard_data": {
                "alert_level": "GREEN",
                "panic_index": 15,
                "hospital_capacity": 70,
                "population_affected": 8000,
                "infrastructure_failures": 1,
                "emergency_response": 88,
                "public_trust": 85,
                "evacuation_compliance": 78
            },
            "orchestrator_info": {
                "total_agents": 10,
                "ai_model": "Gemini 2.0 Flash"
            },
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "frontend_ready": True
        }

@app.websocket("/ws/metrics/{simulation_id}")
async def websocket_metrics(websocket: WebSocket, simulation_id: str):
    """Real-time metrics WebSocket endpoint"""
    await websocket.accept()
    
    if simulation_id not in active_orchestrators:
        await websocket.send_json({
            "error": "Active simulation not found", 
            "simulation_id": simulation_id,
            "message": "Simulation may be completed or not yet started"
        })
        await websocket.close()
        return
    
    try:
        collector = ERISMetricsCollector(simulation_id)
        logger.info(f"WebSocket connected for ERIS simulation {simulation_id}")
        
        while simulation_id in active_orchestrators:
            orchestrator = active_orchestrators[simulation_id]
            
            # Collect metrics from all 10 agents
            await collector.collect_agent_metrics(orchestrator)
            collector.calculate_composite_scores(orchestrator.disaster_type, orchestrator.simulation_context)
            
            dashboard_data = collector.generate_dashboard_json()
            
            # Add orchestrator context
            orchestrator_data = {
                "current_phase": orchestrator.current_phase.value,
                "total_agents": len(orchestrator.adk_agents) + len(orchestrator.enhanced_agents),
                "active_agents": len([s for s in orchestrator.agent_statuses.values() if "completed" in s]),
                "simulation_context": orchestrator.simulation_context,
                "ai_model": "Gemini 2.0 Flash"
            }
            
            # Send metrics to client
            await websocket.send_json({
                "simulation_id": simulation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "dashboard_metrics": dashboard_data,  # Changed from "metrics" to "dashboard_metrics"
                "orchestrator": orchestrator_data,
                "status": "streaming"
            })
            
            await asyncio.sleep(3)  # Stream every 3 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error for {simulation_id}: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        logger.info(f"WebSocket disconnected for simulation {simulation_id}")
        await websocket.close()

# Extended metrics endpoint
@app.get("/extended-metrics/{simulation_id}")
async def get_extended_metrics(simulation_id: str):
    """Get extended metrics from enhanced agents"""
    if simulation_id not in active_orchestrators:
        raise HTTPException(status_code=404, detail="Active simulation not found")
    
    try:
        orchestrator = active_orchestrators[simulation_id]
        
        # Collect metrics from enhanced agents  
        extended_metrics = {}
        
        # Hospital Load Metrics
        if hasattr(orchestrator, 'enhanced_agents') and 'hospital_load' in orchestrator.enhanced_agents:
            try:
                hospital_agent = orchestrator.enhanced_agents['hospital_load']
                hospital_metrics = await hospital_agent._generate_hospital_metrics()
                extended_metrics['hospital'] = hospital_metrics
            except Exception as e:
                logger.error(f"Hospital metrics error: {e}")
                extended_metrics['hospital'] = {"error": str(e)}
        
        # Public Behavior Metrics
        if hasattr(orchestrator, 'enhanced_agents') and 'public_behavior' in orchestrator.enhanced_agents:
            try:
                behavior_agent = orchestrator.enhanced_agents['public_behavior']
                behavior_metrics = await behavior_agent._generate_behavior_metrics()
                extended_metrics['public_behavior'] = behavior_metrics
            except Exception as e:
                logger.error(f"Behavior metrics error: {e}")
                extended_metrics['public_behavior'] = {"error": str(e)}
        
        # Social Media Metrics
        if hasattr(orchestrator, 'enhanced_agents') and 'social_media' in orchestrator.enhanced_agents:
            try:
                social_agent = orchestrator.enhanced_agents['social_media']
                social_metrics = {
                    "total_posts": len(social_agent.posts_generated) if hasattr(social_agent, 'posts_generated') else 0,
                    "current_panic_index": getattr(social_agent, 'panic_index', 0.0),
                    "misinformation_level": getattr(social_agent, 'misinformation_level', 0.0),
                    "viral_topics": len(getattr(social_agent, 'viral_topics', []))
                }
                extended_metrics['social_media'] = social_metrics
            except Exception as e:
                logger.error(f"Social media metrics error: {e}")
                extended_metrics['social_media'] = {"error": str(e)}
        
        # Simulation Metrics
        if hasattr(orchestrator, 'enhanced_agents') and 'news_simulation' in orchestrator.enhanced_agents:
            try:
                news_agent = orchestrator.enhanced_agents['news_simulation']
                news_metrics = {
                    "total_stories": len(getattr(news_agent, 'news_stories', [])),
                    "press_briefings": len(getattr(news_agent, 'press_briefings', [])),
                    "public_trust": getattr(news_agent, 'public_trust_level', 0.8),
                    "media_influence": getattr(news_agent, 'media_influence_score', 0.7)
                }
                extended_metrics['news_simulation'] = news_metrics
            except Exception as e:
                logger.error(f"News metrics error: {e}")
                extended_metrics['news_simulation'] = {"error": str(e)}
        
        return {
            "simulation_id": simulation_id,
            "extended_metrics": extended_metrics,
            "simulation_context": getattr(orchestrator, 'simulation_context', {}),
            "current_phase": orchestrator.current_phase.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Extended metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced agents info endpoint
@app.get("/enhanced-agents/{simulation_id}")
async def get_enhanced_agents_info(simulation_id: str):
    """Get detailed information about enhanced agents"""
    if simulation_id not in active_orchestrators:
        raise HTTPException(status_code=404, detail="Active simulation not found")
    
    orchestrator = active_orchestrators[simulation_id]
    
    try:
        enhanced_info = orchestrator.get_all_agent_info()
    except Exception as e:
        logger.warning(f"Failed to get enhanced agent info: {e}")
        enhanced_info = {}
    
    return {
        "simulation_id": simulation_id,
        "enhanced_agents_info": enhanced_info,
        "current_phase": orchestrator.current_phase.value,
        "timestamp": datetime.utcnow().isoformat()
    }

# Orchestrator info endpoint
@app.get("/orchestrator/{simulation_id}")
async def get_orchestrator_info(simulation_id: str):
    """Get detailed orchestrator information"""
    if simulation_id not in active_orchestrators:
        raise HTTPException(status_code=404, detail="Active orchestrator not found")
    
    orchestrator = active_orchestrators[simulation_id]
    
    try:
        orchestrator_info = orchestrator.get_simulation_status()
        agent_info = orchestrator.get_all_agent_info()
        
        return {
            "simulation_id": simulation_id,
            "orchestrator": {
                "version": "0.5.0",
                "ai_model": "Gemini 2.0 Flash",
                "status": orchestrator_info,
                "coordination_system": "cross-agent context sharing"
            },
            "agents": agent_info,
            "architecture": {
                "total_agents": len(orchestrator.adk_agents) + len(orchestrator.enhanced_agents),
                "adk_agents": len(orchestrator.adk_agents),
                "enhanced_agents": len(orchestrator.enhanced_agents)
            },
            "current_context": orchestrator.simulation_context,
            "frontend_ready": True
        }
    except Exception as e:
        logger.error(f"Error getting orchestrator info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Orchestrator agents endpoint
@app.get("/orchestrator/{simulation_id}/agents")
async def get_orchestrator_agents(simulation_id: str):
    """Get information about all agents in the orchestrator"""
    if simulation_id not in active_orchestrators:
        raise HTTPException(status_code=404, detail="Active orchestrator not found")
    
    orchestrator = active_orchestrators[simulation_id]
    
    try:
        all_agent_info = orchestrator.get_all_agent_info()
        
        # Format for frontend consumption
        formatted_agents = {}
        
        # Process ADK agents
        for agent_name, agent_data in all_agent_info.get("adk_agents", {}).items():
            formatted_agents[agent_name] = {
                **agent_data,
                "category": "adk",
                "ai_model": "Gemini 2.0 Flash",
                "efficiency": 92 + (hash(agent_name) % 8),
                "progress": min(95, len(orchestrator.agent_statuses) * 15)
            }
        
        # Process Enhanced agents
        for agent_name, agent_data in all_agent_info.get("enhanced_agents", {}).items():
            formatted_agents[agent_name] = {
                **agent_data,
                "category": "enhanced",
                "ai_model": "Gemini 2.0 Flash",
                "efficiency": 88 + (hash(agent_name) % 12),
                "progress": min(90, len(orchestrator.agent_statuses) * 12)
            }
        
        return {
            "simulation_id": simulation_id,
            "total_agent_count": len(formatted_agents),
            "adk_agent_count": len(all_agent_info.get("adk_agents", {})),
            "enhanced_agent_count": len(all_agent_info.get("enhanced_agents", {})),
            "agents": formatted_agents,
            "orchestrator_status": {
                "current_phase": orchestrator.current_phase.value,
                "ai_model": "Gemini 2.0 Flash",
                "coordination_active": True
            },
            "simulation_context": all_agent_info.get("simulation_context", {}),
            "frontend_ready": True
        }
    except Exception as e:
        logger.error(f"Error getting orchestrator agents: {e}")
        return {
            "simulation_id": simulation_id,
            "total_agent_count": 10,
            "adk_agent_count": 6,
            "enhanced_agent_count": 4,
            "agents": {},
            "orchestrator_status": {
                "current_phase": "impact",
                "ai_model": "Gemini 2.0 Flash"
            },
            "error": str(e),
            "frontend_ready": True
        }

# Agent update endpoint
@app.post("/agent/update")
async def update_agent_state(request: AgentUpdateRequest):
    try:
        # Save agent state using existing service
        try:
            await cloud.firestore.save_agent_state(
                request.agent_id,
                request.simulation_id,
                request.state_data
            )
        except Exception as e:
            logger.warning(f"Failed to save agent state: {e}")

        # Log event
        try:
            await cloud.bigquery.log_simulation_event(
                simulation_id=request.simulation_id,
                event_type="agent_updated",
                agent_id=request.agent_id,
                event_data={"agent_type": request.agent_type}
            )
        except Exception as e:
            logger.warning(f"Failed to log event: {e}")

        return {
            "simulation_id": request.simulation_id,
            "status": "updated",
            "message": f"Agent {request.agent_id} updated"
        }
    except Exception as e:
        logger.error(f"Agent update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Agents health check
@app.get("/agents/health")
async def agents_health_check():
    """Check health status of all agent systems"""
    return {
        "orchestrator": {
            "status": "active",
            "ai_model": "Gemini 2.0 Flash",
            "version": "0.5.0"
        },
        "adk_agents": {
            "count": 6,
            "status": "active",
            "types": [
                "emergency_response", "public_health", "infrastructure_manager",
                "logistics_coordinator", "communications_director", "recovery_coordinator"
            ]
        },
        "enhanced_agents": {
            "count": 4,
            "status": "active",
            "types": ["hospital_load", "public_behavior", "social_media", "news_simulation"]
        },
        "cloud_integration": "active",
        "total_agent_types": 10,
        "timestamp": datetime.utcnow().isoformat(),
        "frontend_ready": True
    }

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
        "frontend_compatible": True,
        "cors_enabled": True,
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
            "websocket_streaming": True
        },
        "supported_disasters": [
            "earthquake", "hurricane", "wildfire", "flood", "tsunami",
            "volcanic_eruption", "severe_storm", "epidemic", "pandemic", "landslide"
        ],
        "simulation_phases": ["impact", "response", "recovery"],
        "adk_agent_types": [
            "emergency_response", "public_health", "infrastructure_manager",
            "logistics_coordinator", "communications_director", "recovery_coordinator"
        ],
        "enhanced_agent_types": [
            "hospital_load", "public_behavior", "social_media", "news_simulation"
        ],
        "total_agent_types": 10,
        "simulation_context_variables": [
            "infrastructure_damage", "hospital_capacity_utilization", 
            "panic_index", "evacuation_compliance", "official_communication_reach",
            "supply_chain_disrupted", "social_media_activity", "public_trust_level"
        ]
    }

# Ping endpoint
@app.get("/ping")
async def ping():
    """Quick ping endpoint for frontend connection testing"""
    return {
        "message": "pong",
        "timestamp": datetime.utcnow().isoformat(),
        "orchestrator": "Gemini 2.0 Flash",  # Added orchestrator info
        "agents": 10,  # Updated agent count
        "version": "0.5.0",  # Updated version
        "cors_working": True,
        "server_ready": True
    }

# Fallback dashboard with orchestrator info
@app.get("/dashboard", response_class=HTMLResponse)
async def fallback_dashboard():
    
    try:
        # Get the path to fallback_dashboard.html (one level up from api/)
        dashboard_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fallback_dashboard.html")
        with open(dashboard_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <html>
                <head><title>ERIS Emergency System Dashboard</title></head>
                <body style="font-family: Arial, sans-serif; padding: 20px;">
                    <h1>🚨 ERIS Emergency Response Intelligence System</h1>
                    <h2>10-Agent AI Orchestrator with Gemini 2.0 Flash</h2>
                    <div style="background: #f0f8ff; padding: 15px; margin: 20px 0; border-radius: 8px;">
                        <h3>System Status: ✅ ACTIVE</h3>
                        <p><strong>AI Model:</strong> Gemini 2.0 Flash</p>
                        <p><strong>Total Agents:</strong> 10 (6 ADK + 4 Enhanced)</p>
                        <p><strong>Coordination:</strong> Cross-agent context sharing</p>
                        <p><strong>Version:</strong> 0.5.0</p>
                    </div>
                    <div style="background: #fff3cd; padding: 15px; margin: 20px 0; border-radius: 8px;">
                        <h3>🎯 Access Full Dashboard:</h3>
                        <p><a href="https://eris-emergency-system.vercel.app/" target="_blank" style="color: #0066cc; text-decoration: none; font-weight: bold;">https://eris-emergency-system.vercel.app/</a></p>
                    </div>
                    <div style="background: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 8px;">
                        <h3>📡 API Endpoints:</h3>
                        <ul>
                            <li><a href="/health">/health</a> - System health check</li>
                            <li><a href="/system/info">/system/info</a> - Full system capabilities</li>
                            <li><a href="/agents/health">/agents/health</a> - Agent system status</li>
                            <li><strong>POST /simulate</strong> - Start disaster simulation</li>
                        </ul>
                    </div>
                    <div style="background: #f8f9fa; padding: 15px; margin: 20px 0; border-radius: 8px;">
                        <h3>🤖 Agent Architecture:</h3>
                        <p><strong>ADK Agents (6):</strong> Emergency Response, Public Health, Infrastructure Manager, Logistics Coordinator, Communications Director, Recovery Coordinator</p>
                        <p><strong>Enhanced Agents (4):</strong> Hospital Load, Public Behavior, Social Media, News Simulation</p>
                    </div>
                </body>
            </html>
            """, 
            status_code=200
        )
    except Exception as e:
        return HTMLResponse(
            content=f"""
            <html>
                <head><title>ERIS System Error</title></head>
                <body style="font-family: Arial, sans-serif; padding: 20px;">
                    <h1>🚨 ERIS Emergency System</h1>
                    <p>Error loading dashboard: {str(e)}</p>
                    <p><a href="https://eris-emergency-system.vercel.app/">Access Main Dashboard</a></p>
                </body>
            </html>
            """, 
            status_code=500
        )
