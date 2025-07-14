"""
ERIS FastAPI Server
Emergency Response Intelligence System
"""

import uuid
import logging
import asyncio
import json
import time
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, status, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

# Security and middleware
from starlette.middleware.base import BaseHTTPMiddleware

# ===== OPTIONAL IMPORTS WITH FALLBACKS =====
SLOWAPI_AVAILABLE = False
PSUTIL_AVAILABLE = False
TRUSTEDHOST_AVAILABLE = False

# Rate limiting (optional)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SLOWAPI_AVAILABLE = True
except ImportError:
    class MockLimiter:
        def limit(self, rate_limit):
            def decorator(func):
                return func
            return decorator
    
    class RateLimitExceeded(Exception):
        pass
    
    def _rate_limit_exceeded_handler(request, exc):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "detail": "Too many requests"}
        )

# Memory monitoring (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    pass

# Trusted host middleware (optional)
try:
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    TRUSTEDHOST_AVAILABLE = True
except ImportError:
    pass

# ===== ERIS CORE IMPORTS =====
ERIS_IMPORTS_AVAILABLE = False
try:
    from services import get_cloud_services
    from utils.time_utils import SimulationTimeManager, SimulationPhase
    from config import ERISConfig, get_disaster_config
    from orchestrator.orchestrator import ERISOrchestrator
    from services.metrics_collector import ERISMetricsCollector
    ERIS_IMPORTS_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ All ERIS imports successful")
except ImportError as e:
    logging.warning(f"ERIS imports failed: {e}. Using fallback mode.")
    
    class SimulationPhase:
        IMPACT = "impact"
        RESPONSE = "response"
        RECOVERY = "recovery"
    
    def get_disaster_config(disaster_type: str) -> Dict[str, Any]:
        return {"severity_multiplier": 1.0, "duration": 24}

# ===== PRODUCTION CONFIGURATION IMPORTS =====
PRODUCTION_CONFIG_AVAILABLE = False
try:
    from config.production_settings import (
        apply_production_settings, 
        get_production_config,
        validate_production_environment
    )
    from services.government_data_service import GovernmentDataService
    PRODUCTION_CONFIG_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Production configuration loaded")
except ImportError as e:
    PRODUCTION_CONFIG_AVAILABLE = False
    logging.warning(f"⚠️ Production configuration not available: {e}")

# ===== LOGGING CONFIGURATION =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== SETTINGS CLASS =====
class Settings:
    def __init__(self):
        # Load production config if available
        self.production_config = None
        if PRODUCTION_CONFIG_AVAILABLE and os.getenv("ENVIRONMENT") == "production":
            try:
                self.production_config = get_production_config()
                validation = validate_production_environment()
                if not validation["valid"]:
                    logger.error(f"Production validation failed: {validation['errors']}")
                    if validation["warnings"]:
                        logger.warning(f"Production warnings: {validation['warnings']}")
                else:
                    logger.info("✅ Production environment validated")
            except Exception as e:
                logger.error(f"Failed to load production config: {e}")
        
        # Core settings
        self.API_KEYS = os.getenv("ERIS_API_KEYS", "dev-key-12345,prod-key-67890").split(",")
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
        self.DEBUG = os.getenv("DEBUG", "true").lower() == "true"
        
        # Production overrides
        if self.production_config:
            self.MAX_CONCURRENT_SIMULATIONS = self.production_config.production_overrides.get(
                "max_concurrent_simulations", int(os.getenv("MAX_CONCURRENT_SIMULATIONS", "10"))
            )
            self.RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_PER_MINUTE", "1000"))
        else:
            self.MAX_CONCURRENT_SIMULATIONS = int(os.getenv("MAX_CONCURRENT_SIMULATIONS", "10"))
            self.RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        
        self.WEBSOCKET_HEARTBEAT_INTERVAL = int(os.getenv("WEBSOCKET_HEARTBEAT_INTERVAL", "30"))
        self.RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
        self.ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")
        self.CORS_ORIGINS = self._get_cors_origins()
        
    def _get_cors_origins(self):
        """Get CORS origins based on environment"""
        if self.ENVIRONMENT == "production":
            cors_env = os.getenv("CORS_ALLOWED_ORIGINS", "")
            if cors_env:
                return cors_env.split(",")
            return [
                "https://eris-emergency-system.vercel.app",
                "https://eris-emergency-system-*.vercel.app",
            ]
        else:
            return [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:4173",
                "http://127.0.0.1:4173",
                "https://eris-emergency-system.vercel.app",
                "https://eris-emergency-system-*.vercel.app",
            ]

settings = Settings()

# ===== SECURITY MIDDLEWARE =====
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # OWASP Security Headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Custom ERIS headers
        response.headers["X-ERIS-Version"] = "1.0.0"
        response.headers["X-ERIS-Environment"] = settings.ENVIRONMENT
        response.headers["X-ERIS-Features"] = "production,government-data,real-time"
        
        return response

# ===== RATE LIMITING =====
if SLOWAPI_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = MockLimiter()
    logging.warning("Rate limiting disabled - slowapi not available")

# ===== AUTHENTICATION =====
security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Verify API key for protected endpoints"""
    if settings.ENVIRONMENT == "development":
        return True  # Skip auth in development
        
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.credentials not in settings.API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return True

# ===== HEALTH MONITORING =====
class HealthStatus:
    def __init__(self):
        self.startup_time = datetime.utcnow()
        self.last_health_check = datetime.utcnow()
        self.total_requests = 0
        self.failed_requests = 0
        self.active_simulations = 0
        self.websocket_connections = 0
        
    def record_request(self, success: bool = True):
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
        self.last_health_check = datetime.utcnow()
    
    def get_health_metrics(self):
        uptime = datetime.utcnow() - self.startup_time
        error_rate = (self.failed_requests / max(self.total_requests, 1)) * 100
        
        return {
            "status": "healthy" if error_rate < 10 else "degraded",
            "uptime_seconds": int(uptime.total_seconds()),
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "error_rate_percent": round(error_rate, 2),
            "active_simulations": self.active_simulations,
            "websocket_connections": self.websocket_connections,
            "memory_usage_mb": self._get_memory_usage(),
            "last_check": self.last_health_check.isoformat()
        }
    
    def _get_memory_usage(self):
        if PSUTIL_AVAILABLE:
            try:
                import psutil
                process = psutil.Process()
                return round(process.memory_info().rss / 1024 / 1024, 2)
            except Exception:
                return 0
        return 0

health_status = HealthStatus()

# ===== SERVICE INITIALIZATION =====
async def initialize_services():
    """Initialize ERIS services with production configuration and government data"""
    global config, cloud, government_data_service
    
    try:
        if ERIS_IMPORTS_AVAILABLE:
            config = ERISConfig()
            
            # Apply production overrides if in production
            if PRODUCTION_CONFIG_AVAILABLE and settings.ENVIRONMENT == "production":
                prod_config = apply_production_settings(config)
                logger.info("✅ Production configuration applied to ERIS")
            
            cloud = get_cloud_services()
            
            # Initialize government data service
            if PRODUCTION_CONFIG_AVAILABLE:
                government_data_service = GovernmentDataService()
                logger.info("✅ Government data service initialized")
            else:
                government_data_service = None
                logger.warning("⚠️ Government data service not available")
            
            logger.info("✅ ERIS services initialized successfully")
        else:
            # Fallback configuration
            config = type('FallbackConfig', (), {
                'get_disaster_config': lambda self, disaster_type: {"severity_multiplier": 1.0, "duration": 24}
            })()
            
            # Fallback cloud services
            cloud = type('FallbackCloud', (), {
                'firestore': type('MockFirestore', (), {
                    'save_simulation_state': lambda self, sim_id, data, merge=True: asyncio.sleep(0),
                    'get_simulation_state': lambda self, sim_id: {"status": "active"},
                    'log_event': lambda self, sim_id, event_type, data, agent_id=None: "mock-event-id",
                    'get_active_simulations': lambda self: []
                })(),
                'vertex_ai': type('MockVertexAI', (), {
                    'generate_official_statements': lambda self, context, stage, dept, type: {
                        "title": "Mock Emergency Statement",
                        "content": f"Emergency response activated for {context.get('type', 'disaster')}."
                    },
                    'generate_social_media_posts': lambda self, context, phase, num_posts=5, platform="twitter": [
                        {"content": "Mock social media post", "author_type": "resident", "timestamp": "2 min ago", "engagement_level": "medium"}
                    ]
                })()
            })()
            government_data_service = None
            logger.warning("⚠️ Using fallback services - limited functionality available")
            
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        raise

async def cleanup_resources():
    """Clean up active resources"""
    try:
        # Close active orchestrators
        for sim_id in list(active_orchestrators.keys()):
            try:
                await stop_simulation(sim_id)
            except Exception as e:
                logger.error(f"Error stopping simulation {sim_id}: {e}")
        
        # Close websocket connections
        for sim_id, connections in websocket_manager.connections.items():
            for ws in connections:
                try:
                    await ws.close(code=1001, reason="Server shutdown")
                except:
                    pass
        
        # Close government data service
        if government_data_service:
            try:
                if hasattr(government_data_service, 'session') and government_data_service.session:
                    await government_data_service.session.close()
            except Exception as e:
                logger.error(f"Error closing government data service: {e}")
        
        logger.info("🧹 Resource cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting ERIS FastAPI server...")
    await initialize_services()
    logger.info("✅ ERIS server started successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down ERIS server...")
    await cleanup_resources()
    logger.info("✅ ERIS server shutdown complete")

# ===== FASTAPI APP CREATION =====
app = FastAPI(
    title="ERIS Emergency Response Intelligence System",
    version="1.0.0",
    description="Production-ready disaster simulation platform with 10 AI agents and real government data integration",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# ===== MIDDLEWARE SETUP =====
# Security headers
app.add_middleware(SecurityHeadersMiddleware)

# Trusted hosts (optional)
if TRUSTEDHOST_AVAILABLE and settings.ALLOWED_HOSTS != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-ERIS-Version", "X-ERIS-Environment", "X-ERIS-Features"]
)

# Rate limiting
if SLOWAPI_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ===== REQUEST MIDDLEWARE =====
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Request processing middleware"""
    start_time = time.time()
    
    # Record request
    health_status.record_request()
    
    try:
        response = await call_next(request)
        
        # Add processing time header
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        health_status.record_request(success=False)
        logger.error(f"Request failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "request_id": str(uuid.uuid4())}
        )

# ===== GLOBAL STORAGE =====
active_orchestrators: Dict[str, Any] = {}
simulation_cache: Dict[str, Dict] = {}

# ===== PYDANTIC MODELS =====
class SimulationRequest(BaseModel):
    disaster_type: str
    location: str
    severity: int = Field(..., ge=1, le=10)
    duration: int = Field(72, ge=1, le=168)
    use_government_data: bool = Field(True, description="Use real government data APIs")
    
    @validator('disaster_type')
    def validate_disaster_type(cls, v):
        valid_disasters = [
            "earthquake", "hurricane", "flood", "tsunami", "wildfire", 
            "volcanic_eruption", "severe_storm", "epidemic", "pandemic", "landslide", "tornado", "blizzard"
        ]
        if v not in valid_disasters:
            raise ValueError(f"Invalid disaster type. Must be one of: {valid_disasters}")
        return v
    
    @validator('location')
    def validate_location(cls, v):
        if len(v.strip()) < 2:
            raise ValueError("Location must be at least 2 characters")
        return v.strip()

class SimulationResponse(BaseModel):
    simulation_id: str
    status: str
    message: str
    orchestrator_info: Dict[str, Any]
    data: Dict[str, Any]
    government_data_enabled: bool = False

# ===== WEBSOCKET MANAGER =====
class WebSocketManager:
    def __init__(self):
        self.connections: Dict[str, List[WebSocket]] = {}
        self.connection_metadata: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, simulation_id: str):
        """Connect WebSocket with enhanced tracking"""
        await websocket.accept()
        
        if simulation_id not in self.connections:
            self.connections[simulation_id] = []
        
        self.connections[simulation_id].append(websocket)
        
        # Track connection metadata
        connection_id = id(websocket)
        self.connection_metadata[connection_id] = {
            "simulation_id": simulation_id,
            "connected_at": datetime.utcnow(),
            "last_heartbeat": datetime.utcnow()
        }
        
        health_status.websocket_connections = sum(len(conns) for conns in self.connections.values())
        logger.info(f"WebSocket connected for simulation {simulation_id} (total: {health_status.websocket_connections})")
    
    def disconnect(self, websocket: WebSocket, simulation_id: str):
        """Disconnect WebSocket with cleanup"""
        if simulation_id in self.connections:
            if websocket in self.connections[simulation_id]:
                self.connections[simulation_id].remove(websocket)
            
            if not self.connections[simulation_id]:
                del self.connections[simulation_id]
        
        # Clean up metadata
        connection_id = id(websocket)
        if connection_id in self.connection_metadata:
            del self.connection_metadata[connection_id]
        
        health_status.websocket_connections = sum(len(conns) for conns in self.connections.values())
        logger.info(f"WebSocket disconnected for simulation {simulation_id} (total: {health_status.websocket_connections})")
    
    async def broadcast_to_simulation(self, simulation_id: str, data: dict):
        """Broadcast with connection health checking"""
        if simulation_id not in self.connections:
            return
        
        disconnected = []
        successful_sends = 0
        
        for websocket in self.connections[simulation_id]:
            try:
                await websocket.send_json(data)
                successful_sends += 1
                
                # Update heartbeat
                connection_id = id(websocket)
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]["last_heartbeat"] = datetime.utcnow()
                    
            except Exception as e:
                logger.warning(f"WebSocket send failed: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected websockets
        for ws in disconnected:
            self.disconnect(ws, simulation_id)
        
        if successful_sends > 0:
            logger.debug(f"Broadcast to {successful_sends} WebSocket connections for simulation {simulation_id}")
    
    async def cleanup_stale_connections(self):
        """Clean up stale WebSocket connections"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=5)
        
        stale_connections = [
            conn_id for conn_id, metadata in self.connection_metadata.items()
            if metadata["last_heartbeat"] < cutoff_time
        ]
        
        for conn_id in stale_connections:
            metadata = self.connection_metadata[conn_id]
            simulation_id = metadata["simulation_id"]
            
            # Find and remove the stale connection
            if simulation_id in self.connections:
                stale_ws = [ws for ws in self.connections[simulation_id] if id(ws) == conn_id]
                for ws in stale_ws:
                    self.disconnect(ws, simulation_id)

websocket_manager = WebSocketManager()

# ===== METRICS CALCULATOR =====
class DynamicMetricsCalculator:
    def __init__(self, simulation_id: str, orchestrator: Any = None):
        self.simulation_id = simulation_id
        self.orchestrator = orchestrator
        self.last_update = datetime.utcnow()
        
    def calculate_real_time_metrics(self) -> Dict[str, Any]:
        """Calculate real-time metrics with enhanced error handling"""
        try:
            if self.orchestrator and hasattr(self.orchestrator, 'simulation_context'):
                context = self.orchestrator.simulation_context
            else:
                context = simulation_cache.get(self.simulation_id, {})
            
            current_time = datetime.utcnow()
            time_elapsed = (current_time - self.last_update).total_seconds() / 60
            
            # Calculate base metrics
            metrics = self._calculate_base_metrics(context, time_elapsed)
            
            # Add metadata
            metrics.update({
                "simulation_id": self.simulation_id,
                "timestamp": current_time.isoformat(),
                "calculation_duration_ms": int((datetime.utcnow() - current_time).total_seconds() * 1000),
                "data_source": "orchestrator" if self.orchestrator else "cache"
            })
            
            self.last_update = current_time
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {self.simulation_id}: {e}")
            return self._get_fallback_metrics()
    
    def _calculate_base_metrics(self, context: Dict, time_elapsed: float) -> Dict[str, Any]:
        """Calculate base metrics from context"""
        import random
        
        # Base values with realistic evolution
        base_panic = context.get('panic_index', random.uniform(0.2, 0.4))
        base_hospital = context.get('hospital_capacity_utilization', random.uniform(65, 80))
        base_population = context.get('total_population', 175000)
        
        # Evolve metrics over time
        panic_index = min(0.9, base_panic + (time_elapsed * random.uniform(0.01, 0.05)))
        hospital_capacity = min(95, base_hospital + (panic_index * 15) + random.uniform(-5, 5))
        
        population_affected = int(base_population * (0.1 + panic_index * 0.3))
        infrastructure_failures = max(0, int(time_elapsed / 10) + random.randint(0, 3))
        
        # Calculate derived metrics
        alert_level = self._determine_alert_level(panic_index, hospital_capacity, infrastructure_failures)
        emergency_response = max(70, min(98, 90 - (infrastructure_failures * 2) + random.randint(-5, 5)))
        public_trust = max(30, min(90, 75 - (panic_index * 20) + random.randint(-5, 5)))
        evacuation_compliance = max(20, min(95, 70 + (panic_index * 10) + random.randint(-8, 8)))
        
        return {
            "alert_level": alert_level,
            "panic_index": int(panic_index * 100),
            "hospital_capacity": int(hospital_capacity),
            "population_affected": population_affected,
            "infrastructure_failures": infrastructure_failures,
            "emergency_response": emergency_response,
            "public_trust": public_trust,
            "evacuation_compliance": evacuation_compliance
        }
    
    def _determine_alert_level(self, panic_index: float, hospital_capacity: float, infrastructure_failures: int) -> str:
        """Determine alert level based on metrics"""
        risk_score = (panic_index * 40) + max(0, (hospital_capacity - 80) * 2) + (infrastructure_failures * 3)
        
        if risk_score >= 60:
            return "RED"
        elif risk_score >= 35:
            return "ORANGE"
        elif risk_score >= 15:
            return "YELLOW"
        else:
            return "GREEN"
    
    def _get_fallback_metrics(self) -> Dict[str, Any]:
        """Return safe fallback metrics"""
        import random
        return {
            "alert_level": "YELLOW",
            "panic_index": random.randint(20, 40),
            "hospital_capacity": random.randint(70, 85),
            "population_affected": random.randint(8000, 15000),
            "infrastructure_failures": random.randint(1, 4),
            "emergency_response": random.randint(80, 95),
            "public_trust": random.randint(60, 80),
            "evacuation_compliance": random.randint(65, 85),
            "timestamp": datetime.utcnow().isoformat(),
            "simulation_id": self.simulation_id,
            "error": "Using fallback metrics"
        }

# ===== UTILITY FUNCTIONS =====
async def _check_firestore_health() -> bool:
    """Check Firestore service health"""
    try:
        if hasattr(cloud, 'firestore'):
            await cloud.firestore.get_active_simulations()
            return True
    except Exception as e:
        logger.warning(f"Firestore health check failed: {e}")
    return False

async def _check_vertex_ai_health() -> bool:
    """Check Vertex AI service health"""
    try:
        if hasattr(cloud, 'vertex_ai'):
            test_context = {"type": "test", "location": "test", "severity": 1}
            await cloud.vertex_ai.generate_official_statements(test_context, "test", "test", "test")
            return True
    except Exception as e:
        logger.warning(f"Vertex AI health check failed: {e}")
    return False

# ===== CORE ENDPOINTS =====
@app.get("/health")
async def health_check(request: Request):
    """Enhanced health check with detailed metrics"""
    health_metrics = health_status.get_health_metrics()
    
    # Add service-specific health checks
    service_health = {
        "firestore": await _check_firestore_health(),
        "vertex_ai": await _check_vertex_ai_health(),
        "orchestrator": len(active_orchestrators) < settings.MAX_CONCURRENT_SIMULATIONS,
        "government_apis": await _check_government_apis_health() if government_data_service else False
    }
    
    overall_status = "healthy"
    if health_metrics["error_rate_percent"] > 10:
        overall_status = "degraded"
    if not all(service_health.values()):
        overall_status = "unhealthy"
    
    return {
        "status": overall_status,
        "version": "0.5.0",
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat(),
        "health_metrics": health_metrics,
        "service_health": service_health,
        "orchestrator_info": {
            "ai_model": "Gemini 2.0 Flash",
            "total_agents": 10,
            "adk_agents": 6,
            "enhanced_agents": 4,
            "active_simulations": len(active_orchestrators),
            "max_concurrent": settings.MAX_CONCURRENT_SIMULATIONS
        }
    }

async def _check_firestore_health() -> bool:
    """Check Firestore service health"""
    try:
        if hasattr(cloud, 'firestore'):
            # Simple health check - get active simulations
            await cloud.firestore.get_active_simulations()
            return True
    except Exception as e:
        logger.warning(f"Firestore health check failed: {e}")
    return False

async def _check_vertex_ai_health() -> bool:
    """Check Vertex AI service health"""
    try:
        if hasattr(cloud, 'vertex_ai'):
            # Simple test generation
            test_context = {"type": "test", "location": "test", "severity": 1}
            await cloud.vertex_ai.generate_official_statements(test_context, "test", "test", "test")
            return True
    except Exception as e:
        logger.warning(f"Vertex AI health check failed: {e}")
    return False

async def _check_government_apis_health() -> bool:
    """Check government APIs health"""
    try:
        if government_data_service:
            # Quick health check of key government APIs
            health_results = await government_data_service.check_apis_health()
            return health_results.get("overall_healthy", False)
    except Exception as e:
        logger.warning(f"Government APIs health check failed: {e}")
    return False

@app.get("/system/info")
async def get_system_info(request: Request):
    """Enhanced system information"""
    return {
        "eris_version": "0.5.0",
        "environment": settings.ENVIRONMENT,
        "api_version": "v1",
        "orchestrator": {
            "ai_model": "Gemini 2.0 Flash",
            "architecture": "10-agent coordination system",
            "coordination_method": "cross-agent context sharing",
            "version": "0.5.0"
        },
        "agent_system": {
            "total_agents": 10,
            "adk_agents": 6,
            "enhanced_agents": 4,
            "ai_powered": True,
            "concurrent_execution": True
        },
        "capabilities": {
        "disaster_simulation": True,
        "multi_agent_coordination": True,
        "real_time_metrics": True,
        "websocket_streaming": True,
        "cloud_integration": True,
        "production_ready": True,
        "rate_limiting": True,
        "authentication": settings.ENVIRONMENT == "production",
        "monitoring": True,
        "error_recovery": True,
        "government_data_integration": PRODUCTION_CONFIG_AVAILABLE,
        "real_time_emergency_alerts": PRODUCTION_CONFIG_AVAILABLE,
        "earthquake_monitoring": PRODUCTION_CONFIG_AVAILABLE,
        "hospital_data_integration": PRODUCTION_CONFIG_AVAILABLE
        },
        "limits": {
            "max_concurrent_simulations": settings.MAX_CONCURRENT_SIMULATIONS,
            "rate_limit_requests_per_minute": settings.RATE_LIMIT_REQUESTS,
            "websocket_heartbeat_interval": settings.WEBSOCKET_HEARTBEAT_INTERVAL
        },
        "current_load": {
            "active_simulations": len(active_orchestrators),
            "websocket_connections": health_status.websocket_connections,
            "total_requests": health_status.total_requests
        }
    }

@app.post("/simulate", response_model=SimulationResponse)
async def start_simulation(
    request: Request,
    simulation_request: SimulationRequest,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_api_key)
):
    """Start new simulation with enhanced validation and monitoring"""
    
    # Check concurrent simulation limit
    if len(active_orchestrators) >= settings.MAX_CONCURRENT_SIMULATIONS:
        raise HTTPException(
            status_code=429,
            detail=f"Maximum concurrent simulations ({settings.MAX_CONCURRENT_SIMULATIONS}) reached"
        )
    
    simulation_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting simulation {simulation_id}: {simulation_request.disaster_type} in {simulation_request.location}")
        
        # Create simulation context
        simulation_data = {
            "simulation_id": simulation_id,
            "request_id": request_id,
            "disaster_type": simulation_request.disaster_type,
            "location": simulation_request.location,
            "severity": simulation_request.severity,
            "duration": simulation_request.duration,
            "status": "initializing",
            "created_at": datetime.utcnow().isoformat(),
            "orchestrator_version": "0.5.0",
            "ai_model": "Gemini 2.0 Flash"
        }
        
        # Store in cache for immediate access
        simulation_cache[simulation_id] = simulation_data
        
        # Save to persistent storage
        try:
            await cloud.firestore.save_simulation_state(simulation_id, simulation_data)
            await cloud.firestore.log_event(
                simulation_id,
                "simulation_started",
                {"request": simulation_request.dict(), "request_id": request_id}
            )
        except Exception as e:
            logger.warning(f"Failed to persist simulation data: {e}")
        
        # Create orchestrator if ERIS imports are available
        orchestrator_created = False
        if ERIS_IMPORTS_AVAILABLE:
            try:
                orchestrator = ERISOrchestrator(
                    simulation_id=simulation_id,
                    disaster_type=simulation_request.disaster_type,
                    location=simulation_request.location,
                    severity=simulation_request.severity,
                    duration=simulation_request.duration
                )
                
                active_orchestrators[simulation_id] = orchestrator
                orchestrator_created = True
                
                # Update simulation data
                simulation_data.update({
                    "status": "running",
                    "orchestrator_active": True,
                    "orchestrator_version": "0.5.0",
                    "total_agents": 10
                })
                
                # Start orchestration in background
                background_tasks.add_task(run_orchestrated_simulation, orchestrator, simulation_id)
                
                logger.info(f"✅ Orchestrator created and stored for simulation {simulation_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to create orchestrator: {e}")
                # Continue with basic simulation
                simulation_data.update({
                    "orchestrator_active": False,
                    "orchestrator_error": str(e),
                    "status": "running_fallback"
                })
        else:
            logger.info("⚠️ ERIS orchestrator not available - using basic simulation mode")
            simulation_data.update({
                "orchestrator_active": False,
                "status": "running_basic"
            })

        # Store in cache for immediate access
        simulation_cache[simulation_id] = simulation_data
        
        # Start supporting services
        background_tasks.add_task(start_metrics_streaming, simulation_id)
        background_tasks.add_task(start_government_data_monitoring, simulation_id)
        
        # Update health status
        health_status.active_simulations = len(active_orchestrators)
        
        # Generate response
        orchestrator_info = {
            "version": "0.5.0",
            "ai_model": "Gemini 2.0 Flash",
            "total_agents": 10,
            "status": "initializing",
            "real_time_features": True,
            "websocket_endpoint": f"/ws/metrics/{simulation_id}"
        }
        
        response = SimulationResponse(
            simulation_id=simulation_id,
            status="initializing",
            message="ERIS simulation started successfully",
            orchestrator_info=orchestrator_info,
            data=simulation_data
        )
        
        logger.info(f"Simulation {simulation_id} started successfully")
        return response
        
    except Exception as e:
        logger.error(f"Failed to start simulation: {e}")
        # Clean up on failure
        if simulation_id in active_orchestrators:
            del active_orchestrators[simulation_id]
        if simulation_id in simulation_cache:
            del simulation_cache[simulation_id]
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start simulation: {str(e)}"
        )

@app.get("/status/{simulation_id}")
async def get_simulation_status(request: Request, simulation_id: str):
    """Get detailed simulation status with comprehensive metrics"""
    try:
        # Check cache first
        simulation_data = simulation_cache.get(simulation_id)
        
        # Fallback to persistent storage
        if not simulation_data:
            try:
                simulation_data = await cloud.firestore.get_simulation_state(simulation_id)
            except Exception as e:
                logger.warning(f"Failed to retrieve from Firestore: {e}")
        
        if not simulation_data:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        # Get orchestrator status
        orchestrator_status = None
        agent_summary = None
        real_time_active = False
        
        if simulation_id in active_orchestrators:
            orchestrator = active_orchestrators[simulation_id]
            real_time_active = True
            
            try:
                if hasattr(orchestrator, 'get_simulation_status'):
                    orchestrator_status = orchestrator.get_simulation_status()
                
                if hasattr(orchestrator, 'adk_agents') and hasattr(orchestrator, 'enhanced_agents'):
                    agent_summary = {
                        "total_agents": len(orchestrator.adk_agents) + len(orchestrator.enhanced_agents),
                        "adk_agents": len(orchestrator.adk_agents),
                        "enhanced_agents": len(orchestrator.enhanced_agents),
                        "active_agents": len([s for s in getattr(orchestrator, 'agent_statuses', {}).values() 
                                            if "active" in str(s).lower() or "completed" in str(s).lower()]),
                        "current_phase": getattr(orchestrator, 'current_phase', 'unknown')
                    }
            except Exception as e:
                logger.warning(f"Failed to get orchestrator details: {e}")
        
        # Calculate metrics
        metrics_calculator = DynamicMetricsCalculator(simulation_id, active_orchestrators.get(simulation_id))
        current_metrics = metrics_calculator.calculate_real_time_metrics()
        
        return {
            "simulation_id": simulation_id,
            "status": simulation_data.get("status", "unknown"),
            "created_at": simulation_data.get("created_at"),
            "last_updated": datetime.utcnow().isoformat(),
            "orchestrator": {
                "version": "0.5.0",
                "ai_model": "Gemini 2.0 Flash",
                "status": orchestrator_status,
                "agent_summary": agent_summary,
                "real_time_active": real_time_active,
                "websocket_connections": len(websocket_manager.connections.get(simulation_id, []))
            },
            "current_metrics": current_metrics,
            "simulation_data": simulation_data,
            "request_id": simulation_data.get("request_id", "unknown")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check error for {simulation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/dashboard/{simulation_id}")
async def get_dashboard_metrics(request: Request, simulation_id: str):
    """Get real-time dashboard metrics with caching"""
    try:
        # Check if simulation exists
        if simulation_id not in simulation_cache and simulation_id not in active_orchestrators:
            try:
                simulation_data = await cloud.firestore.get_simulation_state(simulation_id)
                if not simulation_data:
                    raise HTTPException(status_code=404, detail="Simulation not found")
                simulation_cache[simulation_id] = simulation_data
            except Exception as e:
                logger.warning(f"Failed to retrieve simulation: {e}")
                raise HTTPException(status_code=404, detail="Simulation not found")
        
        # Calculate real-time metrics
        orchestrator = active_orchestrators.get(simulation_id)
        metrics_calculator = DynamicMetricsCalculator(simulation_id, orchestrator)
        dashboard_data = metrics_calculator.calculate_real_time_metrics()
        
        # Get orchestrator info
        orchestrator_info = {
            "ai_model": "Gemini 2.0 Flash",
            "version": "0.5.0",
            "real_time_enabled": simulation_id in active_orchestrators,
            "total_agents": 10
        }
        
        if orchestrator:
            try:
                orchestrator_info.update({
                    "current_phase": getattr(orchestrator, 'current_phase', 'unknown'),
                    "coordination_active": True,
                    "agent_count": len(getattr(orchestrator, 'adk_agents', {})) + len(getattr(orchestrator, 'enhanced_agents', {}))
                })
            except Exception as e:
                logger.warning(f"Failed to get orchestrator info: {e}")
        
        return {
            "simulation_id": simulation_id,
            "status": "active" if simulation_id in active_orchestrators else "completed",
            "dashboard_data": dashboard_data,
            "orchestrator_info": orchestrator_info,
            "timestamp": datetime.utcnow().isoformat(),
            "frontend_ready": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard metrics error for {simulation_id}: {e}")
        # Return safe fallback
        return {
            "simulation_id": simulation_id,
            "status": "error",
            "dashboard_data": DynamicMetricsCalculator(simulation_id).calculate_real_time_metrics(),
            "orchestrator_info": {"ai_model": "Gemini 2.0 Flash", "version": "0.5.0"},
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "frontend_ready": True
        }

@app.get("/orchestrator/{simulation_id}/agents")
async def get_agents_info(request: Request, simulation_id: str):
    """Get detailed agent information and status"""
    if simulation_id not in active_orchestrators and simulation_id not in simulation_cache:
      raise HTTPException(status_code=404, detail="Simulation not found")
    
    try:
        orchestrator = active_orchestrators[simulation_id]
        
        # Default agent configuration
        agent_configs = [
            {"id": "emergency_response_coordinator", "name": "Emergency Response Coordinator", "type": "adk", "status": "active"},
            {"id": "public_health_manager", "name": "Public Health Manager", "type": "adk", "status": "active"},
            {"id": "infrastructure_manager", "name": "Infrastructure Manager", "type": "adk", "status": "active"},
            {"id": "logistics_coordinator", "name": "Logistics Coordinator", "type": "adk", "status": "active"},
            {"id": "communications_director", "name": "Communications Director", "type": "adk", "status": "active"},
            {"id": "recovery_coordinator", "name": "Recovery Coordinator", "type": "adk", "status": "active"},
            {"id": "hospital_load_modeler", "name": "Hospital Load Modeler", "type": "enhanced", "status": "active"},
            {"id": "public_behavior_simulator", "name": "Public Behavior Simulator", "type": "enhanced", "status": "active"},
            {"id": "social_media_dynamics", "name": "Social Media Dynamics", "type": "enhanced", "status": "active"},
            {"id": "news_coverage_simulator", "name": "News Coverage Simulator", "type": "enhanced", "status": "active"}
        ]
        
        # Enhance with real orchestrator data if available
        if hasattr(orchestrator, 'agent_statuses'):
            for agent in agent_configs:
                agent_id = agent["id"]
                if agent_id in orchestrator.agent_statuses:
                    agent["detailed_status"] = orchestrator.agent_statuses[agent_id]
                    agent["efficiency"] = 95 + (hash(agent_id) % 10) - 5  # Simulated efficiency
                    agent["progress"] = min(100, 20 + (hash(agent_id) % 80))  # Simulated progress
        
        return {
            "simulation_id": simulation_id,
            "agents": agent_configs,
            "orchestrator_info": {
                "ai_model": "Gemini 2.0 Flash",
                "total_agents": len(agent_configs),
                "adk_agents": len([a for a in agent_configs if a["type"] == "adk"]),
                "enhanced_agents": len([a for a in agent_configs if a["type"] == "enhanced"]),
                "coordination_active": True
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting agent info for {simulation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/live-feed/{simulation_id}")
async def get_live_feed(request: Request, simulation_id: str, limit: int = 20):
    """Get live emergency feed with AI-generated content"""
    if simulation_id not in active_orchestrators and simulation_id not in simulation_cache:
        try:
            simulation_data = await cloud.firestore.get_simulation_state(simulation_id)
            if not simulation_data:
                raise HTTPException(status_code=404, detail="Simulation not found")
            simulation_cache[simulation_id] = simulation_data
        except Exception as e:
            raise HTTPException(status_code=404, detail="Simulation not found")
    
    try:
        feed_items = []
        
        # Generate emergency updates based on simulation context
        simulation_data = simulation_cache.get(simulation_id, {})
        disaster_type = simulation_data.get("disaster_type", "emergency")
        location = simulation_data.get("location", "affected area")
        
        # Generate contextual feed items
        feed_templates = [
            {
                "source": "@EmergencyAlert",
                "content": f"🚨 {disaster_type.replace('_', ' ').title()} emergency in {location}. Follow official evacuation orders.",
                "type": "official_alert",
                "priority": "high"
            },
            {
                "source": "@ERISSystem",
                "content": f"AI orchestrator managing {len(active_orchestrators.get(simulation_id, {}))} response agents. Real-time coordination active.",
                "type": "system_update",
                "priority": "medium"
            },
            {
                "source": "@LocalHospital",
                "content": "Emergency medical services at capacity. Non-urgent cases please seek alternative care.",
                "type": "medical_update",
                "priority": "high"
            },
            {
                "source": "@PublicSafety",
                "content": f"Infrastructure assessment ongoing in {location}. Avoid damaged areas.",
                "type": "safety_update",
                "priority": "medium"
            }
        ]
        
        # Generate feed items with timestamps
        for i, template in enumerate(feed_templates[:limit]):
            feed_items.append({
                "id": f"feed_{simulation_id}_{i}",
                "source": template["source"],
                "content": template["content"],
                "type": template["type"],
                "priority": template["priority"],
                "timestamp": (datetime.utcnow() - timedelta(minutes=i*3)).isoformat(),
                "engagement": {
                    "likes": hash(template["content"]) % 100,
                    "shares": hash(template["content"]) % 50,
                    "comments": hash(template["content"]) % 25
                }
            })
        
        # Add AI-generated social media content if Vertex AI is available
        try:
            if hasattr(cloud, 'vertex_ai'):
                disaster_context = {
                    "type": disaster_type,
                    "location": location,
                    "severity": simulation_data.get("severity", 5)
                }
                
                social_posts = await cloud.vertex_ai.generate_social_media_posts(
                    disaster_context, "response", num_posts=3, platform="twitter"
                )
                
                for i, post in enumerate(social_posts):
                    feed_items.append({
                        "id": f"social_{simulation_id}_{i}",
                        "source": f"@{post.get('author_type', 'User').title()}{hash(post['content']) % 1000}",
                        "content": post["content"],
                        "type": "social_media",
                        "priority": "low",
                        "timestamp": (datetime.utcnow() - timedelta(minutes=(i+5)*2)).isoformat(),
                        "engagement": {
                            "likes": hash(post["content"]) % 200,
                            "shares": hash(post["content"]) % 100,
                            "comments": hash(post["content"]) % 50
                        },
                        "ai_generated": True
                    })
        except Exception as e:
            logger.warning(f"Failed to generate AI content: {e}")
        
        # Sort by timestamp (newest first)
        feed_items.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "simulation_id": simulation_id,
            "feed_items": feed_items[:limit],
            "total_items": len(feed_items),
            "last_updated": datetime.utcnow().isoformat(),
            "ai_content_enabled": hasattr(cloud, 'vertex_ai')
        }
        
    except Exception as e:
        logger.error(f"Live feed error for {simulation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== GOVERNMENT DATA ENDPOINTS =====

@app.get("/government/health")
async def get_government_apis_health(request: Request):
    """Check health status of all government APIs"""
    if not government_data_service:
        return {
            "status": "unavailable",
            "message": "Government data service not initialized",
            "apis": {},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        health_results = await government_data_service.check_apis_health()
        return {
            "status": "healthy" if health_results.get("overall_healthy") else "degraded",
            "apis": health_results.get("api_status", {}),
            "response_times": health_results.get("response_times", {}),
            "last_check": datetime.utcnow().isoformat(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Government APIs health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/government/alerts")
async def get_emergency_alerts(request: Request, state: str = None, limit: int = 10):
    """Get current emergency alerts from FEMA and NWS"""
    if not government_data_service:
        raise HTTPException(status_code=503, detail="Government data service unavailable")
    
    try:
        alerts = await government_data_service.get_emergency_alerts(state=state, limit=limit)
        return {
            "alerts": alerts,
            "total_count": len(alerts),
            "state_filter": state,
            "timestamp": datetime.utcnow().isoformat(),
            "sources": ["FEMA", "National Weather Service"]
        }
    except Exception as e:
        logger.error(f"Failed to get emergency alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/government/earthquakes")
async def get_earthquake_data(request: Request, magnitude: float = 2.5, limit: int = 20):
    """Get recent earthquake data from USGS"""
    if not government_data_service:
        raise HTTPException(status_code=503, detail="Government data service unavailable")
    
    try:
        earthquakes = await government_data_service.get_earthquake_data(
            min_magnitude=magnitude, 
            limit=limit
        )
        return {
            "earthquakes": earthquakes,
            "total_count": len(earthquakes),
            "magnitude_filter": magnitude,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "USGS Earthquake Hazards Program"
        }
    except Exception as e:
        logger.error(f"Failed to get earthquake data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/government/hospitals")
async def get_hospital_data(request: Request, state: str, limit: int = 50):
    """Get hospital data by state from CMS"""
    if not government_data_service:
        raise HTTPException(status_code=503, detail="Government data service unavailable")
    
    try:
        hospitals = await government_data_service.get_hospital_data(state=state, limit=limit)
        return {
            "hospitals": hospitals,
            "total_count": len(hospitals),
            "state": state,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "Centers for Medicare & Medicaid Services"
        }
    except Exception as e:
        logger.error(f"Failed to get hospital data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/government/data/{simulation_id}")
async def get_government_data_for_simulation(request: Request, simulation_id: str):
    """Get relevant government data for a specific simulation"""
    # Check if simulation exists
    if simulation_id not in simulation_cache and simulation_id not in active_orchestrators:
        try:
            simulation_data = await cloud.firestore.get_simulation_state(simulation_id)
            if not simulation_data:
                raise HTTPException(status_code=404, detail="Simulation not found")
            simulation_cache[simulation_id] = simulation_data
        except Exception:
            raise HTTPException(status_code=404, detail="Simulation not found")
    
    if not government_data_service:
        return {
            "simulation_id": simulation_id,
            "government_data_available": False,
            "message": "Government data service not available",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        simulation_data = simulation_cache.get(simulation_id, {})
        location = simulation_data.get("location", "")
        disaster_type = simulation_data.get("disaster_type", "")
        
        # Get relevant data based on simulation context
        relevant_data = await government_data_service.get_simulation_context_data(
            location=location,
            disaster_type=disaster_type
        )
        
        return {
            "simulation_id": simulation_id,
            "government_data_available": True,
            "location": location,
            "disaster_type": disaster_type,
            "data": relevant_data,
            "timestamp": datetime.utcnow().isoformat(),
            "sources": ["FEMA", "USGS", "NWS", "CMS"]
        }
        
    except Exception as e:
        logger.error(f"Failed to get government data for simulation {simulation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== WEBSOCKET ENDPOINT =====
@app.websocket("/ws/metrics/{simulation_id}")
async def websocket_metrics(websocket: WebSocket, simulation_id: str):
    """Enhanced WebSocket endpoint with heartbeat and error handling"""
    await websocket_manager.connect(websocket, simulation_id)
    
    try:
        # Send initial state
        metrics_calculator = DynamicMetricsCalculator(simulation_id, active_orchestrators.get(simulation_id))
        initial_metrics = metrics_calculator.calculate_real_time_metrics()
        
        await websocket.send_json({
            "type": "initial_state",
            "simulation_id": simulation_id,
            "dashboard_metrics": initial_metrics,
            "orchestrator_info": {
            "ai_model": "Gemini 2.0 Flash",
            "version": "0.5.0",
            "real_time_enabled": True,
            "total_agents": 10,
            "government_data_enabled": government_data_service is not None
            },
            "timestamp": datetime.utcnow().isoformat(),
            "connection_id": id(websocket)
        })
        
        # Main message loop
        while True:
            try:
                # Wait for message with timeout for heartbeat
                message = await asyncio.wait_for(
                    websocket.receive_json(), 
                    timeout=settings.WEBSOCKET_HEARTBEAT_INTERVAL
                )
                
                # Handle client requests
                await handle_websocket_message(websocket, simulation_id, message)
                
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                    "simulation_id": simulation_id
                })
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected for simulation {simulation_id}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error for {simulation_id}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e),
                "simulation_id": simulation_id,
                "timestamp": datetime.utcnow().isoformat()
            })
        except:
            pass
    finally:
        websocket_manager.disconnect(websocket, simulation_id)

async def handle_websocket_message(websocket: WebSocket, simulation_id: str, message: dict):
    """Handle incoming WebSocket messages"""
    message_type = message.get("type")
    
    if message_type == "request_update":
        # Send current metrics
        metrics_calculator = DynamicMetricsCalculator(simulation_id, active_orchestrators.get(simulation_id))
        current_metrics = metrics_calculator.calculate_real_time_metrics()
        
        await websocket.send_json({
            "type": "metrics_update",
            "simulation_id": simulation_id,
            "dashboard_metrics": current_metrics,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    elif message_type == "ping":
        # Respond to ping
        await websocket.send_json({
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat(),
            "simulation_id": simulation_id
        })
    
    else:
        logger.warning(f"Unknown WebSocket message type: {message_type}")

# ===== BACKGROUND TASKS =====
async def start_metrics_streaming(simulation_id: str):
    """Background task for real-time metrics streaming"""
    try:
        await asyncio.sleep(5)  # Initial delay
        
        while simulation_id in active_orchestrators or simulation_id in simulation_cache:
            try:
                # Calculate metrics
                orchestrator = active_orchestrators.get(simulation_id)
                metrics_calculator = DynamicMetricsCalculator(simulation_id, orchestrator)
                metrics = metrics_calculator.calculate_real_time_metrics()
                
                # Broadcast via WebSocket
                broadcast_data = {
                    "type": "metrics_update",
                    "simulation_id": simulation_id,
                    "dashboard_metrics": metrics,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Add government data if available
                if government_data_service and simulation_id in simulation_cache:
                    try:
                        sim_data = simulation_cache[simulation_id]
                        gov_data = await government_data_service.get_quick_updates(
                            location=sim_data.get("location", ""),
                            disaster_type=sim_data.get("disaster_type", "")
                        )
                        broadcast_data["government_updates"] = gov_data
                    except Exception as e:
                        logger.warning(f"Failed to get government updates: {e}")

                await websocket_manager.broadcast_to_simulation(simulation_id, broadcast_data)

                # Update cache
                if simulation_id in simulation_cache:
                    simulation_cache[simulation_id]["last_metrics"] = metrics
                
                # Save to persistent storage periodically
                try:
                    await cloud.firestore.save_simulation_state(simulation_id, {
                        "last_metrics": metrics,
                        "last_metrics_update": datetime.utcnow().isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Failed to save metrics: {e}")
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Metrics streaming error for {simulation_id}: {e}")
                await asyncio.sleep(10)
        
        logger.info(f"Metrics streaming completed for simulation {simulation_id}")
        
    except Exception as e:
        logger.error(f"Metrics streaming initialization failed: {e}")

async def start_government_data_monitoring(simulation_id: str):
    """Background task for real government data monitoring and updates"""
    try:
        await asyncio.sleep(10)  # Wait for initialization
        
        if not government_data_service:
            logger.warning(f"Government data service not available for simulation {simulation_id}")
            return
        
        while simulation_id in active_orchestrators or simulation_id in simulation_cache:
            try:
                simulation_data = simulation_cache.get(simulation_id, {})
                location = simulation_data.get("location", "")
                disaster_type = simulation_data.get("disaster_type", "")
                
                # Get real government updates
                updates = {}
                
                # Get emergency alerts from FEMA/NWS
                try:
                    state = location.split(",")[-1].strip() if "," in location else None
                    alerts = await government_data_service.get_emergency_alerts(state=state, limit=5)
                    if alerts:
                        updates["emergency_alerts"] = alerts
                except Exception as e:
                    logger.warning(f"Failed to get emergency alerts: {e}")
                
                # Get earthquake data if relevant
                if disaster_type in ["earthquake", "tsunami"]:
                    try:
                        earthquakes = await government_data_service.get_earthquake_data(min_magnitude=2.0, limit=10)
                        if earthquakes:
                            updates["earthquake_data"] = earthquakes
                    except Exception as e:
                        logger.warning(f"Failed to get earthquake data: {e}")
                
                # Get hospital data for the area
                if state:
                    try:
                        hospitals = await government_data_service.get_hospital_data(state=state, limit=20)
                        if hospitals:
                            updates["hospital_data"] = hospitals
                    except Exception as e:
                        logger.warning(f"Failed to get hospital data: {e}")
                
                # Broadcast real government updates
                if updates:
                    await websocket_manager.broadcast_to_simulation(simulation_id, {
                        "type": "government_data_update",
                        "simulation_id": simulation_id,
                        "updates": updates,
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": "real_government_apis"
                    })
                    
                    logger.info(f"Broadcasted real government data updates for simulation {simulation_id}")
                
                # Check for updates every 2 minutes (government data doesn't change as frequently)
                await asyncio.sleep(120)
                
            except Exception as e:
                logger.error(f"Government data monitoring error for {simulation_id}: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
        
        logger.info(f"Government data monitoring completed for simulation {simulation_id}")
        
    except Exception as e:
        logger.error(f"Government data monitoring initialization failed: {e}")

async def run_orchestrated_simulation(orchestrator: Any, simulation_id: str):
    """Background task for running orchestrated simulation"""
    try:
        logger.info(f"Starting orchestrated simulation {simulation_id}")
        
        # Update status
        simulation_cache[simulation_id]["status"] = "running"
        simulation_cache[simulation_id]["orchestrator_started"] = datetime.utcnow().isoformat()
        
        try:
            await cloud.firestore.save_simulation_state(simulation_id, {
                "status": "running",
                "orchestrator_started": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to update status: {e}")
        
        # Run simulation
        if hasattr(orchestrator, 'start_simulation'):
            await orchestrator.start_simulation()
        else:
            # Fallback simulation
            await asyncio.sleep(60)  # Simulate 1 minute of processing
        
        # Mark as completed
        simulation_cache[simulation_id]["status"] = "completed"
        simulation_cache[simulation_id]["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Simulation {simulation_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Simulation {simulation_id} failed: {e}")
        
        # Update error status
        simulation_cache[simulation_id]["status"] = "failed"
        simulation_cache[simulation_id]["error"] = str(e)
        simulation_cache[simulation_id]["failed_at"] = datetime.utcnow().isoformat()
        
    finally:
        # Cleanup
        if simulation_id in active_orchestrators:
            del active_orchestrators[simulation_id]
        
        health_status.active_simulations = len(active_orchestrators)
        logger.info(f"Cleaned up simulation {simulation_id}")

async def stop_simulation(simulation_id: str):
    """Stop and clean up a simulation"""
    try:
        # Update status
        if simulation_id in simulation_cache:
            simulation_cache[simulation_id]["status"] = "stopped"
            simulation_cache[simulation_id]["stopped_at"] = datetime.utcnow().isoformat()
        
        # Remove from active orchestrators
        if simulation_id in active_orchestrators:
            orchestrator = active_orchestrators[simulation_id]
            # Graceful shutdown if supported
            if hasattr(orchestrator, 'stop_simulation'):
                await orchestrator.stop_simulation()
            del active_orchestrators[simulation_id]
        
        # Close WebSocket connections
        if simulation_id in websocket_manager.connections:
            for ws in websocket_manager.connections[simulation_id]:
                try:
                    await ws.close(code=1000, reason="Simulation stopped")
                except:
                    pass
            del websocket_manager.connections[simulation_id]
        
        # Update persistent storage
        try:
            await cloud.firestore.save_simulation_state(simulation_id, {
                "status": "stopped",
                "stopped_at": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to update stopped status: {e}")
        
        health_status.active_simulations = len(active_orchestrators)
        logger.info(f"Simulation {simulation_id} stopped successfully")
        
    except Exception as e:
        logger.error(f"Error stopping simulation {simulation_id}: {e}")

# ===== ADMIN ENDPOINTS =====
@app.post("/admin/stop/{simulation_id}")
async def admin_stop_simulation(
    simulation_id: str,
    authenticated: bool = Depends(verify_api_key)
):
    """Admin endpoint to stop a simulation"""
    if simulation_id not in active_orchestrators and simulation_id not in simulation_cache:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    await stop_simulation(simulation_id)
    
    return {
        "message": f"Simulation {simulation_id} stopped successfully",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/admin/simulations")
async def admin_list_simulations(authenticated: bool = Depends(verify_api_key)):
    """Admin endpoint to list all simulations"""
    simulations = []
    
    # Add active simulations
    for sim_id, orchestrator in active_orchestrators.items():
        sim_data = simulation_cache.get(sim_id, {})
        try:
            status_info = orchestrator.get_simulation_status() if hasattr(orchestrator, 'get_simulation_status') else {}
        except Exception:
            status_info = {}
            
        simulations.append({
            "simulation_id": sim_id,
            "status": "active",
            "orchestrator_active": True,
            "created_at": sim_data.get("created_at"),
            "disaster_type": sim_data.get("disaster_type"),
            "location": sim_data.get("location"),
            "websocket_connections": len(websocket_manager.connections.get(sim_id, [])),
            "current_phase": status_info.get("current_phase", "unknown"),
            "government_data_enabled": government_data_service is not None,
            "use_government_data": sim_data.get("use_government_data", False)
        })
    
    # Add cached simulations
    for sim_id, sim_data in simulation_cache.items():
        if sim_id not in active_orchestrators:
            simulations.append({
                "simulation_id": sim_id,
                "status": sim_data.get("status", "unknown"),
                "orchestrator_active": False,
                "created_at": sim_data.get("created_at"),
                "disaster_type": sim_data.get("disaster_type"),
                "location": sim_data.get("location"),
                "websocket_connections": 0,
                "completed_at": sim_data.get("completed_at"),
                "error": sim_data.get("error"),
                "government_data_enabled": government_data_service is not None,
                "use_government_data": sim_data.get("use_government_data", False)
            })
    
    return {
        "simulations": simulations,
        "total_count": len(simulations),
        "active_count": len(active_orchestrators),
        "cached_count": len(simulation_cache),
        "government_data_available": government_data_service is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/admin/cleanup")
async def admin_cleanup(authenticated: bool = Depends(verify_api_key)):
    """Admin endpoint for system cleanup"""
    cleanup_results = {
        "stopped_simulations": 0,
        "cleaned_websockets": 0,
        "cleared_cache_entries": 0,
        "government_data_cleanup": False
    }
    
    # Stop all simulations
    for sim_id in list(active_orchestrators.keys()):
        await stop_simulation(sim_id)
        cleanup_results["stopped_simulations"] += 1
    
    # Clean up stale WebSocket connections
    await websocket_manager.cleanup_stale_connections()
    cleanup_results["cleaned_websockets"] = sum(len(conns) for conns in websocket_manager.connections.values())
    
    # Clear old cache entries
    cutoff_time = datetime.utcnow() - timedelta(hours=24)
    old_entries = []
    
    for sim_id, sim_data in simulation_cache.items():
        created_at = sim_data.get("created_at")
        if created_at:
            try:
                created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                if created_time < cutoff_time:
                    old_entries.append(sim_id)
            except:
                pass
    
    for sim_id in old_entries:
        del simulation_cache[sim_id]
        cleanup_results["cleared_cache_entries"] += 1
    
    # Clean up government data service if available
    if government_data_service:
        try:
            if hasattr(government_data_service, 'cleanup_cache'):
                await government_data_service.cleanup_cache()
            cleanup_results["government_data_cleanup"] = True
        except Exception as e:
            logger.warning(f"Government data cleanup failed: {e}")
    
    return {
        "message": "System cleanup completed",
        "results": cleanup_results,
        "government_data_service_available": government_data_service is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

# === MONITORING ENDPOINTS ===
@app.get("/metrics")
async def get_system_metrics():
    """Get detailed system metrics for monitoring"""
    # Calculate government data metrics
    government_data_simulations = 0
    if government_data_service:
        government_data_simulations = len([
            s for s in simulation_cache.values() 
            if s.get("use_government_data", False)
        ])
    
    return {
        "system_health": health_status.get_health_metrics(),
        "government_apis": {
            "service_available": government_data_service is not None,
            "service_initialized": PRODUCTION_CONFIG_AVAILABLE,
            "last_health_check": datetime.utcnow().isoformat() if government_data_service else None,
            "apis_monitored": ["FEMA", "USGS", "NWS", "CMS"] if government_data_service else [],
            "simulations_using_gov_data": government_data_simulations
        },
        "simulations": {
            "active_count": len(active_orchestrators),
            "total_cached": len(simulation_cache),
            "max_concurrent": settings.MAX_CONCURRENT_SIMULATIONS,
            "government_data_simulations": government_data_simulations
        },
        "websockets": {
            "total_connections": health_status.websocket_connections,
            "connections_by_simulation": {
                sim_id: len(conns) for sim_id, conns in websocket_manager.connections.items()
            }
        },
        "performance": {
            "uptime_seconds": int((datetime.utcnow() - health_status.startup_time).total_seconds()),
            "requests_per_second": health_status.total_requests / max(1, int((datetime.utcnow() - health_status.startup_time).total_seconds())),
            "error_rate_percent": (health_status.failed_requests / max(1, health_status.total_requests)) * 100
        },
        "configuration": {
            "environment": settings.ENVIRONMENT,
            "rate_limit": f"{settings.RATE_LIMIT_REQUESTS}/{settings.RATE_LIMIT_WINDOW}s",
            "websocket_heartbeat": f"{settings.WEBSOCKET_HEARTBEAT_INTERVAL}s",
            "production_config_loaded": PRODUCTION_CONFIG_AVAILABLE,
            "government_data_integration": government_data_service is not None
        },
        "features": {
            "real_time_metrics": True,
            "websocket_streaming": True,
            "government_data_integration": government_data_service is not None,
            "production_ready": settings.ENVIRONMENT == "production"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# === ERROR HANDLERS ===
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced HTTP exception handler"""
    request_id = str(uuid.uuid4())
    
    # Add context for government data related errors
    error_context = {
        "government_data_available": government_data_service is not None,
        "production_config_loaded": PRODUCTION_CONFIG_AVAILABLE
    }
    
    # Check if this is a government data related endpoint
    if "/government/" in str(request.url.path):
        error_context.update({
            "error_type": "government_data_error",
            "service_status": "available" if government_data_service else "unavailable"
        })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path),
            "context": error_context
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unexpected errors"""
    request_id = str(uuid.uuid4())
    
    logger.error(f"Unhandled exception {request_id}: {exc}", exc_info=True)
    
    # Enhanced error context for debugging
    error_context = {
        "government_data_available": government_data_service is not None,
        "production_config_loaded": PRODUCTION_CONFIG_AVAILABLE,
        "active_simulations": len(active_orchestrators),
        "environment": settings.ENVIRONMENT
    }
    
    # Check if error might be government data related
    error_str = str(exc).lower()
    if any(keyword in error_str for keyword in ["government", "fema", "usgs", "nws", "cms"]):
        error_context.update({
            "potential_government_data_error": True,
            "suggestion": "Check government data service configuration and API connectivity"
        })
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path),
            "context": error_context
        }
    )

# ===== PERIODIC CLEANUP TASK =====
async def periodic_cleanup():
    """Periodic cleanup task with government data maintenance"""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            
            # Clean up stale WebSocket connections
            await websocket_manager.cleanup_stale_connections()
            
            # Clean up old simulations
            cutoff_time = datetime.utcnow() - timedelta(hours=6)
            old_simulations = []
            
            for sim_id, sim_data in simulation_cache.items():
                if sim_id not in active_orchestrators:
                    created_at = sim_data.get("created_at")
                    if created_at:
                        try:
                            created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            if created_time < cutoff_time:
                                old_simulations.append(sim_id)
                        except:
                            pass
            
            for sim_id in old_simulations:
                del simulation_cache[sim_id]
                logger.info(f"Cleaned up old simulation cache: {sim_id}")
            
            # Government data service maintenance
            if government_data_service:
                try:
                    # Clean up government data cache if method exists
                    if hasattr(government_data_service, 'cleanup_cache'):
                        await government_data_service.cleanup_cache()
                        logger.debug("Government data cache cleanup completed")
                    
                    # Refresh API health status
                    if hasattr(government_data_service, 'refresh_api_status'):
                        await government_data_service.refresh_api_status()
                        logger.debug("Government API status refreshed")
                        
                except Exception as e:
                    logger.warning(f"Government data service maintenance failed: {e}")
            
            logger.debug(f"Periodic cleanup completed. Removed {len(old_simulations)} old simulations")
            
        except Exception as e:
            logger.error(f"Periodic cleanup error: {e}")

# Start periodic cleanup task
@app.on_event("startup")
async def startup_event():
    """Additional startup tasks"""
    asyncio.create_task(periodic_cleanup())
    logger.info("Periodic cleanup task started")
    
    # Log government data service status at startup
    if government_data_service:
        logger.info("✅ Government data service available for periodic maintenance")
    else:
        logger.info("⚠️ Government data service not available - skipping gov data maintenance")

# ===== DEVELOPMENT/TESTING ENDPOINTS =====
if settings.DEBUG:
    @app.get("/debug/cache")
    async def debug_cache():
        """Debug endpoint to view cache contents"""
        return {
            "simulation_cache": {
                sim_id: {
                    "status": data.get("status"),
                    "created_at": data.get("created_at"),
                    "disaster_type": data.get("disaster_type"),
                    "use_government_data": data.get("use_government_data", False),
                    "location": data.get("location")
                } for sim_id, data in simulation_cache.items()
            },
            "active_orchestrators": list(active_orchestrators.keys()),
            "websocket_connections": {
                sim_id: len(conns) for sim_id, conns in websocket_manager.connections.items()
            },
            "government_data": {
                "service_available": government_data_service is not None,
                "production_config_loaded": PRODUCTION_CONFIG_AVAILABLE,
                "simulations_using_gov_data": len([
                    s for s in simulation_cache.values() 
                    if s.get("use_government_data", False)
                ])
            }
        }
    
    @app.get("/debug/government-data")
    async def debug_government_data():
        """Debug endpoint for government data service status"""
        if not government_data_service:
            return {
                "error": "Government data service not available",
                "production_config_loaded": PRODUCTION_CONFIG_AVAILABLE,
                "environment": settings.ENVIRONMENT
            }
        
        try:
            # Test API connectivity
            health_check = await government_data_service.check_apis_health()
            return {
                "service_available": True,
                "api_health": health_check,
                "production_config_loaded": PRODUCTION_CONFIG_AVAILABLE,
                "environment": settings.ENVIRONMENT,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "service_available": True,
                "error": str(e),
                "message": "Government data service available but health check failed",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @app.post("/debug/simulate-load")
    async def debug_simulate_load(num_simulations: int = 3, use_government_data: bool = True):
        """Debug endpoint to create multiple test simulations"""
        created_simulations = []
        
        for i in range(min(num_simulations, 5)):  # Limit to 5 for safety
            test_request = SimulationRequest(
                disaster_type="flood",
                location=f"Test Location {i+1}",
                severity=5,
                duration=24,
                use_government_data=use_government_data
            )
            
            try:
                response = await start_simulation(
                    request=None,  # Mock request
                    simulation_request=test_request,
                    background_tasks=BackgroundTasks(),
                    authenticated=True
                )
                created_simulations.append(response.simulation_id)
            except Exception as e:
                logger.error(f"Failed to create test simulation {i+1}: {e}")
        
        return {
            "created_simulations": created_simulations,
            "total_created": len(created_simulations),
            "active_simulations": len(active_orchestrators),
            "government_data_enabled": use_government_data,
            "government_service_available": government_data_service is not None
        }

# === ORCHESTRATOR ENDPOINTS ===

@app.get("/orchestrator/{simulation_id}/status")
async def get_orchestrator_status(simulation_id: str):
    """Get orchestrator-specific status"""
    
    orchestrator = active_orchestrators.get(simulation_id)
    
    if not orchestrator:
        simulation_data = simulation_cache.get(simulation_id)
        
        if not simulation_data:
            try:
                simulation_data = await cloud.firestore.get_simulation_state(simulation_id)
                if simulation_data:
                    simulation_cache[simulation_id] = simulation_data
            except Exception as e:
                logger.warning(f"Failed to retrieve simulation: {e}")
        
        if simulation_data:
            return {
                "simulation_id": simulation_id,
                "orchestrator_active": False,
                "simulation_exists": True,
                "simulation_status": simulation_data.get("status", "unknown"),
                "error": "Orchestrator not running for this simulation",
                "fallback_mode": True,
                "message": "Simulation exists but orchestrator is not active. Basic simulation mode.",
                "government_data_enabled": simulation_data.get("use_government_data", False),
                "government_service_available": government_data_service is not None
            }
        else:
            raise HTTPException(status_code=404, detail="Simulation not found")
    
    try:
        orchestrator_status = orchestrator.get_simulation_status()
        agent_info = orchestrator.get_all_agent_info()
        performance = orchestrator.get_performance_report()
        
        # Get simulation data for government data status
        sim_data = simulation_cache.get(simulation_id, {})
        
        return {
            "simulation_id": simulation_id,
            "orchestrator_active": True,
            "status": orchestrator_status,
            "agent_info": agent_info,
            "performance": performance,
            "fallback_mode": False,
            "real_time_metrics": orchestrator.get_real_time_metrics(),
            "government_data_enabled": sim_data.get("use_government_data", False),
            "government_service_available": government_data_service is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting orchestrator status: {e}")
        sim_data = simulation_cache.get(simulation_id, {})
        return {
            "simulation_id": simulation_id,
            "orchestrator_active": True,
            "error": str(e),
            "fallback_mode": True,
            "message": "Orchestrator active but status retrieval failed",
            "government_data_enabled": sim_data.get("use_government_data", False),
            "government_service_available": government_data_service is not None
        }

@app.get("/orchestrator/{simulation_id}/metrics")
async def get_orchestrator_metrics(simulation_id: str):
    """Get real-time orchestrator metrics"""
    
    orchestrator = active_orchestrators.get(simulation_id)
    
    if not orchestrator:
        simulation_data = simulation_cache.get(simulation_id)
        if not simulation_data:
            try:
                simulation_data = await cloud.firestore.get_simulation_state(simulation_id)
            except Exception:
                pass
        
        if not simulation_data:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        metrics_calculator = DynamicMetricsCalculator(simulation_id, None)
        fallback_metrics = metrics_calculator.calculate_real_time_metrics()
        
        return {
            "simulation_id": simulation_id,
            "orchestrator_active": False,
            "real_time_metrics": fallback_metrics,
            "metrics_history": [],
            "fallback_mode": True,
            "message": "Using fallback metrics - orchestrator not active",
            "government_data_enabled": simulation_data.get("use_government_data", False),
            "government_service_available": government_data_service is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        sim_data = simulation_cache.get(simulation_id, {})
        
        return {
            "simulation_id": simulation_id,
            "orchestrator_active": True,
            "real_time_metrics": orchestrator.get_real_time_metrics(),
            "metrics_history": orchestrator.get_metrics_history(limit=50),
            "fallback_mode": False,
            "government_data_enabled": sim_data.get("use_government_data", False),
            "government_service_available": government_data_service is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting orchestrator metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulations")
async def list_all_simulations():
    """List all simulations (active and cached)"""
    simulations = []
    
    for sim_id, orchestrator in active_orchestrators.items():
        sim_data = simulation_cache.get(sim_id, {})
        try:
            status_info = orchestrator.get_simulation_status() if hasattr(orchestrator, 'get_simulation_status') else {}
        except Exception:
            status_info = {}
            
        simulations.append({
            "simulation_id": sim_id,
            "status": "active",
            "orchestrator_active": True,
            "created_at": sim_data.get("created_at"),
            "disaster_type": sim_data.get("disaster_type"),
            "location": sim_data.get("location"),
            "websocket_connections": len(websocket_manager.connections.get(sim_id, [])),
            "current_phase": status_info.get("current_phase", "unknown"),
            "government_data_enabled": sim_data.get("use_government_data", False)
        })
    
    for sim_id, sim_data in simulation_cache.items():
        if sim_id not in active_orchestrators:
            simulations.append({
                "simulation_id": sim_id,
                "status": sim_data.get("status", "unknown"),
                "orchestrator_active": False,
                "created_at": sim_data.get("created_at"),
                "disaster_type": sim_data.get("disaster_type"),
                "location": sim_data.get("location"),
                "websocket_connections": 0,
                "completed_at": sim_data.get("completed_at"),
                "error": sim_data.get("error"),
                "government_data_enabled": sim_data.get("use_government_data", False)
            })
    
    return {
        "simulations": simulations,
        "total_count": len(simulations),
        "active_count": len(active_orchestrators),
        "cached_count": len(simulation_cache),
        "government_service_available": government_data_service is not None,
        "government_data_simulations": len([
            s for s in simulations if s.get("government_data_enabled", False)
        ]),
        "timestamp": datetime.utcnow().isoformat()
    }

# === LEGACY COMPATIBILITY ENDPOINTS ===
@app.get("/ping")
async def ping(request: Request):
    """Ping endpoint for health checks"""
    return {
        "message": "pong",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.5.0",
        "environment": settings.ENVIRONMENT,
        "orchestrator": "Gemini 2.0 Flash",
        "agents": 10,
        "server_ready": True,
        "active_simulations": len(active_orchestrators),
        "websocket_connections": health_status.websocket_connections,
        "government_data_available": government_data_service is not None,
        "production_config_loaded": PRODUCTION_CONFIG_AVAILABLE
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ERIS Emergency Response Intelligence System",
        "version": "0.5.0",
        "environment": settings.ENVIRONMENT,
        "api_documentation": "/docs" if settings.DEBUG else "Contact admin for API documentation",
        "orchestrator": {
            "ai_model": "Gemini 2.0 Flash",
            "architecture": "10-agent coordination system",
            "total_agents": 10,
            "adk_agents": 6,
            "enhanced_agents": 4
        },
        "features": {
            "real_time_metrics": True,
            "websocket_streaming": True,
            "government_data_integration": government_data_service is not None,
            "real_time_emergency_alerts": government_data_service is not None,
            "earthquake_monitoring": government_data_service is not None,
            "hospital_data_integration": government_data_service is not None,
            "production_ready": True,
            "rate_limiting": True,
            "authentication": settings.ENVIRONMENT == "production",
            "monitoring": True
        },
        "government_apis": {
            "fema_integration": government_data_service is not None,
            "usgs_earthquake_data": government_data_service is not None,
            "nws_weather_alerts": government_data_service is not None,
            "cms_hospital_data": government_data_service is not None
        },
        "endpoints": {
            "health": "/health",
            "system_info": "/system/info",
            "start_simulation": "/simulate",
            "websocket_metrics": "/ws/metrics/{simulation_id}",
            "dashboard_metrics": "/metrics/dashboard/{simulation_id}",
            "live_feed": "/live-feed/{simulation_id}",
            "government_health": "/government/health",
            "emergency_alerts": "/government/alerts",
            "earthquake_data": "/government/earthquakes",
            "hospital_data": "/government/hospitals"
        },
        "limits": {
            "max_concurrent_simulations": settings.MAX_CONCURRENT_SIMULATIONS,
            "rate_limit": f"{settings.RATE_LIMIT_REQUESTS} requests per minute"
        },
        "current_load": {
            "active_simulations": len(active_orchestrators),
            "websocket_connections": health_status.websocket_connections,
            "uptime_seconds": int((datetime.utcnow() - health_status.startup_time).total_seconds()),
            "government_data_simulations": len([
                s for s in simulation_cache.values() 
                if s.get("use_government_data", False)
            ])
        },
        "data_sources": {
            "simulation_metrics": "Real-time AI orchestrator",
            "emergency_alerts": "FEMA & National Weather Service" if government_data_service else "Not available",
            "earthquake_data": "USGS Earthquake Hazards Program" if government_data_service else "Not available",
            "hospital_data": "Centers for Medicare & Medicaid Services" if government_data_service else "Not available"
        }
    }

# ===== GRACEFUL SHUTDOWN =====
@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown procedures"""
    logger.info("🛑 Initiating graceful shutdown...")
    
    # Stop accepting new connections
    for sim_id in list(active_orchestrators.keys()):
        try:
            await stop_simulation(sim_id)
        except Exception as e:
            logger.error(f"Error stopping simulation {sim_id} during shutdown: {e}")
    
    # Close all WebSocket connections
    for sim_id, connections in websocket_manager.connections.items():
        for ws in connections:
            try:
                await ws.close(code=1001, reason="Server shutdown")
            except:
                pass
    
    # Graceful government data service shutdown
    if government_data_service:
        try:
            logger.info("🔄 Shutting down government data service...")
            if hasattr(government_data_service, 'session') and government_data_service.session:
                await government_data_service.session.close()
            if hasattr(government_data_service, 'cleanup'):
                await government_data_service.cleanup()
            logger.info("✅ Government data service shutdown completed")
        except Exception as e:
            logger.error(f"Error during government data service shutdown: {e}")
    
    logger.info("✅ Graceful shutdown completed")

# ===== APPLICATION CONFIGURATION =====
def configure_logging():
    """Configure structured logging"""
    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            # Add file handler for production
            *([logging.FileHandler('eris.log')] if settings.ENVIRONMENT == "production" else [])
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    
    # Set appropriate logging levels for government data APIs
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Reduce HTTP client noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)  # Reduce connection pool noise

# Initialize logging
configure_logging()

if __name__ == "__main__":
    # Log startup information
    logger.info("Starting ERIS Emergency Response Intelligence System")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Government Data Integration: {'✅ Enabled' if government_data_service else '⚠️ Disabled'}")
    logger.info(f"Production Config: {'✅ Loaded' if PRODUCTION_CONFIG_AVAILABLE else '⚠️ Not loaded'}")
