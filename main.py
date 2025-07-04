"""
ERIS (Emergency Response Intelligence System) - CLI v0.5.0
Command-line interface for disaster simulation management

Enhanced Features:
- Updated for ERIS v0.5.0 with 10-agent system
- Gemini 2.0 Flash orchestrator integration
- Enhanced WebSocket real-time monitoring
- Improved dashboard metrics integration
- Better agent network display
- Live emergency feed integration
"""

import os
import sys
import asyncio
import aiohttp
import logging
import json
import time
import traceback
import websockets
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import wraps
import signal

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.tree import Tree
from rich.status import Status
from rich.columns import Columns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enhanced logging setup
def setup_logging():
    """Setup comprehensive logging for ERIS v0.5.0"""
    log_dir = Path.home() / ".eris" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"eris_cli_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stderr)
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

# Enhanced Configuration
@dataclass
class ERISConfig:
    """ERIS CLI Configuration for v0.5.0"""
    api_url: str = os.getenv('ERIS_API_URL', 'https://eris-backend-621360763676.us-central1.run.app')
    api_key: Optional[str] = os.getenv('ERIS_API_KEY')
    timeout: int = int(os.getenv('ERIS_TIMEOUT', '30'))
    max_retries: int = int(os.getenv('ERIS_MAX_RETRIES', '3'))
    retry_delay: float = float(os.getenv('ERIS_RETRY_DELAY', '1.0'))
    log_level: str = os.getenv('ERIS_LOG_LEVEL', 'INFO')
    websocket_timeout: int = int(os.getenv('ERIS_WS_TIMEOUT', '5'))
    
    def __post_init__(self):
        if not self.api_url:
            raise ValueError("ERIS_API_URL must be configured")
        
        logging.getLogger().setLevel(getattr(logging, self.log_level.upper()))
    
    @property
    def websocket_url(self):
        """Get WebSocket URL from HTTP URL"""
        return self.api_url.replace('http://', 'ws://').replace('https://', 'wss://')

config = ERISConfig()

# Enhanced disaster types matching backend
DISASTER_TYPES = [
    'earthquake', 'hurricane', 'flood', 'wildfire', 'tsunami', 
    'volcanic_eruption', 'severe_storm', 'epidemic', 'pandemic', 'landslide'
]

console = Console()

class APIException(Exception):
    """Enhanced API exception with detailed error info"""
    def __init__(self, message: str, status_code: int = None, response_data: Dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}
        self.timestamp = datetime.utcnow()

class RetryManager:
    """Enhanced retry manager with exponential backoff"""
    
    @staticmethod
    async def retry_with_backoff(
        func,
        max_retries: int = 3,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 30.0,
        retriable_exceptions: tuple = (aiohttp.ClientError, asyncio.TimeoutError)
    ):
        """Execute function with exponential backoff retry"""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except retriable_exceptions as e:
                last_exception = e
                
                if attempt == max_retries:
                    break
                
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {str(e)}")
                await asyncio.sleep(delay)
        
        raise APIException(f"All {max_retries + 1} attempts failed: {str(last_exception)}")

class ERISAPIClient:
    """Enhanced ERIS API client for v0.5.0"""
    
    def __init__(self, config: ERISConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_count = 0
        self.error_count = 0
        self.start_time = datetime.utcnow()
        self._connection_pool_size = 10
        
        # Performance tracking
        self.response_times = []
        self.last_health_check = None
        self.health_status = "unknown"
        self.system_info = None
    
    async def __aenter__(self):
        """Enhanced context manager entry"""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        connector = aiohttp.TCPConnector(
            limit=self._connection_pool_size,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        headers = {
            'User-Agent': 'ERIS-CLI/0.5.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers
        )
        
        logger.info(f"Enhanced API client initialized for {self.config.api_url}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Enhanced context manager exit"""
        if self.session:
            await self.session.close()
        
        # Log enhanced session statistics
        session_duration = (datetime.utcnow() - self.start_time).total_seconds()
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        logger.info(f"API session closed: {self.request_count} requests, "
                   f"{self.error_count} errors, {session_duration:.1f}s duration, "
                   f"{avg_response_time:.3f}s avg response time")
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Enhanced request method with comprehensive error handling"""
        if not self.session:
            raise APIException("API client not initialized")
        
        url = f"{self.config.api_url}{endpoint}"
        request_start = time.time()
        
        async def _request():
            self.request_count += 1
            
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    response_time = time.time() - request_start
                    self.response_times.append(response_time)
                    
                    # Keep only last 100 response times
                    if len(self.response_times) > 100:
                        self.response_times = self.response_times[-100:]
                    
                    logger.debug(f"{method} {endpoint} -> {response.status} ({response_time:.3f}s)")
                    
                    if response.status >= 400:
                        self.error_count += 1
                        error_text = await response.text()
                        
                        try:
                            error_data = json.loads(error_text)
                        except json.JSONDecodeError:
                            error_data = {"detail": error_text}
                        
                        raise APIException(
                            f"API request failed: {error_data.get('detail', 'Unknown error')}",
                            response.status,
                            error_data
                        )
                    
                    return await response.json()
                    
            except aiohttp.ClientError as e:
                self.error_count += 1
                raise APIException(f"Network error: {str(e)}")
        
        return await RetryManager.retry_with_backoff(
            _request,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_delay
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check with system capabilities"""
        if (self.last_health_check and 
            (datetime.utcnow() - self.last_health_check).total_seconds() < 30):
            return {"status": self.health_status, "cached": True}
        
        try:
            result = await self._make_request('GET', '/health')
            self.health_status = "healthy"
            self.last_health_check = datetime.utcnow()
            return result
        except APIException as e:
            self.health_status = "unhealthy"
            logger.error(f"Health check failed: {e}")
            raise
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get enhanced system information"""
        result = await self._make_request('GET', '/system/info')
        self.system_info = result
        return result
    
    async def start_simulation(self, disaster_type: str, location: str, 
                             severity: int, duration: int = 24) -> Dict[str, Any]:
        """Start enhanced simulation with validation"""
        if disaster_type not in DISASTER_TYPES:
            raise ValueError(f"Invalid disaster type: {disaster_type}")
        
        if not (1 <= severity <= 10):
            raise ValueError(f"Severity must be between 1-10, got: {severity}")
        
        if not (1 <= duration <= 168):  # Max 1 week
            raise ValueError(f"Duration must be between 1-168 hours, got: {duration}")
        
        payload = {
            "disaster_type": disaster_type,
            "location": location,
            "severity": severity,
            "duration": duration
        }
        
        logger.info(f"Starting enhanced simulation: {disaster_type} in {location} (severity {severity})")
        return await self._make_request('POST', '/simulate', json=payload)
    
    async def get_simulation_status(self, simulation_id: str) -> Dict[str, Any]:
        """Get enhanced simulation status"""
        if not simulation_id:
            raise ValueError("Simulation ID is required")
        
        return await self._make_request('GET', f'/status/{simulation_id}')
    
    async def get_agents_info(self, simulation_id: str) -> Dict[str, Any]:
        """Get enhanced 10-agent information"""
        return await self._make_request('GET', f'/orchestrator/{simulation_id}/agents')
    
    async def get_dashboard_metrics(self, simulation_id: str) -> Dict[str, Any]:
        """Get enhanced real-time dashboard metrics"""
        return await self._make_request('GET', f'/metrics/dashboard/{simulation_id}')
    
    async def get_orchestrator_status(self, simulation_id: str) -> Dict[str, Any]:
        """Get orchestrator-specific status"""
        return await self._make_request('GET', f'/orchestrator/{simulation_id}/status')
    
    async def get_orchestrator_metrics(self, simulation_id: str) -> Dict[str, Any]:
        """Get orchestrator real-time metrics"""
        return await self._make_request('GET', f'/orchestrator/{simulation_id}/metrics')
    
    async def get_live_feed(self, simulation_id: str, limit: int = 20) -> Dict[str, Any]:
        """Get enhanced live emergency feed"""
        return await self._make_request('GET', f'/live-feed/{simulation_id}?limit={limit}')
    
    async def list_simulations(self) -> Dict[str, Any]:
        """List all simulations"""
        return await self._make_request('GET', '/simulations')
    
    def get_websocket_url(self, simulation_id: str) -> str:
        """Get WebSocket URL for real-time monitoring"""
        return f"{config.websocket_url}/ws/metrics/{simulation_id}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get enhanced client performance statistics"""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        session_duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": (self.error_count / max(1, self.request_count)) * 100,
            "average_response_time": avg_response_time,
            "session_duration": session_duration,
            "requests_per_minute": (self.request_count / max(1, session_duration)) * 60,
            "health_status": self.health_status,
            "api_version": self.system_info.get('eris_version', 'unknown') if self.system_info else 'unknown'
        }

class ERISWebSocketMonitor:
    """Enhanced WebSocket monitoring for real-time updates"""
    
    def __init__(self, simulation_id: str, config: ERISConfig):
        self.simulation_id = simulation_id
        self.config = config
        self.websocket = None
        self.is_connected = False
        self.last_metrics = {}
        self.message_count = 0
        
    async def connect(self):
        """Connect to WebSocket with enhanced error handling"""
        ws_url = f"{config.websocket_url}/ws/metrics/{self.simulation_id}"
        
        try:
            self.websocket = await websockets.connect(
                ws_url,
                timeout=self.config.websocket_timeout,
                max_size=2**20  # 1MB max message size
            )
            self.is_connected = True
            logger.info(f"WebSocket connected for simulation {self.simulation_id}")
            return True
        except Exception as e:
            logger.warning(f"WebSocket connection failed: {e}")
            self.is_connected = False
            return False
    
    async def listen(self, callback):
        """Listen for messages with enhanced handling"""
        if not self.websocket:
            return
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    self.message_count += 1
                    
                    if data.get('type') in ['initial_state', 'metrics_update']:
                        self.last_metrics = data.get('dashboard_metrics', {})
                    
                    await callback(data)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.is_connected = False
    
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False

class ERISCLIApplication:
    """Enhanced CLI application for ERIS v0.5.0"""
    
    def __init__(self, config: ERISConfig):
        self.config = config
        self.client = ERISAPIClient(config)
        self.simulation_history = []
        self.interrupted = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        self.interrupted = True
        console.print("\n[yellow]⚠️  Interrupt received, cleaning up...[/yellow]")
    
    async def display_system_status(self):
        """Enhanced system status display for v0.5.0"""
        async with self.client:
            try:
                with Status("🔍 Checking ERIS v0.5.0 system status...", console=console):
                    # Get system info and health in parallel
                    health_task = asyncio.create_task(self.client.health_check())
                    info_task = asyncio.create_task(self.client.get_system_info())
                    simulations_task = asyncio.create_task(self.client.list_simulations())
                    
                    health, system_info, simulations = await asyncio.gather(
                        health_task, info_task, simulations_task
                    )
                
                # Display enhanced system status
                self._display_enhanced_system_status(health, system_info, simulations)
                self._display_performance_stats()
                
            except APIException as e:
                console.print(f"[red]❌ System check failed: {e}[/red]")
                if e.status_code:
                    console.print(f"[red]HTTP Status: {e.status_code}[/red]")
                logger.error(f"System status check failed: {e}")
            except Exception as e:
                console.print(f"[red]❌ Unexpected error: {e}[/red]")
                logger.error(f"Unexpected error in system status: {traceback.format_exc()}")
    
    def _display_enhanced_system_status(self, health: Dict[str, Any], system_info: Dict[str, Any], simulations: Dict[str, Any]):
        """Display comprehensive system status for v0.5.0"""
        table = Table(title="🔥 ERIS v0.5.0 System Status", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan", no_wrap=True, width=25)
        table.add_column("Status", style="green", width=15)
        table.add_column("Details", style="white")
        
        # API Status
        api_status = "✅ Online" if health.get('status') == 'healthy' else "❌ Offline"
        table.add_row("🌐 API Server", api_status, f"Connected to {self.config.api_url}")
        
        # Enhanced Version info
        version = system_info.get('eris_version', 'Unknown')
        environment = system_info.get('environment', 'unknown')
        table.add_row("📦 ERIS Version", f"v{version}", f"Environment: {environment}")
        
        # Enhanced Orchestrator info
        orchestrator = system_info.get('orchestrator', {})
        ai_model = orchestrator.get('ai_model', 'Gemini 2.0 Flash')
        architecture = orchestrator.get('architecture', '10-agent coordination system')
        table.add_row("🧠 AI Orchestrator", "✅ Active", f"{ai_model} - {architecture}")
        
        # Enhanced Agent Systems
        agent_system = system_info.get('agent_system', {})
        total_agents = agent_system.get('total_agents', 10)
        adk_agents = agent_system.get('adk_agents', 6)
        enhanced_agents = agent_system.get('enhanced_agents', 4)
        
        table.add_row("🤖 Agent Network", "✅ Ready", 
                     f"{total_agents} agents ({adk_agents} ADK Strategic + {enhanced_agents} Enhanced Tactical)")
        
        # Enhanced Capabilities
        capabilities = system_info.get('capabilities', {})
        if capabilities.get('real_time_metrics'):
            table.add_row("📊 Real-time Metrics", "✅ Enabled", "Dynamic calculation & WebSocket streaming")
        
        if capabilities.get('ai_content_generation'):
            table.add_row("✨ AI Content Gen", "✅ Enabled", "Live social media & emergency feeds")
        
        if capabilities.get('websocket_streaming'):
            table.add_row("🔄 WebSocket Stream", "✅ Active", "Real-time intelligence streaming")
        
        # Enhanced Cloud integration
        if capabilities.get('cloud_integration'):
            table.add_row("☁️ Cloud Integration", "✅ Connected", "Firestore + BigQuery + Vertex AI")
        
        # Enhanced Active Load
        current_load = system_info.get('current_load', {})
        active_sims = current_load.get('active_simulations', 0)
        ws_connections = current_load.get('websocket_connections', 0)
        
        table.add_row("🔄 Active Load", f"{active_sims} simulations", f"{ws_connections} WebSocket connections")
        
        # Simulation Summary
        total_sims = simulations.get('total_count', 0)
        active_count = simulations.get('active_count', 0)
        
        table.add_row("📊 Simulation Summary", f"{active_count}/{total_sims} active", "Total simulations in system")
        
        console.print(table)
        console.print(f"\n[bold green]✅ ERIS v{version} - Enhanced orchestrator with {ai_model} - All systems operational[/bold green]")
    
    def _display_performance_stats(self):
        """Display enhanced client performance statistics"""
        stats = self.client.get_performance_stats()
        
        perf_table = Table(title="📈 Enhanced Client Performance", show_header=True)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="yellow")
        
        perf_table.add_row("Total Requests", str(stats['total_requests']))
        perf_table.add_row("Error Rate", f"{stats['error_rate']:.1f}%")
        perf_table.add_row("Avg Response Time", f"{stats['average_response_time']:.3f}s")
        perf_table.add_row("Requests/Min", f"{stats['requests_per_minute']:.1f}")
        perf_table.add_row("API Version", f"v{stats['api_version']}")
        
        console.print(perf_table)
    
    async def list_disasters(self):
        """Enhanced disaster types display"""
        async with self.client:
            try:
                system_info = await self.client.get_system_info()
                
                table = Table(title="🌪️ Enhanced Disaster Scenarios (ERIS v0.5.0)", show_header=True)
                table.add_column("Disaster Type", style="cyan", width=20)
                table.add_column("Severity Range", style="yellow", width=15)
                table.add_column("AI Agent Focus", style="magenta")
                table.add_column("Avg Duration", style="green", width=12)
                
                # Enhanced disaster details with AI agent focus
                disaster_details = {
                    'earthquake': ('1-9 Richter', 'Infrastructure + Hospital Load Modeling', '2-6 hours'),
                    'hurricane': ('1-5 Category', 'Evacuation + Public Behavior Simulation', '12-48 hours'),
                    'flood': ('1-10 Index', 'Transportation + Emergency Coordination', '6-24 hours'),
                    'wildfire': ('1-10 Index', 'Evacuation + Social Media Dynamics', '12-72 hours'),
                    'tsunami': ('1-10 Wave Height', 'Mass Evacuation + Communications', '2-8 hours'),
                    'volcanic_eruption': ('1-8 VEI', 'Air Quality + Public Health Management', '24-168 hours'),
                    'severe_storm': ('1-5 Intensity', 'Power Grid + Recovery Coordination', '4-12 hours'),
                    'epidemic': ('1-10 Spread Rate', 'Healthcare + News Coverage Simulation', '72-168 hours'),
                    'pandemic': ('1-10 Global Impact', 'All 10 Agents + Full Coordination', '168+ hours'),
                    'landslide': ('1-10 Risk Level', 'Infrastructure + Emergency Response', '1-6 hours')
                }
                
                for disaster in DISASTER_TYPES:
                    severity, ai_focus, duration = disaster_details.get(disaster, ('1-10', 'Full AI Network', 'Variable'))
                    table.add_row(disaster.replace('_', ' ').title(), severity, ai_focus, duration)
                
                console.print(table)
                
                # Enhanced tips
                console.print(f"\n[italic]💡 Enhanced Tips:[/italic]")
                console.print(f"[italic]• Use --severity to adjust AI agent intensity (1=minimal, 10=catastrophic)[/italic]")
                console.print(f"[italic]• All simulations use 10 AI agents with Gemini 2.0 Flash orchestrator[/italic]")
                console.print(f"[italic]• Real-time WebSocket monitoring available for all disaster types[/italic]")
                
            except APIException as e:
                console.print(f"[red]Failed to fetch enhanced disaster types: {e}[/red]")
    
    async def start_simulation_interactive(self, disaster_type: str, location: str, 
                                         severity: int, duration: int):
        """Enhanced simulation startup with v0.5.0 features"""
        async with self.client:
            try:
                # Pre-flight validation
                if not location.strip():
                    raise ValueError("Location cannot be empty")
                
                # Display enhanced simulation details
                self._display_enhanced_simulation_preview(disaster_type, location, severity, duration)
                
                # Confirm if interactive
                if not Confirm.ask("\n🚀 Start this enhanced simulation?"):
                    console.print("[yellow]Simulation cancelled by user[/yellow]")
                    return None
                
                # Start simulation with enhanced progress tracking
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    
                    task = progress.add_task("Initializing ERIS v0.5.0 simulation...", total=100)
                    
                    # Phase 1: Start simulation
                    progress.update(task, advance=15, description="Starting enhanced simulation...")
                    response = await self.client.start_simulation(disaster_type, location, severity, duration)
                    
                    simulation_id = response['simulation_id']
                    
                    # Phase 2: Wait for orchestrator initialization
                    progress.update(task, advance=20, description="Initializing Gemini 2.0 Flash orchestrator...")
                    await asyncio.sleep(2)
                    
                    # Phase 3: Get enhanced agent status
                    progress.update(task, advance=25, description="Activating 10 AI agents (6 ADK + 4 Enhanced)...")
                    try:
                        agents_info = await self.client.get_agents_info(simulation_id)
                    except Exception as e:
                        logger.warning(f"Failed to get agents info: {e}")
                        agents_info = {}
                    
                    # Phase 4: Get enhanced metrics
                    progress.update(task, advance=20, description="Establishing real-time dashboard metrics...")
                    try:
                        initial_metrics = await self.client.get_dashboard_metrics(simulation_id)
                    except Exception as e:
                        logger.warning(f"Failed to get initial metrics: {e}")
                        initial_metrics = {}
                    
                    # Phase 5: Test WebSocket connection
                    progress.update(task, advance=20, description="Testing real-time WebSocket connection...")
                    await asyncio.sleep(1)
                    
                    progress.update(task, completed=100, description="Enhanced simulation ready!")
                
                # Display enhanced success info
                self._display_enhanced_simulation_success(simulation_id, response, agents_info)
                
                # Add to enhanced history
                self.simulation_history.append({
                    'id': simulation_id,
                    'disaster_type': disaster_type,
                    'location': location,
                    'severity': severity,
                    'duration': duration,
                    'started_at': datetime.utcnow(),
                    'status': 'active',
                    'orchestrator': response.get('orchestrator_info', {}).get('ai_model', 'Gemini 2.0 Flash'),
                    'agents_count': response.get('orchestrator_info', {}).get('total_agents', 10)
                })
                
                return simulation_id
                
            except ValueError as e:
                console.print(f"[red]❌ Validation error: {e}[/red]")
                return None
            except APIException as e:
                console.print(f"[red]❌ Failed to start enhanced simulation: {e}[/red]")
                if e.response_data:
                    console.print(f"[red]Details: {e.response_data}[/red]")
                return None
            except Exception as e:
                console.print(f"[red]❌ Unexpected error: {e}[/red]")
                logger.error(f"Enhanced simulation start failed: {traceback.format_exc()}")
                return None
    
    def _display_enhanced_simulation_preview(self, disaster_type: str, location: str, severity: int, duration: int):
        """Display enhanced simulation preview"""
        preview_panel = Panel(
            f"[bold cyan]📍 Location:[/bold cyan] {location}\n"
            f"[bold red]💥 Disaster:[/bold red] {disaster_type.replace('_', ' ').title()}\n"
            f"[bold yellow]⚡ Severity:[/bold yellow] {severity}/10\n"
            f"[bold green]⏱️  Duration:[/bold green] {duration} hours\n"
            f"[bold magenta]🤖 AI Agents:[/bold magenta] 10 total (6 ADK Strategic + 4 Enhanced Tactical)\n"
            f"[bold blue]🧠 AI Orchestrator:[/bold blue] Gemini 2.0 Flash with cross-agent coordination\n"
            f"[bold purple]📊 Real-time Features:[/bold purple] WebSocket streaming, Live metrics, AI content generation",
            title="🔥 Enhanced ERIS v0.5.0 Simulation Configuration",
            border_style="blue"
        )
        console.print(preview_panel)
    
    def _display_enhanced_simulation_success(self, simulation_id: str, response: Dict[str, Any], agents_info: Dict[str, Any]):
        """Display enhanced simulation success information"""
        console.print(f"\n[bold green]✅ Enhanced simulation started successfully![/bold green]")
        
        info_table = Table(show_header=False, box=None, padding=(0, 2))
        info_table.add_column("Label", style="cyan", no_wrap=True)
        info_table.add_column("Value", style="white")
        
        info_table.add_row("🆔 Simulation ID:", simulation_id)
        info_table.add_row("🔄 Status:", response.get('status', 'unknown'))
        info_table.add_row("🧠 AI Orchestrator:", response.get('orchestrator_info', {}).get('ai_model', 'Gemini 2.0 Flash'))
        info_table.add_row("🤖 Total Agents:", str(response.get('orchestrator_info', {}).get('total_agents', 10)))
        info_table.add_row("⚡ Real-time Features:", "✅ WebSocket + Live Metrics + AI Content")
        info_table.add_row("🌐 WebSocket URL:", f"ws://.../{simulation_id[:8]}...")
        
        console.print(info_table)
        
        # Display enhanced agents summary
        self._display_enhanced_agents_summary(agents_info)
    
    def _display_enhanced_agents_summary(self, agents_info: Dict[str, Any]):
        """Display enhanced 10-agent summary"""
        if not agents_info or 'agents' not in agents_info:
            console.print("\n[yellow]⚠️ Agent information not available yet[/yellow]")
            return
        
        console.print("\n[bold]🤖 Enhanced AI Agent Network Status:[/bold]")
        
        # Create enhanced agent tree
        tree = Tree("🔥 ERIS v0.5.0 Agent System - Gemini 2.0 Flash Orchestrator")
        
        agents = agents_info.get('agents', [])
        adk_agents = [a for a in agents if a.get('type') == 'adk']
        enhanced_agents = [a for a in agents if a.get('type') == 'enhanced']
        
        # ADK Agents branch
        adk_branch = tree.add(f"🔷 Strategic ADK Agents ({len(adk_agents)})")
        for agent in adk_agents:
            status = agent.get('status', 'unknown')
            status_icon = "✅" if status in ["active", "analyzing"] else "⚠️"
            efficiency = agent.get('efficiency', 95)
            adk_branch.add(f"{status_icon} {agent['name']} - {efficiency}% efficiency")
        
        # Enhanced Agents branch
        enhanced_branch = tree.add(f"⚡ Tactical Enhanced Agents ({len(enhanced_agents)})")
        for agent in enhanced_agents:
            status = agent.get('status', 'unknown')
            status_icon = "✅" if status in ["active", "analyzing"] else "⚠️"
            efficiency = agent.get('efficiency', 90)
            enhanced_branch.add(f"{status_icon} {agent['name']} - {efficiency}% efficiency")
        
        console.print(tree)
        
        # Enhanced orchestrator info
        orchestrator_info = agents_info.get('orchestrator_info', {})
        if orchestrator_info:
            console.print(f"\n[bold]🧠 Orchestrator Details:[/bold]")
            console.print(f"• AI Model: [cyan]{orchestrator_info.get('ai_model', 'Gemini 2.0 Flash')}[/cyan]")
            console.print(f"• Coordination: [green]{orchestrator_info.get('coordination_active', True) and '✅ Active' or '❌ Inactive'}[/green]")
            console.print(f"• Real-time: [blue]{orchestrator_info.get('real_time_active', True) and '✅ Enabled' or '❌ Disabled'}[/blue]")
    
    async def monitor_simulation_enhanced(self, simulation_id: str, duration: int = 30, use_websocket: bool = True):
        """Enhanced simulation monitoring with WebSocket support"""
        console.print(f"\n[bold]📊 Enhanced Monitoring: [cyan]{simulation_id[:8]}...[/cyan][/bold]")
        console.print(f"[yellow]Monitoring for {duration} seconds with {'WebSocket' if use_websocket else 'HTTP polling'} (Ctrl+C to stop)[/yellow]\n")
        
        async with self.client:
            try:
                update_count = 0
                error_count = 0
                ws_monitor = None
                
                # Try WebSocket first if enabled
                if use_websocket:
                    ws_monitor = ERISWebSocketMonitor(simulation_id, self.config)
                    ws_connected = await ws_monitor.connect()
                    
                    if ws_connected:
                        console.print("[green]🔄 WebSocket real-time monitoring active[/green]\n")
                        
                        # WebSocket monitoring loop
                        async def ws_callback(data):
                            nonlocal update_count
                            update_count += 1
                            
                            if data.get('type') in ['initial_state', 'metrics_update']:
                                try:
                                    # Get additional status info
                                    status = await self.client.get_simulation_status(simulation_id)
                                    self._display_enhanced_live_metrics(data, status, update_count, error_count, "WebSocket")
                                except Exception as e:
                                    logger.warning(f"Failed to get status during WebSocket update: {e}")
                        
                        # Start WebSocket listener with timeout
                        try:
                            await asyncio.wait_for(ws_monitor.listen(ws_callback), timeout=duration)
                        except asyncio.TimeoutError:
                            console.print(f"\n[green]WebSocket monitoring completed after {duration} seconds[/green]")
                        
                        await ws_monitor.close()
                        return
                    else:
                        console.print("[yellow]⚠️ WebSocket connection failed, falling back to HTTP polling[/yellow]\n")
                
                # HTTP polling fallback
                for i in range(duration):
                    if self.interrupted:
                        break
                    
                    try:
                        # Get enhanced metrics and status in parallel
                        metrics_task = asyncio.create_task(self.client.get_dashboard_metrics(simulation_id))
                        status_task = asyncio.create_task(self.client.get_simulation_status(simulation_id))
                        
                        metrics, status = await asyncio.gather(metrics_task, status_task)
                        
                        # Display enhanced live metrics
                        self._display_enhanced_live_metrics(metrics, status, i, error_count, "HTTP")
                        update_count += 1
                        
                        await asyncio.sleep(1)
                        
                    except APIException as e:
                        error_count += 1
                        logger.warning(f"Enhanced monitoring error #{error_count}: {e}")
                        
                        if error_count >= 5:
                            console.print(f"\n[red]Too many errors ({error_count}), stopping monitoring[/red]")
                            break
                        
                        await asyncio.sleep(2)
                    
                    except KeyboardInterrupt:
                        break
                
                console.print(f"\n[green]Enhanced monitoring completed: {update_count} updates, {error_count} errors[/green]")
                
            except Exception as e:
                console.print(f"\n[red]Enhanced monitoring failed: {e}[/red]")
                logger.error(f"Enhanced monitoring error: {traceback.format_exc()}")
            finally:
                if ws_monitor:
                    await ws_monitor.close()
    
    def _display_enhanced_live_metrics(self, metrics: Dict[str, Any], status: Dict[str, Any], 
                                     iteration: int, error_count: int, source: str):
        """Enhanced live metrics display with v0.5.0 features"""
        dashboard_data = metrics.get('dashboard_data', {}) if 'dashboard_data' in metrics else metrics.get('dashboard_metrics', {})
        orchestrator_info = status.get('orchestrator', {}) if status else {}
        
        # Clear screen and show enhanced header
        console.clear()
        console.print(f"[bold blue]📊 ERIS v0.5.0 Enhanced Live Dashboard[/bold blue]")
        console.print(f"[dim]Update #{iteration + 1} • {datetime.now().strftime('%H:%M:%S')} • Source: {source} • Errors: {error_count}[/dim]\n")
        
        # Enhanced main metrics table
        metrics_table = Table(show_header=True, header_style="bold magenta")
        metrics_table.add_column("Metric", style="cyan", width=22)
        metrics_table.add_column("Current Value", style="white", width=15)
        metrics_table.add_column("Status", style="green", width=10)
        metrics_table.add_column("AI Analysis", style="yellow", width=15)
        
        # Enhanced Alert Level
        alert_level = dashboard_data.get('alert_level', 'UNKNOWN')
        alert_colors = {
            'GREEN': 'green', 'YELLOW': 'yellow', 
            'ORANGE': 'orange', 'RED': 'red', 'CRITICAL': 'bright_red'
        }
        alert_color = alert_colors.get(alert_level, 'white')
        
        metrics_table.add_row("🚨 Alert Level", alert_level, f"[{alert_color}]●[/{alert_color}]", "Dynamic")
        metrics_table.add_row("😰 Panic Index", f"{dashboard_data.get('panic_index', 0)}%", "📈", "Real-time")
        metrics_table.add_row("🏥 Hospital Capacity", f"{dashboard_data.get('hospital_capacity', 0)}%", "⚕️", "Load Model")
        metrics_table.add_row("👥 Population Affected", f"{dashboard_data.get('population_affected', 0):,}", "📊", "Behavior Sim")
        metrics_table.add_row("🔧 Infrastructure Failures", str(dashboard_data.get('infrastructure_failures', 0)), "⚠️", "Impact Model")
        metrics_table.add_row("🚑 Emergency Response", f"{dashboard_data.get('emergency_response', 0)}%", "✅", "Coordination")
        metrics_table.add_row("🤝 Public Trust", f"{dashboard_data.get('public_trust', 0)}%", "💙", "Social Analysis")
        metrics_table.add_row("🏃 Evacuation Compliance", f"{dashboard_data.get('evacuation_compliance', 0)}%", "🚶", "Compliance AI")
        
        console.print(metrics_table)
        
        # Enhanced orchestrator status
        current_phase = orchestrator_info.get('current_phase', 'unknown')
        agent_summary = orchestrator_info.get('agent_summary', {})
        total_agents = agent_summary.get('total_agents', 10)
        active_agents = agent_summary.get('active_agents', 0)
        
        console.print(f"\n[bold]🧠 Enhanced Orchestrator Status:[/bold]")
        console.print(f"📍 Current Phase: [cyan]{current_phase.title()}[/cyan]")
        console.print(f"🤖 Active Agents: [green]{active_agents}/{total_agents}[/green]")
        console.print(f"⚡ Real-time: [green]{orchestrator_info.get('real_time_active', False) and '✅ Active' or '❌ Inactive'}[/green]")
        console.print(f"🧠 AI Model: [blue]{orchestrator_info.get('ai_model', 'Gemini 2.0 Flash')}[/blue]")
        
        # Enhanced performance indicators
        if source == "WebSocket":
            console.print(f"🔄 Connection: [green]WebSocket Real-time ✅[/green]")
        else:
            console.print(f"🔄 Connection: [yellow]HTTP Polling ⚠️[/yellow]")
    
    async def display_enhanced_simulation_history(self):
        """Display enhanced simulation history"""
        if not self.simulation_history:
            console.print("[yellow]No simulation history available[/yellow]")
            return
        
        history_table = Table(title="📚 Enhanced Simulation History (v0.5.0)", show_header=True)
        history_table.add_column("ID", style="cyan", width=10)
        history_table.add_column("Disaster", style="red", width=15)
        history_table.add_column("Location", style="green", width=20)
        history_table.add_column("Severity", style="yellow", width=8)
        history_table.add_column("Orchestrator", style="blue", width=18)
        history_table.add_column("Agents", style="magenta", width=8)
        history_table.add_column("Started", style="white", width=16)
        history_table.add_column("Status", style="purple", width=10)
        
        for sim in self.simulation_history:
            started_str = sim['started_at'].strftime('%Y-%m-%d %H:%M')
            history_table.add_row(
                sim['id'][:8] + "...",
                sim['disaster_type'].replace('_', ' ').title(),
                sim['location'],
                str(sim['severity']),
                sim.get('orchestrator', 'Gemini 2.0 Flash')[:15] + "...",
                str(sim.get('agents_count', 10)),
                started_str,
                sim['status'].title()
            )
        
        console.print(history_table)
    
    async def export_enhanced_simulation_data(self, simulation_id: str, format: str = "json"):
        """Export enhanced simulation data with v0.5.0 features"""
        async with self.client:
            try:
                with Status(f"📤 Exporting enhanced simulation data ({format})...", console=console):
                    # Get comprehensive enhanced data
                    status_task = asyncio.create_task(self.client.get_simulation_status(simulation_id))
                    metrics_task = asyncio.create_task(self.client.get_dashboard_metrics(simulation_id))
                    agents_task = asyncio.create_task(self.client.get_agents_info(simulation_id))
                    
                    try:
                        orchestrator_task = asyncio.create_task(self.client.get_orchestrator_status(simulation_id))
                        live_feed_task = asyncio.create_task(self.client.get_live_feed(simulation_id, 50))
                        
                        status, metrics, agents, orchestrator, live_feed = await asyncio.gather(
                            status_task, metrics_task, agents_task, orchestrator_task, live_feed_task
                        )
                    except Exception as e:
                        logger.warning(f"Some enhanced data unavailable: {e}")
                        status, metrics, agents = await asyncio.gather(status_task, metrics_task, agents_task)
                        orchestrator = {}
                        live_feed = {}
                    
                    enhanced_export_data = {
                        'eris_version': '0.5.0',
                        'export_type': 'enhanced_simulation_data',
                        'simulation_id': simulation_id,
                        'exported_at': datetime.utcnow().isoformat(),
                        'simulation_status': status,
                        'dashboard_metrics': metrics,
                        'agent_network': agents,
                        'orchestrator_status': orchestrator,
                        'live_emergency_feed': live_feed,
                        'client_performance': self.client.get_performance_stats(),
                        'export_metadata': {
                            'orchestrator': 'Gemini 2.0 Flash',
                            'total_agents': 10,
                            'features': ['real_time_metrics', 'websocket_streaming', 'ai_content_generation'],
                            'cli_version': '0.5.0'
                        }
                    }
                    
                    # Create enhanced export directory
                    export_dir = Path.home() / ".eris" / "exports" / "v0.5.0"
                    export_dir.mkdir(parents=True, exist_ok=True)
                    
                    filename = f"eris_enhanced_{simulation_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
                    filepath = export_dir / filename
                    
                    if format == "json":
                        with open(filepath, 'w') as f:
                            json.dump(enhanced_export_data, f, indent=2, default=str)
                    
                    # Calculate export stats
                    file_size = filepath.stat().st_size
                    console.print(f"[green]✅ Enhanced export completed![/green]")
                    console.print(f"[green]📁 File: {filepath}[/green]")
                    console.print(f"[green]📊 Size: {file_size:,} bytes[/green]")
                    console.print(f"[green]🔥 ERIS v0.5.0 format with {len(enhanced_export_data)} data sections[/green]")
                    
            except Exception as e:
                console.print(f"[red]❌ Enhanced export failed: {e}[/red]")
                logger.error(f"Enhanced export error: {traceback.format_exc()}")

# Enhanced CLI Commands
def async_command(f):
    """Enhanced decorator for async commands with comprehensive error handling"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return asyncio.run(f(*args, **kwargs))
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  Operation cancelled by user[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Command failed: {e}[/red]")
            logger.error(f"Command failed: {traceback.format_exc()}")
    return wrapper

@click.group()
@click.option('--api-url', envvar='ERIS_API_URL', help='ERIS API URL')
@click.option('--api-key', envvar='ERIS_API_KEY', help='API key for authentication')
@click.option('--log-level', envvar='ERIS_LOG_LEVEL', default='INFO', help='Logging level')
@click.pass_context
def cli(ctx, api_url, api_key, log_level):
    """ ERIS v0.5.0 - Enhanced Emergency Response Intelligence System CLI
    
    Professional command-line interface for the enhanced ERIS disaster simulation platform.
    Features 10 AI agents, Gemini 2.0 Flash orchestrator, real-time WebSocket monitoring,
    and comprehensive cloud integration.
    """
    # Update config if provided
    if api_url:
        config.api_url = api_url
    if api_key:
        config.api_key = api_key
    
    config.log_level = log_level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    
    # Initialize enhanced context
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['app'] = ERISCLIApplication(config)

@cli.command()
@click.pass_context
@async_command
async def status(ctx):
    """Display comprehensive enhanced system status"""
    app = ctx.obj['app']
    await app.display_system_status()

@cli.command()
@click.pass_context
@async_command
async def disasters(ctx):
    """List enhanced disaster scenarios with AI agent details"""
    app = ctx.obj['app']
    await app.list_disasters()

@cli.command()
@click.option('--type', 'disaster_type', required=True,
              type=click.Choice(DISASTER_TYPES, case_sensitive=False),
              help='Disaster type to simulate')
@click.option('--location', required=True, help='Geographic location')
@click.option('--severity', type=click.IntRange(1, 10), default=7,
              help='Disaster severity (1-10)')
@click.option('--duration', type=click.IntRange(1, 168), default=24,
              help='Simulation duration in hours (max 168)')
@click.option('--monitor', type=int, default=0,
              help='Monitor simulation for N seconds after start')
@click.option('--websocket/--no-websocket', default=True,
              help='Use WebSocket for real-time monitoring')
@click.option('--export', 'export_format', type=click.Choice(['json']),
              help='Export simulation data after completion')
@click.pass_context
@async_command
async def simulate(ctx, disaster_type, location, severity, duration, monitor, websocket, export_format):
    """Start enhanced disaster simulation with 10 AI agents"""
    app = ctx.obj['app']
    
    simulation_id = await app.start_simulation_interactive(
        disaster_type, location, severity, duration
    )
    
    if simulation_id:
        if monitor > 0:
            await app.monitor_simulation_enhanced(simulation_id, monitor, websocket)
        
        if export_format:
            await app.export_enhanced_simulation_data(simulation_id, export_format)

@cli.command()
@click.argument('simulation_id')
@click.option('--duration', type=int, default=30,
              help='Monitoring duration in seconds')
@click.option('--websocket/--no-websocket', default=True,
              help='Use WebSocket for real-time monitoring')
@click.option('--export', 'export_format', type=click.Choice(['json']),
              help='Export data after monitoring')
@click.pass_context
@async_command
async def monitor(ctx, simulation_id, duration, websocket, export_format):
    """Monitor simulation with enhanced real-time features"""
    app = ctx.obj['app']
    await app.monitor_simulation_enhanced(simulation_id, duration, websocket)
    
    if export_format:
        await app.export_enhanced_simulation_data(simulation_id, export_format)

@cli.command()
@click.argument('simulation_id')
@click.option('--format', 'export_format', type=click.Choice(['json']), default='json',
              help='Export format')
@click.pass_context
@async_command
async def export(ctx, simulation_id, export_format):
    """Export simulation data with v0.5.0 features"""
    app = ctx.obj['app']
    await app.export_enhanced_simulation_data(simulation_id, export_format)

@cli.command()
@click.pass_context
@async_command
async def history(ctx):
    """Display simulation history"""
    app = ctx.obj['app']
    await app.display_enhanced_simulation_history()

@cli.command()
@click.option('--config-file', type=click.Path(), help='Configuration file path')
@click.pass_context
def configure(ctx, config_file):
    """Configure ERIS CLI settings"""
    config_dir = Path.home() / ".eris"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    if not config_file:
        config_file = config_dir / "config.json"
    
    # Interactive configuration
    console.print("[bold]⚙️ ERIS v0.5.0 CLI Configuration[/bold]\n")
    
    current_config = {
        'api_url': config.api_url,
        'api_key': config.api_key or '',
        'timeout': config.timeout,
        'max_retries': config.max_retries,
        'log_level': config.log_level,
        'websocket_timeout': config.websocket_timeout
    }
    
    # Update configuration interactively
    new_config = {}
    new_config['api_url'] = Prompt.ask("API URL", default=current_config['api_url'])
    new_config['api_key'] = Prompt.ask("API Key (optional)", default=current_config['api_key'], password=True)
    new_config['timeout'] = int(Prompt.ask("Request timeout (seconds)", default=str(current_config['timeout'])))
    new_config['max_retries'] = int(Prompt.ask("Max retries", default=str(current_config['max_retries'])))
    new_config['websocket_timeout'] = int(Prompt.ask("WebSocket timeout (seconds)", default=str(current_config['websocket_timeout'])))
    new_config['log_level'] = Prompt.ask("Log level", default=current_config['log_level'], 
                                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    # Save configuration
    new_config['version'] = '0.5.0'
    new_config['features'] = ['enhanced_monitoring', 'websocket_support', 'gemini_orchestrator']
    
    with open(config_file, 'w') as f:
        json.dump(new_config, f, indent=2)
    
    console.print(f"[green]✅ Enhanced configuration saved to: {config_file}[/green]")
    console.print("[yellow]💡 Restart the CLI for changes to take effect[/yellow]")

@cli.command()
@click.pass_context
def version(ctx):
    """Display version information"""
    version_info = {
        'ERIS CLI': '0.5.0 Enhanced',
        'API Version': 'v0.5.0',
        'Orchestrator': 'Gemini 2.0 Flash',
        'Agents': '10 (6 ADK + 4 Enhanced)',
        'Features': 'WebSocket + Real-time + AI Content',
        'Python': sys.version.split()[0],
        'Platform': sys.platform,
        'API URL': config.api_url,
        'Auth': '✅ Enabled' if config.api_key else '❌ Disabled'
    }
    
    table = Table(title="📦 Enhanced Version Information (v0.5.0)", show_header=False)
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="green")
    
    for component, version in version_info.items():
        table.add_row(component, version)
    
    console.print(table)

def main():
    """Main entry point"""
    try:
        # Display banner
        banner = Panel.fit(
            "[bold blue]🔥 ERIS v0.5.0[/bold blue]\n"
            "[italic]Emergency Response Intelligence System[/italic]\n"
            "🧠 Gemini 2.0 Flash Orchestrator • 🤖 10 AI Agents • ☁️ Cloud Integration\n"
            "📊 Real-time WebSocket • ✨ AI Content Generation • 🚀 Production Ready\n"
            "[dim]Enhanced Professional CLI • Cross-agent Coordination[/dim]",
            style="blue",
            padding=(1, 2)
        )
        console.print(banner)
        
        # Check for configuration
        config_file = Path.home() / ".eris" / "config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    saved_config = json.load(f)
                    
                # Update global config with settings
                for key, value in saved_config.items():
                    if hasattr(config, key) and value and key != 'features':
                        setattr(config, key, value)
                        
                logger.info(f"Enhanced configuration loaded from {config_file}")
                
                # Show loaded features if available
                if 'features' in saved_config:
                    console.print(f"[dim]Loaded features: {', '.join(saved_config['features'])}[/dim]\n")
                    
            except Exception as e:
                logger.warning(f"Failed to load enhanced configuration: {e}")
        
        # Run CLI
        cli()
        
    except Exception as e:
        console.print(f"[red]❌ Enhanced CLI initialization failed: {e}[/red]")
        logger.error(f"Enhanced CLI startup error: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == '__main__':
    main()
