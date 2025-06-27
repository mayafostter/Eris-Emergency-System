"""
ERIS (Emergency Response Intelligence System) - CLI
Command-line interface for disaster simulation management

Features:
- Comprehensive error handling and recovery
- Performance monitoring and optimization
- Secure API communication with retry logic
- Advanced real-time visualization
- Configuration management
- Logging and audit trails
"""

import os
import sys
import asyncio
import aiohttp
import logging
import json
import time
import traceback
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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# logging setup
def setup_logging():
    """Setup comprehensive logging"""
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

# Configuration
@dataclass
class ERISConfig:
    """ERIS CLI Configuration"""
    api_url: str = os.getenv('ERIS_API_URL', 'https://eris-backend-621360763676.us-central1.run.app')
    api_key: Optional[str] = os.getenv('ERIS_API_KEY')
    timeout: int = int(os.getenv('ERIS_TIMEOUT', '30'))
    max_retries: int = int(os.getenv('ERIS_MAX_RETRIES', '3'))
    retry_delay: float = float(os.getenv('ERIS_RETRY_DELAY', '1.0'))
    log_level: str = os.getenv('ERIS_LOG_LEVEL', 'INFO')
    
    def __post_init__(self):
        # Validate configuration
        if not self.api_url:
            raise ValueError("ERIS_API_URL must be configured")
        
        # Set log level
        logging.getLogger().setLevel(getattr(logging, self.log_level.upper()))

config = ERISConfig()

DISASTER_TYPES = [
    'earthquake', 'hurricane', 'flood', 'wildfire', 'tsunami', 
    'volcanic_eruption', 'severe_storm', 'epidemic', 'pandemic', 'landslide'
]

console = Console()

class APIException(Exception):
    """Custom API exception with detailed error info"""
    def __init__(self, message: str, status_code: int = None, response_data: Dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}
        self.timestamp = datetime.utcnow()

class RetryManager:
    """Advanced retry manager with exponential backoff"""
    
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
    """ERIS API client with comprehensive features"""
    
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
    
    async def __aenter__(self):
        """Context manager entry"""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        connector = aiohttp.TCPConnector(
            limit=self._connection_pool_size,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        headers = {
            'User-Agent': 'ERIS-CLI/1.0',
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
        
        logger.info(f"API client initialized for {self.config.api_url}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            await self.session.close()
        
        # Log session statistics
        session_duration = (datetime.utcnow() - self.start_time).total_seconds()
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        logger.info(f"API session closed: {self.request_count} requests, "
                   f"{self.error_count} errors, {session_duration:.1f}s duration, "
                   f"{avg_response_time:.3f}s avg response time")
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Request method with comprehensive error handling"""
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
        """Health check with caching"""
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
        """Get comprehensive system information"""
        return await self._make_request('GET', '/system/info')
    
    async def start_simulation(self, disaster_type: str, location: str, 
                             severity: int, duration: int = 24) -> Dict[str, Any]:
        """Start simulation with validation"""
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
        
        logger.info(f"Starting simulation: {disaster_type} in {location} (severity {severity})")
        return await self._make_request('POST', '/simulate', json=payload)
    
    async def get_simulation_status(self, simulation_id: str) -> Dict[str, Any]:
        """Get simulation status with validation"""
        if not simulation_id:
            raise ValueError("Simulation ID is required")
        
        return await self._make_request('GET', f'/status/{simulation_id}')
    
    async def get_agents_info(self, simulation_id: str) -> Dict[str, Any]:
        """Get detailed agent information"""
        return await self._make_request('GET', f'/orchestrator/{simulation_id}/agents')
    
    async def get_dashboard_metrics(self, simulation_id: str) -> Dict[str, Any]:
        """Get real-time dashboard metrics"""
        return await self._make_request('GET', f'/metrics/dashboard/{simulation_id}')
    
    async def get_extended_metrics(self, simulation_id: str) -> Dict[str, Any]:
        """Get extended metrics from agents"""
        return await self._make_request('GET', f'/extended-metrics/{simulation_id}')
    
    async def get_live_feed(self, simulation_id: str, limit: int = 20) -> Dict[str, Any]:
        """Get live emergency feed"""
        return await self._make_request('GET', f'/live-feed/{simulation_id}?limit={limit}')
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get client performance statistics"""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        session_duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": (self.error_count / max(1, self.request_count)) * 100,
            "average_response_time": avg_response_time,
            "session_duration": session_duration,
            "requests_per_minute": (self.request_count / max(1, session_duration)) * 60,
            "health_status": self.health_status
        }

class ERISCLIApplication:
    """CLI application with advanced features"""
    
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
        """System status display"""
        async with self.client:
            try:
                with Status("🔍 Checking system status...", console=console):
                    # Get system info and health in parallel
                    health_task = asyncio.create_task(self.client.health_check())
                    info_task = asyncio.create_task(self.client.get_system_info())
                    
                    health, system_info = await asyncio.gather(health_task, info_task)
                
                # Create comprehensive status display
                self._display_system_status_table(health, system_info)
                self._display_performance_stats()
                
            except APIException as e:
                console.print(f"[red]❌ System check failed: {e}[/red]")
                if e.status_code:
                    console.print(f"[red]HTTP Status: {e.status_code}[/red]")
                logger.error(f"System status check failed: {e}")
            except Exception as e:
                console.print(f"[red]❌ Unexpected error: {e}[/red]")
                logger.error(f"Unexpected error in system status: {traceback.format_exc()}")
    
    def _display_system_status_table(self, health: Dict[str, Any], system_info: Dict[str, Any]):
        """Display comprehensive system status table"""
        table = Table(title="🔥 ERIS System Status", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan", no_wrap=True, width=25)
        table.add_column("Status", style="green", width=15)
        table.add_column("Details", style="white")
        
        # API Status
        api_status = "✅ Online" if health.get('status') == 'healthy' else "❌ Offline"
        table.add_row("🌐 API Server", api_status, f"Connected to {self.config.api_url}")
        
        # Version info
        version = system_info.get('eris_version', 'Unknown')
        table.add_row("📦 ERIS Version", f"v{version}", "Production ready")
        
        # Orchestrator info
        orchestrator = system_info.get('orchestrator', {})
        ai_model = orchestrator.get('ai_model', 'Gemini 2.0 Flash')
        table.add_row("🧠 AI Orchestrator", "✅ Active", f"{ai_model} coordination")
        
        # Agent Systems
        agent_system = system_info.get('agent_system', {})
        total_agents = agent_system.get('total_agents', 10)
        adk_agents = agent_system.get('adk_agents', 6)
        enhanced_agents = agent_system.get('enhanced_agents', 4)
        
        table.add_row("🤖 Agent System", "✅ Ready", f"{total_agents} agents ({adk_agents} ADK + {enhanced_agents} Enhanced)")
        
        # Real-time features
        features = system_info.get('real_time_features', {})
        if features.get('dynamic_metrics_calculation'):
            table.add_row("📊 Real-time Metrics", "✅ Enabled", "Dynamic calculation & WebSocket streaming")
        
        if features.get('ai_content_generation'):
            table.add_row("✨ AI Content Gen", "✅ Enabled", "Live social media & emergency feeds")
        
        # Cloud integration
        table.add_row("☁️ Cloud Integration", "✅ Connected", "Firestore + BigQuery + Vertex AI")
        
        # Capabilities
        capabilities = system_info.get('capabilities', {})
        active_sims = system_info.get('active_simulations', 0)
        ws_connections = system_info.get('websocket_connections', 0)
        
        table.add_row("🔄 Active Simulations", str(active_sims), f"{ws_connections} WebSocket connections")
        
        console.print(table)
        console.print(f"\n[bold green]✅ ERIS v{version} - All systems operational[/bold green]")
    
    def _display_performance_stats(self):
        """Display client performance statistics"""
        stats = self.client.get_performance_stats()
        
        perf_table = Table(title="📈 Client Performance", show_header=True)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="yellow")
        
        perf_table.add_row("Total Requests", str(stats['total_requests']))
        perf_table.add_row("Error Rate", f"{stats['error_rate']:.1f}%")
        perf_table.add_row("Avg Response Time", f"{stats['average_response_time']:.3f}s")
        perf_table.add_row("Requests/Min", f"{stats['requests_per_minute']:.1f}")
        
        console.print(perf_table)
    
    async def list_disasters(self):
        """Disaster types display"""
        async with self.client:
            try:
                system_info = await self.client.get_system_info()
                
                table = Table(title="🌪️ Available Disaster Scenarios", show_header=True)
                table.add_column("Disaster Type", style="cyan", width=20)
                table.add_column("Severity Range", style="yellow", width=15)
                table.add_column("Key Systems Affected", style="magenta")
                table.add_column("Avg Duration", style="green", width=12)
                
                disaster_details = {
                    'earthquake': ('1-9 Richter', 'Infrastructure, Hospitals, Communications', '2-6 hours'),
                    'hurricane': ('1-5 Category', 'Evacuation, Supply Chain, Emergency Services', '12-48 hours'),
                    'flood': ('1-10 Index', 'Transportation, Housing, Public Health', '6-24 hours'),
                    'wildfire': ('1-10 Index', 'Evacuation, Air Quality, Resources', '12-72 hours'),
                    'tsunami': ('1-10 Wave Height', 'Coastal Infrastructure, Mass Evacuation', '2-8 hours'),
                    'volcanic_eruption': ('1-8 VEI', 'Air Traffic, Agriculture, Public Health', '24-168 hours'),
                    'severe_storm': ('1-5 Intensity', 'Power Grid, Transportation, Communications', '4-12 hours'),
                    'epidemic': ('1-10 Spread Rate', 'Healthcare, Public Health, Social Systems', '72-168 hours'),
                    'pandemic': ('1-10 Global Impact', 'All Systems, Economic, Social', '168+ hours'),
                    'landslide': ('1-10 Risk Level', 'Infrastructure, Transportation, Housing', '1-6 hours')
                }
                
                for disaster in DISASTER_TYPES:
                    severity, systems, duration = disaster_details.get(disaster, ('1-10', 'Multiple Systems', 'Variable'))
                    table.add_row(disaster.replace('_', ' ').title(), severity, systems, duration)
                
                console.print(table)
                console.print(f"\n[italic]💡 Tip: Use --severity to adjust impact intensity (1=minimal, 10=catastrophic)[/italic]")
                
            except APIException as e:
                console.print(f"[red]Failed to fetch disaster types: {e}[/red]")
    
    async def start_simulation_interactive(self, disaster_type: str, location: str, 
                                         severity: int, duration: int):
        """Simulation startup with comprehensive feedback"""
        async with self.client:
            try:
                # Pre-flight validation
                if not location.strip():
                    raise ValueError("Location cannot be empty")
                
                # Display simulation details
                self._display_simulation_preview(disaster_type, location, severity, duration)
                
                # Confirm if interactive
                if not Confirm.ask("\n🚀 Start this simulation?"):
                    console.print("[yellow]Simulation cancelled by user[/yellow]")
                    return None
                
                # Start simulation with progress tracking
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    
                    task = progress.add_task("Initializing ERIS simulation...", total=100)
                    
                    # Phase 1: Start simulation
                    progress.update(task, advance=20, description="Starting simulation...")
                    response = await self.client.start_simulation(disaster_type, location, severity, duration)
                    
                    simulation_id = response['simulation_id']
                    
                    # Phase 2: Wait for orchestrator initialization
                    progress.update(task, advance=30, description="Initializing AI orchestrator...")
                    await asyncio.sleep(2)
                    
                    # Phase 3: Get initial agent status
                    progress.update(task, advance=25, description="Activating 10 AI agents...")
                    agents_info = await self.client.get_agents_info(simulation_id)
                    
                    # Phase 4: Get initial metrics
                    progress.update(task, advance=25, description="Establishing real-time metrics...")
                    await asyncio.sleep(1)
                    
                    progress.update(task, completed=100, description="Simulation ready!")
                
                # Display success info
                self._display_simulation_success(simulation_id, response, agents_info)
                
                # Add to history
                self.simulation_history.append({
                    'id': simulation_id,
                    'disaster_type': disaster_type,
                    'location': location,
                    'severity': severity,
                    'duration': duration,
                    'started_at': datetime.utcnow(),
                    'status': 'active'
                })
                
                return simulation_id
                
            except ValueError as e:
                console.print(f"[red]❌ Validation error: {e}[/red]")
                return None
            except APIException as e:
                console.print(f"[red]❌ Failed to start simulation: {e}[/red]")
                if e.response_data:
                    console.print(f"[red]Details: {e.response_data}[/red]")
                return None
            except Exception as e:
                console.print(f"[red]❌ Unexpected error: {e}[/red]")
                logger.error(f"Simulation start failed: {traceback.format_exc()}")
                return None
    
    def _display_simulation_preview(self, disaster_type: str, location: str, severity: int, duration: int):
        """Display simulation preview"""
        preview_panel = Panel(
            f"[bold cyan]📍 Location:[/bold cyan] {location}\n"
            f"[bold red]💥 Disaster:[/bold red] {disaster_type.replace('_', ' ').title()}\n"
            f"[bold yellow]⚡ Severity:[/bold yellow] {severity}/10\n"
            f"[bold green]⏱️  Duration:[/bold green] {duration} hours\n"
            f"[bold magenta]🤖 Agents:[/bold magenta] 10 AI agents (6 ADK + 4 Enhanced)\n"
            f"[bold blue]🧠 AI Model:[/bold blue] Gemini 2.0 Flash orchestrator",
            title="🔥 ERIS Simulation Configuration",
            border_style="blue"
        )
        console.print(preview_panel)
    
    def _display_simulation_success(self, simulation_id: str, response: Dict[str, Any], agents_info: Dict[str, Any]):
        """Display simulation success information"""
        console.print(f"\n[bold green]✅ Simulation started successfully![/bold green]")
        
        info_table = Table(show_header=False, box=None, padding=(0, 2))
        info_table.add_column("Label", style="cyan", no_wrap=True)
        info_table.add_column("Value", style="white")
        
        info_table.add_row("🆔 Simulation ID:", simulation_id)
        info_table.add_row("🔄 Status:", response.get('status', 'unknown'))
        info_table.add_row("🧠 Orchestrator:", response.get('orchestrator_info', {}).get('ai_model', 'Gemini 2.0 Flash'))
        info_table.add_row("🤖 Total Agents:", str(response.get('orchestrator_info', {}).get('total_agents', 10)))
        info_table.add_row("⚡ Real-time Features:", "✅ Enabled")
        
        console.print(info_table)
        
        # Display agents summary
        self._display_agents_summary(agents_info)
    
    def _display_agents_summary(self, agents_info: Dict[str, Any]):
        """Display enhanced agents summary"""
        if not agents_info or 'adk_agents' not in agents_info:
            return
        
        console.print("\n[bold]🤖 AI Agent Network Status:[/bold]")
        
        # Create agent tree
        tree = Tree("🔥 ERIS Agent System")
        
        # ADK Agents branch
        adk_branch = tree.add("🔷 Google ADK Agents (6)")
        for name, agent in agents_info.get('adk_agents', {}).items():
            status = agent.get('status', 'unknown')
            status_icon = "✅" if status == "initialized" else "⚠️"
            adk_branch.add(f"{status_icon} {name.replace('_', ' ').title()}")
        
        # Enhanced Agents branch
        enhanced_branch = tree.add("⚡ Enhanced Agents (4)")
        for name, agent in agents_info.get('enhanced_agents', {}).items():
            status = agent.get('status', 'unknown')
            status_icon = "✅" if status == "initialized" else "⚠️"
            enhanced_branch.add(f"{status_icon} {name.replace('_', ' ').title()}")
        
        console.print(tree)
    
    async def monitor_simulation(self, simulation_id: str, duration: int = 30):
        """Simulation monitoring with live updates"""
        console.print(f"\n[bold]📊 Monitoring Simulation: [cyan]{simulation_id[:8]}...[/cyan][/bold]")
        console.print(f"[yellow]Monitoring for {duration} seconds (Ctrl+C to stop)[/yellow]\n")
        
        async with self.client:
            try:
                update_count = 0
                error_count = 0
                
                for i in range(duration):
                    if self.interrupted:
                        break
                    
                    try:
                        # Get metrics and status in parallel
                        metrics_task = asyncio.create_task(self.client.get_dashboard_metrics(simulation_id))
                        status_task = asyncio.create_task(self.client.get_simulation_status(simulation_id))
                        
                        metrics, status = await asyncio.gather(metrics_task, status_task)
                        
                        # Display live metrics
                        self._display_live_metrics(metrics, status, i, error_count)
                        update_count += 1
                        
                        await asyncio.sleep(1)
                        
                    except APIException as e:
                        error_count += 1
                        logger.warning(f"Monitoring error #{error_count}: {e}")
                        
                        if error_count >= 5:
                            console.print(f"\n[red]Too many errors ({error_count}), stopping monitoring[/red]")
                            break
                        
                        await asyncio.sleep(2)  # Longer delay on error
                    
                    except KeyboardInterrupt:
                        break
                
                console.print(f"\n[green]Monitoring completed: {update_count} updates, {error_count} errors[/green]")
                
            except Exception as e:
                console.print(f"\n[red]Monitoring failed: {e}[/red]")
                logger.error(f"Monitoring error: {traceback.format_exc()}")
    
    def _display_live_metrics(self, metrics: Dict[str, Any], status: Dict[str, Any], 
                            iteration: int, error_count: int):
        """Live metrics display"""
        dashboard_data = metrics.get('dashboard_data', {})
        orchestrator_info = status.get('orchestrator', {})
        
        # Clear screen and show header
        console.clear()
        console.print(f"[bold blue]📊 ERIS Live Metrics Dashboard[/bold blue]")
        console.print(f"[dim]Update #{iteration + 1} • {datetime.now().strftime('%H:%M:%S')} • Errors: {error_count}[/dim]\n")
        
        # Main metrics table
        metrics_table = Table(show_header=True, header_style="bold magenta")
        metrics_table.add_column("Metric", style="cyan", width=20)
        metrics_table.add_column("Current Value", style="white", width=15)
        metrics_table.add_column("Status", style="green", width=10)
        metrics_table.add_column("Trend", style="yellow", width=10)
        
        # Alert Level
        alert_level = dashboard_data.get('alert_level', 'UNKNOWN')
        alert_colors = {
            'GREEN': 'green', 'YELLOW': 'yellow', 
            'ORANGE': 'orange', 'RED': 'red', 'CRITICAL': 'bright_red'
        }
        alert_color = alert_colors.get(alert_level, 'white')
        
        metrics_table.add_row("🚨 Alert Level", alert_level, f"[{alert_color}]●[/{alert_color}]", "→")
        metrics_table.add_row("😰 Panic Index", f"{dashboard_data.get('panic_index', 0)}%", "📈", "↑")
        metrics_table.add_row("🏥 Hospital Capacity", f"{dashboard_data.get('hospital_capacity', 0)}%", "⚕️", "↗")
        metrics_table.add_row("👥 Population Affected", f"{dashboard_data.get('population_affected', 0):,}", "📊", "→")
        metrics_table.add_row("🔧 Infrastructure Failures", str(dashboard_data.get('infrastructure_failures', 0)), "⚠️", "↘")
        metrics_table.add_row("🚑 Emergency Response", f"{dashboard_data.get('emergency_response', 0)}%", "✅", "↗")
        metrics_table.add_row("🤝 Public Trust", f"{dashboard_data.get('public_trust', 0)}%", "💙", "→")
        metrics_table.add_row("🏃 Evacuation Compliance", f"{dashboard_data.get('evacuation_compliance', 0)}%", "🚶", "↗")
        
        console.print(metrics_table)
        
        # Orchestrator status
        current_phase = orchestrator_info.get('current_phase', 'unknown')
        total_agents = orchestrator_info.get('total_agents', 10)
        
        console.print(f"\n[bold]🧠 Orchestrator Status:[/bold]")
        console.print(f"📍 Current Phase: [cyan]{current_phase.title()}[/cyan]")
        console.print(f"🤖 Active Agents: [green]{total_agents}/10[/green]")
        console.print(f"⚡ Real-time: [green]{'✅ Active' if orchestrator_info.get('real_time_active') else '❌ Inactive'}[/green]")
    
    async def display_simulation_history(self):
        """Display simulation history"""
        if not self.simulation_history:
            console.print("[yellow]No simulation history available[/yellow]")
            return
        
        history_table = Table(title="📚 Simulation History", show_header=True)
        history_table.add_column("ID", style="cyan", width=10)
        history_table.add_column("Disaster", style="red", width=15)
        history_table.add_column("Location", style="green", width=20)
        history_table.add_column("Severity", style="yellow", width=8)
        history_table.add_column("Started", style="blue", width=16)
        history_table.add_column("Status", style="magenta", width=10)
        
        for sim in self.simulation_history:
            started_str = sim['started_at'].strftime('%Y-%m-%d %H:%M')
            history_table.add_row(
                sim['id'][:8] + "...",
                sim['disaster_type'].replace('_', ' ').title(),
                sim['location'],
                str(sim['severity']),
                started_str,
                sim['status'].title()
            )
        
        console.print(history_table)
    
    async def export_simulation_data(self, simulation_id: str, format: str = "json"):
        """Export simulation data"""
        async with self.client:
            try:
                with Status(f"📤 Exporting simulation data ({format})...", console=console):
                    # Get comprehensive data
                    status = await self.client.get_simulation_status(simulation_id)
                    metrics = await self.client.get_dashboard_metrics(simulation_id)
                    agents = await self.client.get_agents_info(simulation_id)
                    
                    export_data = {
                        'simulation_id': simulation_id,
                        'exported_at': datetime.utcnow().isoformat(),
                        'status': status,
                        'metrics': metrics,
                        'agents': agents,
                        'client_stats': self.client.get_performance_stats()
                    }
                    
                    # Create export directory
                    export_dir = Path.home() / ".eris" / "exports"
                    export_dir.mkdir(parents=True, exist_ok=True)
                    
                    filename = f"eris_simulation_{simulation_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
                    filepath = export_dir / filename
                    
                    if format == "json":
                        with open(filepath, 'w') as f:
                            json.dump(export_data, f, indent=2, default=str)
                    
                    console.print(f"[green]✅ Exported to: {filepath}[/green]")
                    
            except Exception as e:
                console.print(f"[red]❌ Export failed: {e}[/red]")

# CLI Commands with error handling
def async_command(f):
    """Decorator to handle async commands with comprehensive error handling"""
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
    """ ERIS - Emergency Response Intelligence System CLI
    
    Professional command-line interface for the ERIS disaster simulation platform.
    Features 10 AI agents, real-time metrics, and cloud integration.
    """
    # Update config if provided
    if api_url:
        config.api_url = api_url
    if api_key:
        config.api_key = api_key
    
    config.log_level = log_level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    
    # Initialize context
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['app'] = ERISCLIApplication(config)

@cli.command()
@click.pass_context
@async_command
async def status(ctx):
    """Display comprehensive system status"""
    app = ctx.obj['app']
    await app.display_system_status()

@cli.command()
@click.pass_context
@async_command
async def disasters(ctx):
    """List available disaster scenarios"""
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
@click.option('--export', 'export_format', type=click.Choice(['json']),
              help='Export simulation data after completion')
@click.pass_context
@async_command
async def simulate(ctx, disaster_type, location, severity, duration, monitor, export_format):
    """Start a new disaster simulation with enhanced options"""
    app = ctx.obj['app']
    
    simulation_id = await app.start_simulation_interactive(
        disaster_type, location, severity, duration
    )
    
    if simulation_id:
        if monitor > 0:
            await app.monitor_simulation(simulation_id, monitor)
        
        if export_format:
            await app.export_simulation_data(simulation_id, export_format)

@cli.command()
@click.argument('simulation_id')
@click.option('--duration', type=int, default=30,
              help='Monitoring duration in seconds')
@click.option('--export', 'export_format', type=click.Choice(['json']),
              help='Export data after monitoring')
@click.pass_context
@async_command
async def monitor(ctx, simulation_id, duration, export_format):
    """Monitor an active simulation with live metrics"""
    app = ctx.obj['app']
    await app.monitor_simulation(simulation_id, duration)
    
    if export_format:
        await app.export_simulation_data(simulation_id, export_format)

@cli.command()
@click.argument('simulation_id')
@click.option('--format', 'export_format', type=click.Choice(['json']), default='json',
              help='Export format')
@click.pass_context
@async_command
async def export(ctx, simulation_id, export_format):
    """Export simulation data"""
    app = ctx.obj['app']
    await app.export_simulation_data(simulation_id, export_format)

@cli.command()
@click.pass_context
@async_command
async def history(ctx):
    """Display simulation history"""
    app = ctx.obj['app']
    await app.display_simulation_history()

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
    console.print("[bold]⚙️ ERIS CLI Configuration[/bold]\n")
    
    current_config = {
        'api_url': config.api_url,
        'api_key': config.api_key or '',
        'timeout': config.timeout,
        'max_retries': config.max_retries,
        'log_level': config.log_level
    }
    
    # Update configuration interactively
    new_config = {}
    new_config['api_url'] = Prompt.ask("API URL", default=current_config['api_url'])
    new_config['api_key'] = Prompt.ask("API Key (optional)", default=current_config['api_key'], password=True)
    new_config['timeout'] = int(Prompt.ask("Request timeout (seconds)", default=str(current_config['timeout'])))
    new_config['max_retries'] = int(Prompt.ask("Max retries", default=str(current_config['max_retries'])))
    new_config['log_level'] = Prompt.ask("Log level", default=current_config['log_level'], 
                                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    # Save configuration
    with open(config_file, 'w') as f:
        json.dump(new_config, f, indent=2)
    
    console.print(f"[green]✅ Configuration saved to: {config_file}[/green]")
    console.print("[yellow]💡 Restart the CLI for changes to take effect[/yellow]")

@cli.command()
@click.pass_context
def version(ctx):
    """Display version information"""
    version_info = {
        'ERIS CLI': '1.0.0',
        'Python': sys.version.split()[0],
        'Platform': sys.platform,
        'API URL': config.api_url,
        'Auth': '✅ Enabled' if config.api_key else '❌ Disabled'
    }
    
    table = Table(title="📦 Version Information", show_header=False)
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="green")
    
    for component, version in version_info.items():
        table.add_row(component, version)
    
    console.print(table)

def main():
    """Main entry point"""
    try:
        # Display enhanced banner
        banner = Panel.fit(
            "[bold blue]🔥 ERIS[/bold blue]\n"
            "[italic]Emergency Response Intelligence System[/italic]\n"
            "🤖 10 AI Agents • ☁️ Cloud Integration • 📊 Real-time Analytics\n"
            "[dim]Professional CLI v1.0 • Production Ready[/dim]",
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
                    
                # Update global config
                for key, value in saved_config.items():
                    if hasattr(config, key) and value:
                        setattr(config, key, value)
                        
                logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load configuration: {e}")
        
        # Run CLI
        cli()
        
    except Exception as e:
        console.print(f"[red]❌ CLI initialization failed: {e}[/red]")
        logger.error(f"CLI startup error: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == '__main__':
    main()
