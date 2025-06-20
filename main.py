"""
ERIS (Emergency Response Intelligence System)
CLI Client - Interfaces with ERIS API for command-line disaster simulation management

This CLI demonstrates the full ERIS system capabilities:
- 10 AI Agents (6 Google ADK + 4 Enhanced)
- Real-time metrics streaming
- Multi-phase disaster simulation
- Cloud integration (Firestore, BigQuery, Vertex AI)
"""

import os
import sys
import asyncio
import aiohttp
import logging
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = os.getenv('ERIS_API_URL', 'https://eris-backend-621360763676.asia-southeast1.run.app')
DISASTER_TYPES = ['earthquake', 'hurricane', 'flood', 'wildfire', 'tsunami', 'volcanic_eruption', 'severe_storm']

console = Console()

class ERISAPIClient:
    """Professional ERIS API client for command-line interface."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health and connectivity."""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise aiohttp.ClientError(f"Health check failed: {response.status}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to ERIS API at {self.base_url}: {e}")
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        async with self.session.get(f"{self.base_url}/system/info") as response:
            response.raise_for_status()
            return await response.json()
    
    async def start_simulation(self, disaster_type: str, location: str, 
                             severity: int, duration: int = 24) -> Dict[str, Any]:
        """Start a new disaster simulation."""
        payload = {
            "disaster_type": disaster_type,
            "location": location,
            "severity": severity,
            "duration": duration
        }
        
        async with self.session.post(f"{self.base_url}/simulate", 
                                   json=payload) as response:
            response.raise_for_status()
            return await response.json()
    
    async def get_simulation_status(self, simulation_id: str) -> Dict[str, Any]:
        """Get current simulation status."""
        async with self.session.get(f"{self.base_url}/status/{simulation_id}") as response:
            response.raise_for_status()
            return await response.json()
    
    async def get_agents_info(self, simulation_id: str) -> Dict[str, Any]:
        """Get detailed information about all agents."""
        async with self.session.get(f"{self.base_url}/orchestrator/{simulation_id}/agents") as response:
            response.raise_for_status()
            return await response.json()
    
    async def get_dashboard_metrics(self, simulation_id: str) -> Dict[str, Any]:
        """Get real-time dashboard metrics."""
        async with self.session.get(f"{self.base_url}/metrics/dashboard/{simulation_id}") as response:
            response.raise_for_status()
            return await response.json()
    
    async def get_extended_metrics(self, simulation_id: str) -> Dict[str, Any]:
        """Get extended metrics from enhanced agents."""
        async with self.session.get(f"{self.base_url}/extended-metrics/{simulation_id}") as response:
            response.raise_for_status()
            return await response.json()

class ERISCLIApplication:
    """Professional CLI application showcasing ERIS capabilities."""
    
    def __init__(self):
        self.client = ERISAPIClient()
    
    async def display_system_status(self):
        """Display comprehensive system status."""
        async with ERISAPIClient() as client:
            try:
                # Get system info and health
                health = await client.health_check()
                system_info = await client.get_system_info()
                
                # Create status table
                table = Table(title="🔥 ERIS System Status", show_header=True)
                table.add_column("Component", style="cyan", no_wrap=True)
                table.add_column("Status", style="green")
                table.add_column("Details", style="white")
                
                # API Status
                table.add_row("🌐 API Server", "✅ Online", f"Connected to {API_BASE_URL}")
                
                # Agent Systems
                adk_status = "✅ Active" if health.get('adk_status') == 'active' else "❌ Offline"
                enhanced_status = "✅ Active" if health.get('enhanced_agents') == 'active' else "❌ Offline"
                
                table.add_row("🤖 Google ADK Agents", adk_status, "6 agents ready")
                table.add_row("⚡ Enhanced Agents", enhanced_status, "4 specialized agents")
                table.add_row("🔢 Total Agent Types", "✅ Ready", f"{system_info.get('total_agent_types', 10)} agents")
                
                # Capabilities
                capabilities = system_info.get('capabilities', {})
                if capabilities.get('real_time_metrics'):
                    table.add_row("📊 Real-time Metrics", "✅ Enabled", "WebSocket streaming")
                if capabilities.get('cloud_integration'):
                    table.add_row("☁️ Cloud Integration", "✅ Connected", "Firestore + BigQuery + Vertex AI")
                
                # Supported disasters
                disasters = len(system_info.get('supported_disasters', []))
                table.add_row("🌪️ Disaster Types", "✅ Loaded", f"{disasters} scenarios available")
                
                console.print(table)
                console.print(f"\n[bold green]✅ ERIS v{system_info.get('eris_version', '0.4.0')} - All systems operational[/bold green]")
                
            except Exception as e:
                console.print(f"[red]❌ System check failed: {e}[/red]")
    
    async def list_disasters(self):
        """Display available disaster types with details."""
        async with ERISAPIClient() as client:
            try:
                system_info = await client.get_system_info()
                disasters = system_info.get('supported_disasters', DISASTER_TYPES)
                
                table = Table(title="🌪️ Available Disaster Scenarios")
                table.add_column("Disaster Type", style="cyan")
                table.add_column("Severity Range", style="yellow")
                table.add_column("Key Systems Affected", style="magenta")
                
                disaster_details = {
                    'earthquake': ('1-9 Richter', 'Infrastructure, Hospitals, Communications'),
                    'hurricane': ('1-5 Category', 'Evacuation, Supply Chain, Emergency Services'),
                    'flood': ('1-10 Index', 'Transportation, Housing, Public Health'),
                    'wildfire': ('1-10 Index', 'Evacuation, Air Quality, Resources'),
                    'tsunami': ('1-10 Wave Height', 'Coastal Infrastructure, Mass Evacuation'),
                    'volcanic_eruption': ('1-8 VEI', 'Air Traffic, Agriculture, Public Health'),
                    'severe_storm': ('1-5 Intensity', 'Power Grid, Transportation, Communications')
                }
                
                for disaster in disasters:
                    severity, systems = disaster_details.get(disaster, ('1-10', 'Multiple Systems'))
                    table.add_row(disaster.replace('_', ' ').title(), severity, systems)
                
                console.print(table)
                
            except Exception as e:
                console.print(f"[red]Failed to fetch disaster types: {e}[/red]")
    
    async def start_simulation_interactive(self, disaster_type: str, location: str, 
                                         severity: int, duration: int):
        """Start simulation with real-time progress display."""
        async with ERISAPIClient() as client:
            try:
                console.print(f"\n[bold]🚀 Initializing ERIS Simulation...[/bold]")
                console.print(f"📍 Location: {location}")
                console.print(f"💥 Disaster: {disaster_type.title()}")
                console.print(f"⚡ Severity: {severity}/10")
                console.print(f"⏱️  Duration: {duration} hours")
                
                # Start simulation
                with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                    task = progress.add_task("Starting simulation...", total=None)
                    response = await client.start_simulation(disaster_type, location, severity, duration)
                
                simulation_id = response['simulation_id']
                console.print(f"\n[green]✅ Simulation started successfully![/green]")
                console.print(f"[cyan]🆔 Simulation ID: {simulation_id}[/cyan]")
                
                # Display initial agent setup
                agents_info = await client.get_agents_info(simulation_id)
                self.display_agents_table(agents_info)
                
                return simulation_id
                
            except Exception as e:
                console.print(f"[red]❌ Failed to start simulation: {e}[/red]")
                return None
    
    async def monitor_simulation(self, simulation_id: str, duration: int = 30):
        """Monitor simulation with live updates."""
        console.print(f"\n[bold]📊 Monitoring Simulation: {simulation_id[:8]}...[/bold]")
        console.print(f"[yellow]Monitoring for {duration} seconds (Ctrl+C to stop)[/yellow]\n")
        
        async with ERISAPIClient() as client:
            try:
                for i in range(duration):
                    # Get real-time metrics
                    metrics = await client.get_dashboard_metrics(simulation_id)
                    agents = await client.get_agents_info(simulation_id)
                    
                    # Create live display
                    self.display_live_metrics(metrics, agents, i)
                    
                    await asyncio.sleep(1)
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Monitoring stopped by user[/yellow]")
            except Exception as e:
                console.print(f"\n[red]Monitoring error: {e}[/red]")
    
    def display_agents_table(self, agents_info: Dict[str, Any]):
        """Display agents in a formatted table."""
        table = Table(title="🤖 Active AI Agents")
        table.add_column("Agent Name", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Status", style="green")
        table.add_column("Efficiency", style="magenta")
        table.add_column("Progress", style="blue")
        
        agents = agents_info.get('agents', {})
        for agent_id, agent in agents.items():
            agent_type = "🔷 Google ADK" if agent.get('category') == 'adk' else "⚡ Enhanced"
            efficiency = f"{agent.get('efficiency', 95)}%"
            progress = f"{agent.get('progress', 0)}%"
            
            table.add_row(
                agent.get('name', 'Unknown Agent'),
                agent_type,
                agent.get('status', 'unknown').title(),
                efficiency,
                progress
            )
        
        console.print(table)
    
    def display_live_metrics(self, metrics: Dict[str, Any], agents: Dict[str, Any], iteration: int):
        """Display live metrics dashboard."""
        dashboard_data = metrics.get('dashboard_data', {})
        
        # Clear screen and show metrics
        console.clear()
        console.print(f"[bold]📊 Live Metrics Dashboard - Update #{iteration + 1}[/bold]")
        console.print(f"⏰ {datetime.now().strftime('%H:%M:%S')}\n")
        
        # Key metrics
        metrics_table = Table(show_header=False)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")
        metrics_table.add_column("Status", style="green")
        
        alert_level = dashboard_data.get('alert_level', 'UNKNOWN')
        alert_color = {
            'GREEN': 'green', 'YELLOW': 'yellow', 
            'RED': 'red', 'CRITICAL': 'bright_red'
        }.get(alert_level, 'white')
        
        metrics_table.add_row("🚨 Alert Level", alert_level, f"[{alert_color}]●[/{alert_color}]")
        metrics_table.add_row("😰 Panic Index", f"{dashboard_data.get('panic_index', 0)}%", "📈")
        metrics_table.add_row("🏥 Hospital Capacity", f"{dashboard_data.get('hospital_capacity', 0)}%", "⚕️")
        metrics_table.add_row("👥 Population Affected", f"{dashboard_data.get('population_affected', 0):,}", "📊")
        metrics_table.add_row("🔧 Infrastructure Failures", str(dashboard_data.get('infrastructure_failures', 0)), "⚠️")
        metrics_table.add_row("🚑 Emergency Response", f"{dashboard_data.get('emergency_response', 0)}%", "✅")
        
        console.print(metrics_table)
        
        # Agent status summary
        total_agents = agents.get('total_agent_count', 10)
        active_agents = sum(1 for agent in agents.get('agents', {}).values() 
                          if agent.get('status') == 'active')
        
        console.print(f"\n🤖 Agents: {active_agents}/{total_agents} active")

# CLI Commands
@click.group()
def cli():
    """🔥 ERIS - Emergency Response Intelligence System CLI
    
    Professional command-line interface for the ERIS disaster simulation platform.
    Features 10 AI agents, real-time metrics, and cloud integration.
    """
    pass

@cli.command()
async def status():
    """📊 Display comprehensive system status"""
    app = ERISCLIApplication()
    await app.display_system_status()

@cli.command()
async def disasters():
    """🌪️ List available disaster scenarios"""
    app = ERISCLIApplication()
    await app.list_disasters()

@cli.command()
@click.option('--type', 'disaster_type', required=True,
              type=click.Choice(DISASTER_TYPES, case_sensitive=False),
              help='Disaster type to simulate')
@click.option('--location', required=True, help='Geographic location')
@click.option('--severity', type=click.IntRange(1, 10), default=7,
              help='Disaster severity (1-10)')
@click.option('--duration', type=int, default=4,
              help='Simulation duration in hours')
@click.option('--monitor', type=int, default=0,
              help='Monitor simulation for N seconds')
async def simulate(disaster_type: str, location: str, severity: int, 
                  duration: int, monitor: int):
    """🚀 Start a new disaster simulation with optional monitoring"""
    app = ERISCLIApplication()
    
    simulation_id = await app.start_simulation_interactive(
        disaster_type, location, severity, duration
    )
    
    if simulation_id and monitor > 0:
        await app.monitor_simulation(simulation_id, monitor)

@cli.command()
@click.argument('simulation_id')
@click.option('--duration', type=int, default=30,
              help='Monitoring duration in seconds')
async def monitor(simulation_id: str, duration: int):
    """📊 Monitor an active simulation with live metrics"""
    app = ERISCLIApplication()
    await app.monitor_simulation(simulation_id, duration)

def main():
    """Main entry point with async support."""
    import asyncio
    
    def run_async_command(cmd):
        def wrapper(*args, **kwargs):
            return asyncio.run(cmd(*args, **kwargs))
        return wrapper
    
    # Convert async commands
    for command in [status, disasters, simulate, monitor]:
        if asyncio.iscoroutinefunction(command.callback):
            command.callback = run_async_command(command.callback)
    
    # Display banner
    console.print(Panel.fit(
        "[bold blue]🔥 ERIS[/bold blue]\n"
        "[italic]Emergency Response Intelligence System[/italic]\n"
        "🤖 10 AI Agents • ☁️ Cloud Integration • 📊 Real-time Analytics",
        style="blue"
    ))
    
    cli()

if __name__ == '__main__':
    main()
