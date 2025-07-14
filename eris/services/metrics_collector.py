"""
ERIS Metrics Collection & Dashboard Output Module
Aggregates data across all agents and generates real-time metrics for dashboard integration
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

# Use existing ERIS services
from services import get_cloud_services
from utils.time_utils import SimulationPhase
from config import ERISConfig
import websockets

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    CRITICAL = "critical"

@dataclass
class CompositeMetrics:
    """Core composite metrics for ERIS dashboard"""
    panic_index: float  # 0-100
    hospital_load_percent: float  # 0-100
    infrastructure_failure_count: int
    population_impact_score: float
    response_effectiveness_rating: float  # 0-10
    information_quality_index: float  # 0-100
    resilience_score: float  # 0-100
    timestamp: str
    alert_level: AlertLevel

@dataclass
class AgentMetrics:
    """Individual agent metrics structure"""
    agent_name: str
    status: str
    actions_taken: int
    effectiveness_score: float
    resource_utilization: float
    last_update: str
    specific_metrics: Dict[str, Any]

class ERISMetricsCollector:
    """
    Main metrics aggregation and dashboard output engine for ERIS
    Integrates with existing ERIS orchestrator and cloud services
    """
    
    def __init__(self, simulation_id: str = None):
        # Use existing ERIS infrastructure
        self.cloud = get_cloud_services()
        self.config = ERISConfig()
        self.simulation_id = simulation_id
        
        # Metrics storage
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.composite_metrics: Optional[CompositeMetrics] = None
        self.historical_data: List[Dict] = []
        
        # WebSocket connections for real-time updates
        self.websocket_clients = set()
        self.websocket_port = 8765
    
    
    async def collect_agent_metrics(self, orchestrator) -> Dict[str, AgentMetrics]:
        """
        Aggregate raw agent metrics from existing ERIS orchestrator
        Integrates with your ERISOrchestrator and enhanced agents
        """
        logger.info("Collecting metrics from ERIS orchestrator")
        
        try:
            # Get metrics from orchestrator
            all_metrics = await orchestrator.request_metrics()
            
            # Process ADK agent metrics
            adk_metrics = all_metrics.get("adk_agents", {})
            for agent_id, metrics in adk_metrics.items():
                agent_metrics = AgentMetrics(
                    agent_name=agent_id,
                    status=metrics.get('status', 'active'),
                    actions_taken=0,  # ADK agents don't track this directly
                    effectiveness_score=0.8,  # Default effectiveness
                    resource_utilization=0.6,  # Default utilization
                    last_update=datetime.now().isoformat(),
                    specific_metrics=metrics
                )
                self.agent_metrics[agent_id] = agent_metrics
            
            # Process Enhanced agent metrics
            enhanced_metrics = all_metrics.get("enhanced_agents", {})
            for agent_name, metrics in enhanced_metrics.items():
                
                # Extract specific metrics based on agent type
                if agent_name == 'hospital_load' and 'capacity_metrics' in metrics:
                    specific_metrics = {
                        'hospital_capacity_utilization': metrics['capacity_metrics'].get('bed_utilization_percentage', 80),
                        'icu_utilization': metrics['capacity_metrics'].get('icu_utilization_percentage', 70),
                        'staff_availability': metrics['capacity_metrics'].get('staff_utilization_percentage', 85)
                    }
                elif agent_name == 'public_behavior' and 'behavioral_metrics' in metrics:
                    specific_metrics = {
                        'panic_index': metrics['behavioral_metrics'].get('panic_index', 0.0),
                        'evacuation_compliance': metrics['evacuation_metrics'].get('evacuation_compliance_rate', 0.0)
                    }
                elif agent_name == 'social_media' and 'current_panic_index' in metrics:
                    specific_metrics = {
                        'social_panic_index': metrics.get('current_panic_index', 0.0),
                        'misinformation_level': metrics.get('current_misinformation_level', 0.0)
                    }
                else:
                    specific_metrics = metrics
                
                agent_metrics = AgentMetrics(
                    agent_name=agent_name,
                    status='active',
                    actions_taken=len(metrics.keys()),
                    effectiveness_score=0.8,
                    resource_utilization=0.7,
                    last_update=datetime.now().isoformat(),
                    specific_metrics=specific_metrics
                )
                self.agent_metrics[agent_name] = agent_metrics
            
            return self.agent_metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics from orchestrator: {e}")
            return self.agent_metrics
    
    def _extract_basic_metrics(self, agent: Any) -> Dict[str, Any]:
        """Extract basic metrics from agents without direct metrics support"""
        return {
            'status': getattr(agent, 'status', 'unknown'),
            'actions_taken': getattr(agent, 'actions_count', 0),
            'effectiveness_score': getattr(agent, 'effectiveness', 0.0),
            'resource_utilization': getattr(agent, 'resource_usage', 0.0),
            'specific_metrics': {}
        }
    
    def calculate_composite_scores(self, disaster_type: str, simulation_context: Dict[str, Any]) -> CompositeMetrics:
        """
        Calculate composite metrics using existing ERIS simulation context
        """
        logger.info("Calculating composite scores from ERIS context")
        
        # Extract values from simulation context (from your orchestrator)
        base_population = simulation_context.get('total_population', 175000)
        
        # FIXED: Handle disaster config properly - use hardcoded multipliers
        disaster_multipliers = {
            'earthquake': 1.6,
            'hurricane': 1.8,
            'wildfire': 1.5,
            'flood': 1.7,
            'tsunami': 2.0,
            'volcanic_eruption': 1.9,
            'severe_storm': 1.4,
            'epidemic': 1.6,
            'pandemic': 2.0,
            'landslide': 1.3
        }
        
        multiplier = disaster_multipliers.get(disaster_type.lower(), 1.5)
        
        # Calculate Panic Index from multiple sources
        panic_factors = []
        
        # From public behavior agent
        if 'public_behavior' in self.agent_metrics:
            behavior_metrics = self.agent_metrics['public_behavior'].specific_metrics
            panic_factors.append(behavior_metrics.get('panic_index', 0.0) * 100)
        
        # From social media agent  
        if 'social_media' in self.agent_metrics:
            social_metrics = self.agent_metrics['social_media'].specific_metrics
            panic_factors.append(social_metrics.get('social_panic_index', 0.0) * 100)
        
        # From simulation context
        panic_factors.append(simulation_context.get('panic_index', 0.0) * 100)
        
        panic_index = np.mean(panic_factors) if panic_factors else 25.0
        
        # Calculate Hospital Load from hospital agent or context
        hospital_load = simulation_context.get('hospital_capacity_utilization', 80.0)
        if 'hospital_load' in self.agent_metrics:
            hospital_metrics = self.agent_metrics['hospital_load'].specific_metrics
            hospital_load = hospital_metrics.get('hospital_capacity_utilization', hospital_load)
        
        # Infrastructure failures from context
        infrastructure_failures = max(0, int(simulation_context.get('infrastructure_damage', 0) / 10))
        
        # Population impact using disaster multiplier
        population_impact = (base_population * multiplier) / 1000  # Scale for readability
        
        # Response effectiveness from agent performance
        effectiveness_scores = []
        for agent in self.agent_metrics.values():
            if agent.status == 'active':
                effectiveness_scores.append(agent.effectiveness_score)
        
        response_effectiveness = (np.mean(effectiveness_scores) * 10) if effectiveness_scores else 5.0
        
        # Information quality from social media and communication
        info_quality = 100.0 - (simulation_context.get('panic_index', 0.0) * 50)
        if 'social_media' in self.agent_metrics:
            social_metrics = self.agent_metrics['social_media'].specific_metrics
            misinformation = social_metrics.get('misinformation_level', 0.0)
            info_quality = max(20, 100 - (misinformation * 100))
        
        # Resilience score from overall system health
        active_agents = len([a for a in self.agent_metrics.values() if a.status == 'active'])
        total_agents = len(self.agent_metrics)
        agent_resilience = (active_agents / total_agents * 100) if total_agents > 0 else 50
        
        avg_utilization = np.mean([a.resource_utilization for a in self.agent_metrics.values()])
        resource_resilience = (1 - avg_utilization) * 100
        
        resilience_score = (agent_resilience * 0.5 + resource_resilience * 0.3 + (response_effectiveness * 10) * 0.2)
        
        # Determine alert level
        alert_level = self._calculate_alert_level(panic_index, hospital_load, infrastructure_failures, response_effectiveness)
        
        self.composite_metrics = CompositeMetrics(
            panic_index=round(panic_index, 2),
            hospital_load_percent=round(hospital_load, 2),
            infrastructure_failure_count=infrastructure_failures,
            population_impact_score=round(population_impact, 2),
            response_effectiveness_rating=round(response_effectiveness, 2),
            information_quality_index=round(info_quality, 2),
            resilience_score=round(resilience_score, 2),
            timestamp=datetime.now().isoformat(),
            alert_level=alert_level
        )
        
        return self.composite_metrics
    
    def _calculate_alert_level(self, panic_index: float, hospital_load: float, 
                             infrastructure_failures: int, response_effectiveness: float) -> AlertLevel:
        """Determine overall alert level based on key metrics"""
        
        critical_conditions = (
            panic_index > 80 or 
            hospital_load > 90 or 
            infrastructure_failures > 10 or 
            response_effectiveness < 3
        )
        
        red_conditions = (
            panic_index > 60 or 
            hospital_load > 75 or 
            infrastructure_failures > 5 or 
            response_effectiveness < 5
        )
        
        yellow_conditions = (
            panic_index > 40 or 
            hospital_load > 60 or 
            infrastructure_failures > 2 or 
            response_effectiveness < 7
        )
        
        if critical_conditions:
            return AlertLevel.CRITICAL
        elif red_conditions:
            return AlertLevel.RED
        elif yellow_conditions:
            return AlertLevel.YELLOW
        else:
            return AlertLevel.GREEN
    
    def generate_dashboard_json(self) -> Dict[str, Any]:
        """
        Generate React-dashboard-ready JSON payload
        """
        if not self.composite_metrics:
            logger.warning("No composite metrics available for dashboard JSON")
            return {"error": "No metrics available"}
        
        # Prepare agent summary for dashboard
        agent_summary = []
        for agent_name, metrics in self.agent_metrics.items():
            agent_summary.append({
                "name": agent_name,
                "status": metrics.status,
                "effectiveness": metrics.effectiveness_score,
                "actions": metrics.actions_taken,
                "utilization": metrics.resource_utilization
            })
        
        # Main dashboard payload
        dashboard_data = {
            "timestamp": self.composite_metrics.timestamp,
            "alert_level": self.composite_metrics.alert_level.value,
            "metrics": {
                "panic_index": self.composite_metrics.panic_index,
                "hospital_capacity": 100 - self.composite_metrics.hospital_load_percent,  # Invert for capacity display
                "hospital_load": self.composite_metrics.hospital_load_percent,
                "infrastructure_failures": self.composite_metrics.infrastructure_failure_count,
                "population_affected": self.composite_metrics.population_impact_score * 1000,  # Scale back up
                "response_effectiveness": self.composite_metrics.response_effectiveness_rating,
                "information_quality": self.composite_metrics.information_quality_index,
                "resilience_score": self.composite_metrics.resilience_score
            },
            "agents": agent_summary,
            "trends": self._generate_trend_data(),
            "alerts": self._generate_active_alerts()
        }
        
        return dashboard_data
    
    def _generate_trend_data(self) -> Dict[str, List]:
        """Generate time series data for dashboard graphs"""
        # Use last 24 data points from historical data
        recent_data = self.historical_data[-24:] if len(self.historical_data) >= 24 else self.historical_data
        
        return {
            "panic_trend": [d.get("panic_index", 0) for d in recent_data],
            "hospital_trend": [d.get("hospital_load_percent", 0) for d in recent_data],
            "response_trend": [d.get("response_effectiveness_rating", 0) for d in recent_data],
            "timestamps": [d.get("timestamp", "") for d in recent_data]
        }
    
    def _generate_active_alerts(self) -> List[Dict]:
        """Generate active alerts for dashboard"""
        alerts = []
        
        if not self.composite_metrics:
            return alerts
        
        if self.composite_metrics.panic_index > 70:
            alerts.append({
                "level": "high",
                "message": f"High panic levels detected: {self.composite_metrics.panic_index}%",
                "timestamp": self.composite_metrics.timestamp
            })
        
        if self.composite_metrics.hospital_load_percent > 85:
            alerts.append({
                "level": "critical",
                "message": f"Hospital system overloaded: {self.composite_metrics.hospital_load_percent}%",
                "timestamp": self.composite_metrics.timestamp
            })
        
        if self.composite_metrics.infrastructure_failure_count > 5:
            alerts.append({
                "level": "high",
                "message": f"Multiple infrastructure failures: {self.composite_metrics.infrastructure_failure_count}",
                "timestamp": self.composite_metrics.timestamp
            })
        
        return alerts
    
    async def publish_to_firestore(self, collection_name: str = "eris_metrics"):
        """Publish real-time metrics to Firestore using existing ERIS services"""
        if not self.composite_metrics:
            logger.warning("No metrics to publish to Firestore")
            return
        
        try:
            metrics_dict = asdict(self.composite_metrics)
            metrics_dict['alert_level'] = self.composite_metrics.alert_level.value
            
            # Use existing ERIS cloud services
            await self.cloud.firestore.save_simulation_state(self.simulation_id, {
                **metrics_dict,
                "agent_metrics": [asdict(agent) for agent in self.agent_metrics.values()],
                "created_at": datetime.now()
            })
            
            logger.info(f"Published metrics to Firestore for simulation {self.simulation_id}")
            
        except Exception as e:
            logger.error(f"Error publishing to Firestore: {e}")
    
    async def insert_to_bigquery(self, dataset_id: str = "eris_analytics", table_id: str = "metrics_history"):
        """Insert historical metrics to BigQuery using existing ERIS services"""
        if not self.composite_metrics:
            logger.warning("No metrics to insert to BigQuery")
            return
        
        try:
            event_data = {
                "timestamp": self.composite_metrics.timestamp,
                "panic_index": self.composite_metrics.panic_index,
                "hospital_load_percent": self.composite_metrics.hospital_load_percent,
                "infrastructure_failure_count": self.composite_metrics.infrastructure_failure_count,
                "population_impact_score": self.composite_metrics.population_impact_score,
                "response_effectiveness_rating": self.composite_metrics.response_effectiveness_rating,
                "information_quality_index": self.composite_metrics.information_quality_index,
                "resilience_score": self.composite_metrics.resilience_score,
                "alert_level": self.composite_metrics.alert_level.value,
                "agent_count": len(self.agent_metrics),
                "active_agents": len([a for a in self.agent_metrics.values() if a.status == 'active'])
            }
            
            # Use existing ERIS BigQuery service
            await self.cloud.bigquery.log_simulation_event(
                simulation_id=self.simulation_id,
                event_type="metrics_update",
                event_data=event_data
            )
            
            logger.info("Successfully inserted metrics to BigQuery")
                
        except Exception as e:
            logger.error(f"Error inserting to BigQuery: {e}")
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections for real-time dashboard updates"""
        self.websocket_clients.add(websocket)
        logger.info(f"New WebSocket client connected: {len(self.websocket_clients)} total")
        
        try:
            await websocket.wait_closed()
        finally:
            self.websocket_clients.remove(websocket)
            logger.info(f"WebSocket client disconnected: {len(self.websocket_clients)} remaining")
    
    async def broadcast_updates(self):
        """Broadcast real-time updates to all connected WebSocket clients"""
        if not self.websocket_clients:
            return
        
        dashboard_data = self.generate_dashboard_json()
        message = json.dumps(dashboard_data)
        
        # Send to all connected clients
        disconnected_clients = []
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.websocket_clients.discard(client)
    
    async def start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        logger.info(f"Starting WebSocket server on port {self.websocket_port}")
        return await websockets.serve(self.websocket_handler, "localhost", self.websocket_port)
    
    def export_to_csv(self, filename: str = None) -> str:
        """Export current metrics to CSV"""
        import csv
        from io import StringIO
        
        if filename is None:
            filename = f"eris_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        output = StringIO()
        
        if self.composite_metrics:
            fieldnames = ['timestamp', 'panic_index', 'hospital_load_percent', 'infrastructure_failure_count',
                         'population_impact_score', 'response_effectiveness_rating', 'information_quality_index',
                         'resilience_score', 'alert_level']
            
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            metrics_dict = asdict(self.composite_metrics)
            metrics_dict['alert_level'] = self.composite_metrics.alert_level.value
            writer.writerow(metrics_dict)
        
        return output.getvalue()
    
    def print_cli_summary(self):
        """Print color-coded CLI summary of current metrics"""
        if not self.composite_metrics:
            print("❌ No metrics available")
            return
        
        # Color codes
        colors = {
            AlertLevel.GREEN: '\033[92m',    # Green
            AlertLevel.YELLOW: '\033[93m',   # Yellow
            AlertLevel.RED: '\033[91m',      # Red
            AlertLevel.CRITICAL: '\033[95m', # Magenta
        }
        reset_color = '\033[0m'
        
        alert_color = colors.get(self.composite_metrics.alert_level, '')
        
        print(f"\n{'='*60}")
        print(f"{alert_color}🚨 ERIS METRICS SUMMARY - {self.composite_metrics.alert_level.value.upper()}{reset_color}")
        print(f"{'='*60}")
        print(f"⏰ Timestamp: {self.composite_metrics.timestamp}")
        print(f"😰 Panic Index: {self.composite_metrics.panic_index}%")
        print(f"🏥 Hospital Load: {self.composite_metrics.hospital_load_percent}%")
        print(f"⚡ Infrastructure Failures: {self.composite_metrics.infrastructure_failure_count}")
        print(f"👥 Population Impact: {self.composite_metrics.population_impact_score * 1000:,.0f} people")
        print(f"🎯 Response Effectiveness: {self.composite_metrics.response_effectiveness_rating}/10")
        print(f"📊 Information Quality: {self.composite_metrics.information_quality_index}%")
        print(f"💪 Resilience Score: {self.composite_metrics.resilience_score}%")
        print(f"\n🤖 Active Agents: {len([a for a in self.agent_metrics.values() if a.status == 'active'])}/{len(self.agent_metrics)}")
        print(f"{'='*60}\n")
    
    async def run_continuous_collection(self, agents: Dict[str, Any], interval: int = 2):
        """
        Run continuous metrics collection with specified interval (seconds)
        """
        logger.info(f"Starting continuous metrics collection (interval: {interval}s)")
        
        while True:
            try:
                # Collect and calculate metrics
                await self.collect_agent_metrics(agents)
                disaster_type = getattr(self, 'current_disaster_type', 'earthquake')
                self.calculate_composite_scores(disaster_type)
                
                # Store historical data
                if self.composite_metrics:
                    metrics_dict = asdict(self.composite_metrics)
                    metrics_dict['alert_level'] = self.composite_metrics.alert_level.value
                    self.historical_data.append(metrics_dict)
                    
                    # Keep only last 100 data points
                    if len(self.historical_data) > 100:
                        self.historical_data = self.historical_data[-100:]
                
                # Publish to various outputs
                await self.publish_to_firestore()
                await self.insert_to_bigquery()
                await self.broadcast_updates()
                
                # CLI output
                self.print_cli_summary()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in continuous collection: {e}")
                await asyncio.sleep(interval)

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_metrics_collector():
        # Mock agents for testing
        mock_agents = {
            'healthcare_agent': type('MockAgent', (), {
                'status': 'active',
                'actions_count': 15,
                'effectiveness': 0.85,
                'resource_usage': 0.65,
                'get_metrics': lambda: {
                    'status': 'active',
                    'actions_taken': 15,
                    'effectiveness_score': 0.85,
                    'resource_utilization': 0.65,
                    'specific_metrics': {
                        'er_occupancy_percent': 78,
                        'icu_occupancy_percent': 45,
                        'staff_availability_percent': 85
                    }
                }
            })(),
            'social_media_agent': type('MockAgent', (), {
                'status': 'active',
                'actions_count': 25,
                'effectiveness': 0.72,
                'resource_usage': 0.45,
                'get_metrics': lambda: {
                    'status': 'active',
                    'actions_taken': 25,
                    'effectiveness_score': 0.72,
                    'resource_utilization': 0.45,
                    'specific_metrics': {
                        'official_reach': 65,
                        'misinformation_spread': 15
                    }
                }
            })()
        }
        
        # Initialize collector
        collector = ERISMetricsCollector(project_id="test-project")
        collector.current_disaster_type = "flood"
        
        # Collect metrics
        await collector.collect_agent_metrics(mock_agents)
        collector.calculate_composite_scores("flood", base_population=50000)
        
        # Generate outputs
        dashboard_json = collector.generate_dashboard_json()
        print("Dashboard JSON:", json.dumps(dashboard_json, indent=2))
        
        csv_output = collector.export_to_csv()
        print("CSV Output:", csv_output)
        
        collector.print_cli_summary()
    
    # Run test
    asyncio.run(test_metrics_collector())
