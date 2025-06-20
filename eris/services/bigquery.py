"""
BigQuery service integration for ERIS disaster simulation platform.
Handles analytics and historical data storage for simulation events, metrics, and patterns.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import uuid

try:
    from google.cloud import bigquery
    from google.cloud.bigquery import Client, Table, QueryJobConfig
    from google.cloud.exceptions import NotFound
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

import os

logger = logging.getLogger(__name__)

# Simple settings for this service
class Settings:
    def __init__(self):
        self.GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "eris-simulation-project")
        self.BIGQUERY_DATASET_ID = os.getenv("BIGQUERY_DATASET_ID", "eris_simulation_data")

def get_settings():
    return Settings()

class BigQueryService:
    """
    Service for analytics and historical data storage using Google Cloud BigQuery.
    Supports both production and mock modes for development.
    """
    
    def __init__(self, use_mock: bool = False):
        self.settings = get_settings()
        self.use_mock = use_mock or not BIGQUERY_AVAILABLE
        self.project_id = self.settings.GOOGLE_CLOUD_PROJECT_ID
        self.dataset_id = getattr(self.settings, 'BIGQUERY_DATASET_ID', 'eris_simulation_data')
        
        # Table definitions
        self.tables = {
            'simulation_events': f"{self.project_id}.{self.dataset_id}.simulation_events",
            'agent_metrics': f"{self.project_id}.{self.dataset_id}.agent_metrics",
            'disaster_timeline': f"{self.project_id}.{self.dataset_id}.disaster_timeline",
            'historical_patterns': f"{self.project_id}.{self.dataset_id}.historical_patterns"
        }
        
        # Mock storage for development
        self.mock_data = {
            'simulation_events': [],
            'agent_metrics': [],
            'disaster_timeline': [],
            'historical_patterns': []
        }
        
        if not self.use_mock:
            try:
                self.client = Client(project=self.project_id)
                asyncio.create_task(self._ensure_dataset_and_tables())
                logger.info(f"Initialized BigQuery with project {self.project_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize BigQuery: {e}. Using mock mode.")
                self.use_mock = True
                self.client = None
    
    async def _ensure_dataset_and_tables(self):
        """Ensure dataset and tables exist, create if necessary."""
        try:
            # Create dataset if it doesn't exist
            dataset_ref = self.client.dataset(self.dataset_id)
            try:
                await asyncio.to_thread(self.client.get_dataset, dataset_ref)
            except NotFound:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"
                await asyncio.to_thread(self.client.create_dataset, dataset)
                logger.info(f"Created dataset {self.dataset_id}")
            
            # Create tables if they don't exist
            await self._create_tables()
            
        except Exception as e:
            logger.error(f"Error ensuring dataset and tables: {e}")
    
    async def _create_tables(self):
        """Create BigQuery tables with appropriate schemas."""
        try:
            # Define table schemas
            schemas = {
                'simulation_events': [
                    bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("simulation_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                    bigquery.SchemaField("event_type", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("agent_id", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("event_data", "JSON", mode="REQUIRED"),
                    bigquery.SchemaField("phase", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("severity_level", "INTEGER", mode="NULLABLE"),
                    bigquery.SchemaField("location", "STRING", mode="NULLABLE"),
                ],
                'agent_metrics': [
                    bigquery.SchemaField("metric_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("simulation_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("agent_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                    bigquery.SchemaField("agent_type", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("performance_score", "FLOAT", mode="NULLABLE"),
                    bigquery.SchemaField("actions_taken", "INTEGER", mode="NULLABLE"),
                    bigquery.SchemaField("response_time", "FLOAT", mode="NULLABLE"),
                    bigquery.SchemaField("effectiveness_rating", "FLOAT", mode="NULLABLE"),
                    bigquery.SchemaField("resource_utilization", "FLOAT", mode="NULLABLE"),
                    bigquery.SchemaField("custom_metrics", "JSON", mode="NULLABLE"),
                ],
                'disaster_timeline': [
                    bigquery.SchemaField("timeline_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("simulation_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                    bigquery.SchemaField("disaster_type", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("phase", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("severity_level", "INTEGER", mode="REQUIRED"),
                    bigquery.SchemaField("affected_population", "INTEGER", mode="NULLABLE"),
                    bigquery.SchemaField("geographic_area", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("response_actions", "JSON", mode="NULLABLE"),
                    bigquery.SchemaField("outcome_metrics", "JSON", mode="NULLABLE"),
                ],
                'historical_patterns': [
                    bigquery.SchemaField("pattern_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("disaster_type", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("pattern_type", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("confidence_score", "FLOAT", mode="REQUIRED"),
                    bigquery.SchemaField("frequency", "INTEGER", mode="REQUIRED"),
                    bigquery.SchemaField("pattern_data", "JSON", mode="REQUIRED"),
                    bigquery.SchemaField("identified_date", "TIMESTAMP", mode="REQUIRED"),
                    bigquery.SchemaField("validation_status", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("effectiveness_impact", "FLOAT", mode="NULLABLE"),
                ]
            }
            
            for table_name, schema in schemas.items():
                table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
                
                try:
                    await asyncio.to_thread(self.client.get_table, table_id)
                except NotFound:
                    table = bigquery.Table(table_id, schema=schema)
                    
                    # Set partitioning for time-series data
                    if table_name in ['simulation_events', 'agent_metrics', 'disaster_timeline']:
                        table.time_partitioning = bigquery.TimePartitioning(
                            type_=bigquery.TimePartitioningType.DAY,
                            field="timestamp"
                        )
                    
                    await asyncio.to_thread(self.client.create_table, table)
                    logger.info(f"Created table {table_name}")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
    
    async def log_simulation_event(
        self,
        simulation_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        agent_id: Optional[str] = None,
        phase: Optional[str] = None,
        severity_level: Optional[int] = None,
        location: Optional[str] = None
    ) -> str:
        """
        Log a simulation event to BigQuery.
        
        Args:
            simulation_id: Associated simulation ID
            event_type: Type of event
            event_data: Event data to log
            agent_id: Associated agent ID (optional)
            phase: Simulation phase (optional)
            severity_level: Severity level 1-5 (optional)
            location: Event location (optional)
            
        Returns:
            Event ID if successful, empty string otherwise
        """
        try:
            event_id = str(uuid.uuid4())
            
            record = {
                'event_id': event_id,
                'simulation_id': simulation_id,
                'timestamp': datetime.utcnow(),
                'event_type': event_type,
                'agent_id': agent_id,
                'event_data': json.dumps(event_data),
                'phase': phase,
                'severity_level': severity_level,
                'location': location
            }
            
            if self.use_mock:
                self.mock_data['simulation_events'].append(record)
                logger.info(f"Mock: Logged simulation event {event_type} for {simulation_id}")
                return event_id
            
            # Insert into BigQuery
            table_ref = self.client.get_table(self.tables['simulation_events'])
            errors = await asyncio.to_thread(
                self.client.insert_rows_json,
                table_ref,
                [record]
            )
            
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
                return ""
            
            logger.info(f"Logged simulation event {event_type} for {simulation_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error logging simulation event: {e}")
            return ""
    
    async def log_agent_metrics(
        self,
        simulation_id: str,
        agent_id: str,
        agent_type: str,
        performance_score: Optional[float] = None,
        actions_taken: Optional[int] = None,
        response_time: Optional[float] = None,
        effectiveness_rating: Optional[float] = None,
        resource_utilization: Optional[float] = None,
        custom_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log agent performance metrics to BigQuery.
        
        Args:
            simulation_id: Associated simulation ID
            agent_id: Agent identifier
            agent_type: Type of agent
            performance_score: Overall performance score (0-1)
            actions_taken: Number of actions taken
            response_time: Average response time in seconds
            effectiveness_rating: Effectiveness rating (0-1)
            resource_utilization: Resource utilization (0-1)
            custom_metrics: Additional custom metrics
            
        Returns:
            Metric ID if successful, empty string otherwise
        """
        try:
            metric_id = str(uuid.uuid4())
            
            record = {
                'metric_id': metric_id,
                'simulation_id': simulation_id,
                'agent_id': agent_id,
                'timestamp': datetime.utcnow(),
                'agent_type': agent_type,
                'performance_score': performance_score,
                'actions_taken': actions_taken,
                'response_time': response_time,
                'effectiveness_rating': effectiveness_rating,
                'resource_utilization': resource_utilization,
                'custom_metrics': json.dumps(custom_metrics) if custom_metrics else None
            }
            
            if self.use_mock:
                self.mock_data['agent_metrics'].append(record)
                logger.info(f"Mock: Logged metrics for agent {agent_id} in simulation {simulation_id}")
                return metric_id
            
            # Insert into BigQuery
            table_ref = self.client.get_table(self.tables['agent_metrics'])
            errors = await asyncio.to_thread(
                self.client.insert_rows_json,
                table_ref,
                [record]
            )
            
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
                return ""
            
            logger.info(f"Logged metrics for agent {agent_id} in simulation {simulation_id}")
            return metric_id
            
        except Exception as e:
            logger.error(f"Error logging agent metrics: {e}")
            return ""
    
    async def log_disaster_timeline_event(
        self,
        simulation_id: str,
        disaster_type: str,
        phase: str,
        severity_level: int,
        affected_population: Optional[int] = None,
        geographic_area: Optional[str] = None,
        response_actions: Optional[Dict[str, Any]] = None,
        outcome_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log disaster timeline event to BigQuery.
        
        Args:
            simulation_id: Associated simulation ID
            disaster_type: Type of disaster
            phase: Current phase
            severity_level: Severity level 1-5
            affected_population: Number of people affected
            geographic_area: Geographic description
            response_actions: Actions taken in response
            outcome_metrics: Outcome measurements
            
        Returns:
            Timeline ID if successful, empty string otherwise
        """
        try:
            timeline_id = str(uuid.uuid4())
            
            record = {
                'timeline_id': timeline_id,
                'simulation_id': simulation_id,
                'timestamp': datetime.utcnow(),
                'disaster_type': disaster_type,
                'phase': phase,
                'severity_level': severity_level,
                'affected_population': affected_population,
                'geographic_area': geographic_area,
                'response_actions': json.dumps(response_actions) if response_actions else None,
                'outcome_metrics': json.dumps(outcome_metrics) if outcome_metrics else None
            }
            
            if self.use_mock:
                self.mock_data['disaster_timeline'].append(record)
                logger.info(f"Mock: Logged timeline event for {disaster_type} simulation {simulation_id}")
                return timeline_id
            
            # Insert into BigQuery
            table_ref = self.client.get_table(self.tables['disaster_timeline'])
            errors = await asyncio.to_thread(
                self.client.insert_rows_json,
                table_ref,
                [record]
            )
            
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
                return ""
            
            logger.info(f"Logged timeline event for {disaster_type} simulation {simulation_id}")
            return timeline_id
            
        except Exception as e:
            logger.error(f"Error logging disaster timeline event: {e}")
            return ""
    
    async def query_historical_data(
        self,
        query_type: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Query historical simulation data.
        
        Args:
            query_type: Type of query (events, metrics, timeline, patterns)
            filters: Query filters (simulation_id, date_range, disaster_type, etc.)
            limit: Maximum number of records to return
            
        Returns:
            Query results
        """
        try:
            if self.use_mock:
                return self._mock_query_historical_data(query_type, filters, limit)
            
            # Build query based on type and filters
            query = self._build_historical_query(query_type, filters, limit)
            
            if not query:
                logger.warning(f"Unable to build query for type: {query_type}")
                return []
            
            # Execute query
            job_config = QueryJobConfig()
            query_job = await asyncio.to_thread(
                self.client.query,
                query,
                job_config=job_config
            )
            
            results = await asyncio.to_thread(query_job.result)
            
            # Convert to list of dictionaries
            data = [dict(row) for row in results]
            
            logger.info(f"Retrieved {len(data)} historical records for query type {query_type}")
            return data
            
        except Exception as e:
            logger.error(f"Error querying historical data: {e}")
            return []
    
    def _build_historical_query(
        self,
        query_type: str,
        filters: Optional[Dict[str, Any]],
        limit: int
    ) -> str:
        """Build SQL query based on parameters."""
        filters = filters or {}
        
        base_queries = {
            'events': f"""
                SELECT *
                FROM `{self.tables['simulation_events']}`
            """,
            'metrics': f"""
                SELECT *
                FROM `{self.tables['agent_metrics']}`
            """,
            'timeline': f"""
                SELECT *
                FROM `{self.tables['disaster_timeline']}`
            """,
            'patterns': f"""
                SELECT *
                FROM `{self.tables['historical_patterns']}`
            """
        }
        
        if query_type not in base_queries:
            return ""
        
        query = base_queries[query_type]
        where_clauses = []
        
        # Add filters
        if 'simulation_id' in filters:
            where_clauses.append(f"simulation_id = '{filters['simulation_id']}'")
        
        if 'disaster_type' in filters:
            where_clauses.append(f"disaster_type = '{filters['disaster_type']}'")
        
        if 'agent_type' in filters and query_type == 'metrics':
            where_clauses.append(f"agent_type = '{filters['agent_type']}'")
        
        if 'date_range' in filters:
            start_date = filters['date_range'].get('start')
            end_date = filters['date_range'].get('end')
            if start_date:
                where_clauses.append(f"timestamp >= '{start_date}'")
            if end_date:
                where_clauses.append(f"timestamp <= '{end_date}'")
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        return query
    
    def _mock_query_historical_data(
        self,
        query_type: str,
        filters: Optional[Dict[str, Any]],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Mock implementation for development."""
        filters = filters or {}
        
        if query_type not in self.mock_data:
            return []
        
        data = self.mock_data[query_type].copy()
        
        # Apply filters
        if 'simulation_id' in filters:
            data = [d for d in data if d.get('simulation_id') == filters['simulation_id']]
        
        if 'disaster_type' in filters:
            data = [d for d in data if d.get('disaster_type') == filters['disaster_type']]
        
        # Sort by timestamp descending and apply limit
        data.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        data = data[:limit]
        
        logger.info(f"Mock: Retrieved {len(data)} historical records for query type {query_type}")
        return data
    
    async def create_analytics_report(
        self,
        report_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create analytics report from historical data.
        
        Args:
            report_type: Type of report (performance, effectiveness, trends, comparison)
            parameters: Report parameters
            
        Returns:
            Generated report data
        """
        try:
            if self.use_mock:
                return self._create_mock_analytics_report(report_type, parameters)
            
            report_data = {}
            
            if report_type == 'performance':
                report_data = await self._generate_performance_report(parameters)
            elif report_type == 'effectiveness':
                report_data = await self._generate_effectiveness_report(parameters)
            elif report_type == 'trends':
                report_data = await self._generate_trends_report(parameters)
            elif report_type == 'comparison':
                report_data = await self._generate_comparison_report(parameters)
            else:
                logger.warning(f"Unknown report type: {report_type}")
                return {}
            
            report_data.update({
                'report_type': report_type,
                'generated_at': datetime.utcnow().isoformat(),
                'parameters': parameters
            })
            
            logger.info(f"Generated {report_type} analytics report")
            return report_data
            
        except Exception as e:
            logger.error(f"Error creating analytics report: {e}")
            return {}
    
    async def _generate_performance_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate agent performance analytics report."""
        query = f"""
            SELECT 
                agent_type,
                AVG(performance_score) as avg_performance,
                AVG(response_time) as avg_response_time,
                AVG(effectiveness_rating) as avg_effectiveness,
                COUNT(*) as total_records
            FROM `{self.tables['agent_metrics']}`
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {parameters.get('days', 30)} DAY)
            GROUP BY agent_type
            ORDER BY avg_performance DESC
        """
        
        job_config = QueryJobConfig()
        query_job = await asyncio.to_thread(self.client.query, query, job_config=job_config)
        results = await asyncio.to_thread(query_job.result)
        
        return {
            'agent_performance': [dict(row) for row in results],
            'summary': {
                'total_agents_analyzed': sum(row['total_records'] for row in results),
                'time_period_days': parameters.get('days', 30)
            }
        }
    
    async def _generate_effectiveness_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate disaster response effectiveness report."""
        query = f"""
            SELECT 
                disaster_type,
                phase,
                AVG(severity_level) as avg_severity,
                COUNT(*) as total_events,
                AVG(EXTRACT(EPOCH FROM (
                    SELECT MIN(timestamp) 
                    FROM `{self.tables['simulation_events']}` se2 
                    WHERE se2.simulation_id = dt.simulation_id 
                    AND se2.event_type = 'response_initiated'
                )) - EXTRACT(EPOCH FROM timestamp)) as avg_response_time_seconds
            FROM `{self.tables['disaster_timeline']}` dt
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {parameters.get('days', 30)} DAY)
            GROUP BY disaster_type, phase
            ORDER BY disaster_type, phase
        """
        
        job_config = QueryJobConfig()
        query_job = await asyncio.to_thread(self.client.query, query, job_config=job_config)
        results = await asyncio.to_thread(query_job.result)
        
        return {
            'effectiveness_metrics': [dict(row) for row in results],
            'summary': {
                'time_period_days': parameters.get('days', 30)
            }
        }
    
    async def _generate_trends_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trends analysis report."""
        # This would involve more complex time-series queries
        # Simplified version for demonstration
        return {
            'trends': [],
            'summary': {
                'analysis_period': parameters.get('days', 30),
                'trend_indicators': []
            }
        }
    
    async def _generate_comparison_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison report between simulations or agents."""
        return {
            'comparisons': [],
            'summary': {
                'comparison_type': parameters.get('comparison_type', 'simulations'),
                'baseline': parameters.get('baseline'),
                'targets': parameters.get('targets', [])
            }
        }
    
    def _create_mock_analytics_report(self, report_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create mock analytics report for development."""
        mock_reports = {
            'performance': {
                'agent_performance': [
                    {
                        'agent_type': 'emergency_responder',
                        'avg_performance': 0.85,
                        'avg_response_time': 45.2,
                        'avg_effectiveness': 0.82,
                        'total_records': 150
                    },
                    {
                        'agent_type': 'social_media_monitor',
                        'avg_performance': 0.78,
                        'avg_response_time': 12.5,
                        'avg_effectiveness': 0.75,
                        'total_records': 89
                    }
                ],
                'summary': {
                    'total_agents_analyzed': 239,
                    'time_period_days': parameters.get('days', 30)
                }
            },
            'effectiveness': {
                'effectiveness_metrics': [
                    {
                        'disaster_type': 'earthquake',
                        'phase': 'response',
                        'avg_severity': 3.2,
                        'total_events': 25,
                        'avg_response_time_seconds': 180.5
                    }
                ],
                'summary': {
                    'time_period_days': parameters.get('days', 30)
                }
            },
            'trends': {
                'trends': [
                    {
                        'metric': 'response_time',
                        'trend': 'improving',
                        'change_percent': -15.2
                    }
                ],
                'summary': {
                    'analysis_period': parameters.get('days', 30),
                    'trend_indicators': ['response_time_improving', 'effectiveness_stable']
                }
            },
            'comparison': {
                'comparisons': [
                    {
                        'metric': 'effectiveness',
                        'baseline_score': 0.75,
                        'comparison_score': 0.82,
                        'improvement': 0.07
                    }
                ],
                'summary': {
                    'comparison_type': parameters.get('comparison_type', 'simulations'),
                    'baseline': parameters.get('baseline'),
                    'targets': parameters.get('targets', [])
                }
            }
        }
        
        report = mock_reports.get(report_type, {})
        report.update({
            'report_type': report_type,
            'generated_at': datetime.utcnow().isoformat(),
            'parameters': parameters,
            'mock_data': True
        })
        
        logger.info(f"Mock: Generated {report_type} analytics report")
        return report
    
    async def identify_patterns(
        self,
        disaster_type: str,
        min_confidence: float = 0.7,
        lookback_days: int = 90
    ) -> List[Dict[str, Any]]:
        """
        Identify patterns in historical disaster response data.
        
        Args:
            disaster_type: Type of disaster to analyze
            min_confidence: Minimum confidence score for patterns
            lookback_days: How many days back to analyze
            
        Returns:
            List of identified patterns
        """
        try:
            if self.use_mock:
                return self._identify_mock_patterns(disaster_type, min_confidence)
            
            # Complex pattern identification query
            query = f"""
                WITH response_patterns AS (
                    SELECT 
                        disaster_type,
                        phase,
                        EXTRACT(HOUR FROM timestamp) as hour_of_day,
                        COUNT(*) as frequency,
                        AVG(severity_level) as avg_severity
                    FROM `{self.tables['disaster_timeline']}`
                    WHERE disaster_type = '{disaster_type}'
                        AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {lookback_days} DAY)
                    GROUP BY disaster_type, phase, hour_of_day
                    HAVING COUNT(*) >= 3
                )
                SELECT 
                    disaster_type,
                    phase,
                    hour_of_day,
                    frequency,
                    avg_severity,
                    CASE 
                        WHEN frequency >= 10 THEN 0.9
                        WHEN frequency >= 5 THEN 0.8
                        ELSE 0.7
                    END as confidence_score
                FROM response_patterns
                WHERE CASE 
                    WHEN frequency >= 10 THEN 0.9
                    WHEN frequency >= 5 THEN 0.8
                    ELSE 0.7
                END >= {min_confidence}
                ORDER BY confidence_score DESC, frequency DESC
            """
            
            job_config = QueryJobConfig()
            query_job = await asyncio.to_thread(self.client.query, query, job_config=job_config)
            results = await asyncio.to_thread(query_job.result)
            
            patterns = []
            for row in results:
                pattern_id = str(uuid.uuid4())
                pattern = {
                    'pattern_id': pattern_id,
                    'disaster_type': row['disaster_type'],
                    'pattern_type': 'temporal_response',
                    'confidence_score': float(row['confidence_score']),
                    'frequency': int(row['frequency']),
                    'pattern_data': {
                        'phase': row['phase'],
                        'hour_of_day': int(row['hour_of_day']),
                        'avg_severity': float(row['avg_severity'])
                    },
                    'identified_date': datetime.utcnow(),
                    'validation_status': 'pending',
                    'effectiveness_impact': None
                }
                patterns.append(pattern)
                
                # Store pattern in historical_patterns table
                await self._store_identified_pattern(pattern)
            
            logger.info(f"Identified {len(patterns)} patterns for {disaster_type}")
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying patterns: {e}")
            return []
    
    async def _store_identified_pattern(self, pattern: Dict[str, Any]) -> bool:
        """Store an identified pattern in the historical_patterns table."""
        try:
            record = {
                'pattern_id': pattern['pattern_id'],
                'disaster_type': pattern['disaster_type'],
                'pattern_type': pattern['pattern_type'],
                'confidence_score': pattern['confidence_score'],
                'frequency': pattern['frequency'],
                'pattern_data': json.dumps(pattern['pattern_data']),
                'identified_date': pattern['identified_date'],
                'validation_status': pattern['validation_status'],
                'effectiveness_impact': pattern['effectiveness_impact']
            }
            
            if self.use_mock:
                self.mock_data['historical_patterns'].append(record)
                return True
            
            table_ref = self.client.get_table(self.tables['historical_patterns'])
            errors = await asyncio.to_thread(
                self.client.insert_rows_json,
                table_ref,
                [record]
            )
            
            return len(errors) == 0
            
        except Exception as e:
            logger.error(f"Error storing pattern: {e}")
            return False
    
    def _identify_mock_patterns(self, disaster_type: str, min_confidence: float) -> List[Dict[str, Any]]:
        """Mock pattern identification for development."""
        mock_patterns = [
            {
                'pattern_id': str(uuid.uuid4()),
                'disaster_type': disaster_type,
                'pattern_type': 'temporal_response',
                'confidence_score': 0.85,
                'frequency': 12,
                'pattern_data': {
                    'phase': 'initial',
                    'hour_of_day': 14,
                    'avg_severity': 3.2
                },
                'identified_date': datetime.utcnow(),
                'validation_status': 'pending',
                'effectiveness_impact': None
            },
            {
                'pattern_id': str(uuid.uuid4()),
                'disaster_type': disaster_type,
                'pattern_type': 'resource_allocation',
                'confidence_score': 0.78,
                'frequency': 8,
                'pattern_data': {
                    'resource_type': 'medical',
                    'allocation_pattern': 'centralized_then_distributed'
                },
                'identified_date': datetime.utcnow(),
                'validation_status': 'pending',
                'effectiveness_impact': None
            }
        ]
        
        # Filter by confidence
        patterns = [p for p in mock_patterns if p['confidence_score'] >= min_confidence]
        
        logger.info(f"Mock: Identified {len(patterns)} patterns for {disaster_type}")
        return patterns
    
    async def batch_insert_events(self, events: List[Dict[str, Any]]) -> bool:
        """
        Batch insert multiple events for efficiency.
        
        Args:
            events: List of event records to insert
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not events:
                return True
            
            if self.use_mock:
                self.mock_data['simulation_events'].extend(events)
                logger.info(f"Mock: Batch inserted {len(events)} events")
                return True
            
            # Add required fields and convert timestamps
            processed_events = []
            for event in events:
                if 'event_id' not in event:
                    event['event_id'] = str(uuid.uuid4())
                if 'timestamp' not in event:
                    event['timestamp'] = datetime.utcnow()
                if 'event_data' in event and isinstance(event['event_data'], dict):
                    event['event_data'] = json.dumps(event['event_data'])
                processed_events.append(event)
            
            # Insert in batches to avoid size limits
            batch_size = 1000
            table_ref = self.client.get_table(self.tables['simulation_events'])
            
            for i in range(0, len(processed_events), batch_size):
                batch = processed_events[i:i + batch_size]
                errors = await asyncio.to_thread(
                    self.client.insert_rows_json,
                    table_ref,
                    batch
                )
                
                if errors:
                    logger.error(f"BigQuery batch insert errors: {errors}")
                    return False
            
            logger.info(f"Batch inserted {len(events)} events")
            return True
            
        except Exception as e:
            logger.error(f"Error batch inserting events: {e}")
            return False
    
    async def cleanup_old_data(self, days_old: int = 90) -> bool:
        """
        Clean up old data from BigQuery tables.
        
        Args:
            days_old: Delete data older than this many days
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.use_mock:
                cutoff_date = datetime.utcnow() - timedelta(days=days_old)
                for collection in self.mock_data.values():
                    # Remove old records
                    collection[:] = [
                        record for record in collection
                        if record.get('timestamp', datetime.utcnow()) >= cutoff_date
                    ]
                logger.info(f"Mock: Cleaned up data older than {days_old} days")
                return True
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Clean up each table
            tables_to_clean = ['simulation_events', 'agent_metrics', 'disaster_timeline']
            
            for table_name in tables_to_clean:
                query = f"""
                    DELETE FROM `{self.tables[table_name]}`
                    WHERE timestamp < TIMESTAMP('{cutoff_date.isoformat()}')
                """
                
                job_config = QueryJobConfig()
                query_job = await asyncio.to_thread(
                    self.client.query,
                    query,
                    job_config=job_config
                )
                
                await asyncio.to_thread(query_job.result)
                logger.info(f"Cleaned up old data from {table_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return False