"""
Firestore service integration for ERIS disaster simulation platform.
Handles real-time data storage for simulations, agent states, metrics, and events.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import uuid

try:
    from google.cloud import firestore
    from google.cloud.firestore import AsyncClient
    from google.api_core import retry
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False

import os

logger = logging.getLogger(__name__)

# Simple settings for this service
class Settings:
    def __init__(self):
        self.GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "eris-simulation-project")

def get_settings():
    return Settings()

class FirestoreService:
    """
    Service for real-time data storage using Google Cloud Firestore.
    Supports both production and mock modes for development.
    """
    
    def __init__(self, use_mock: bool = False):
        self.settings = get_settings()
        self.use_mock = use_mock or not FIRESTORE_AVAILABLE
        self.project_id = self.settings.GOOGLE_CLOUD_PROJECT_ID
        
        # Collection names
        self.collections = {
            'simulations': 'simulations',
            'agent_states': 'agent_states',
            'metrics': 'metrics',
            'events': 'events'
        }
        
        # Mock storage for development
        self.mock_data = {
            'simulations': {},
            'agent_states': {},
            'metrics': {},
            'events': {}
        }
        
        if not self.use_mock:
            try:
                self.db = AsyncClient(project=self.project_id)
                logger.info(f"Initialized Firestore with project {self.project_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize Firestore: {e}. Using mock mode.")
                self.use_mock = True
                self.db = None
    
    async def save_simulation_state(
        self,
        simulation_id: str,
        state_data: Dict[str, Any],
        merge: bool = True
    ) -> bool:
        """
        Save or update simulation state data.
        
        Args:
            simulation_id: Unique identifier for the simulation
            state_data: State data to save
            merge: Whether to merge with existing data or overwrite
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add metadata
            state_data.update({
                'simulation_id': simulation_id,
                'last_updated': datetime.utcnow(),
                'version': state_data.get('version', 1) + (1 if merge else 0)
            })
            
            if self.use_mock:
                if merge and simulation_id in self.mock_data['simulations']:
                    self.mock_data['simulations'][simulation_id].update(state_data)
                else:
                    self.mock_data['simulations'][simulation_id] = state_data
                logger.info(f"Mock: Saved simulation state for {simulation_id}")
                logger.info(f"Mock: Current simulations: {list(self.mock_data['simulations'].keys())}")
                return True
            
            doc_ref = self.db.collection(self.collections['simulations']).document(simulation_id)
            
            if merge:
                await doc_ref.set(state_data, merge=True)
            else:
                await doc_ref.set(state_data)
            
            logger.info(f"Saved simulation state for {simulation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving simulation state: {e}")
            return False
    
    async def get_simulation_state(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve simulation state data.
        
        Args:
            simulation_id: Unique identifier for the simulation
            
        Returns:
            Simulation state data or None if not found
        """
        try:
            if self.use_mock:
                logger.info(f"Mock: Looking for simulation {simulation_id}")
                logger.info(f"Mock: Available simulations: {list(self.mock_data['simulations'].keys())}")
                state = self.mock_data['simulations'].get(simulation_id)
                if state:
                    logger.info(f"Mock: Retrieved simulation state for {simulation_id}")
                    return state
                else:
                    logger.warning(f"Mock: Simulation {simulation_id} not found")
                    return None
            
            doc_ref = self.db.collection(self.collections['simulations']).document(simulation_id)
            doc = await doc_ref.get()
            
            if doc.exists:
                logger.info(f"Retrieved simulation state for {simulation_id}")
                return doc.to_dict()
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving simulation state: {e}")
            return None
    
    async def save_agent_state(
        self,
        agent_id: str,
        simulation_id: str,
        agent_data: Dict[str, Any]
    ) -> bool:
        """
        Save agent state data.
        
        Args:
            agent_id: Unique identifier for the agent
            simulation_id: Associated simulation ID
            agent_data: Agent state data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            doc_id = f"{simulation_id}_{agent_id}"
            
            # Add metadata
            agent_data.update({
                'agent_id': agent_id,
                'simulation_id': simulation_id,
                'last_updated': datetime.utcnow(),
                'state_timestamp': datetime.utcnow()
            })
            
            if self.use_mock:
                self.mock_data['agent_states'][doc_id] = agent_data
                logger.info(f"Mock: Saved agent state for {agent_id} in simulation {simulation_id}")
                return True
            
            doc_ref = self.db.collection(self.collections['agent_states']).document(doc_id)
            await doc_ref.set(agent_data, merge=True)
            
            logger.info(f"Saved agent state for {agent_id} in simulation {simulation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving agent state: {e}")
            return False
    
    async def get_agent_metrics(
        self,
        simulation_id: str,
        agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve agent metrics data.
        
        Args:
            simulation_id: Simulation ID to filter by
            agent_id: Specific agent ID (optional)
            
        Returns:
            List of agent metrics
        """
        try:
            if self.use_mock:
                metrics = []
                for key, data in self.mock_data['agent_states'].items():
                    if data.get('simulation_id') == simulation_id:
                        if agent_id is None or data.get('agent_id') == agent_id:
                            metrics.append(data)
                logger.info(f"Mock: Retrieved {len(metrics)} agent metrics for simulation {simulation_id}")
                return metrics
            
            collection_ref = self.db.collection(self.collections['agent_states'])
            query = collection_ref.where('simulation_id', '==', simulation_id)
            
            if agent_id:
                query = query.where('agent_id', '==', agent_id)
            
            docs = await query.get()
            metrics = [doc.to_dict() for doc in docs]
            
            logger.info(f"Retrieved {len(metrics)} agent metrics for simulation {simulation_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error retrieving agent metrics: {e}")
            return []
    
    async def log_event(
        self,
        simulation_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        agent_id: Optional[str] = None
    ) -> str:
        """
        Log a simulation event.
        
        Args:
            simulation_id: Associated simulation ID
            event_type: Type of event (agent_action, system_event, user_input, etc.)
            event_data: Event data to log
            agent_id: Associated agent ID (optional)
            
        Returns:
            Event ID if successful, empty string otherwise
        """
        try:
            event_id = str(uuid.uuid4())
            
            event_record = {
                'event_id': event_id,
                'simulation_id': simulation_id,
                'event_type': event_type,
                'timestamp': datetime.utcnow(),
                'data': event_data
            }
            
            if agent_id:
                event_record['agent_id'] = agent_id
            
            if self.use_mock:
                self.mock_data['events'][event_id] = event_record
                logger.info(f"Mock: Logged {event_type} event for simulation {simulation_id}")
                return event_id
            
            doc_ref = self.db.collection(self.collections['events']).document(event_id)
            await doc_ref.set(event_record)
            
            logger.info(f"Logged {event_type} event for simulation {simulation_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error logging event: {e}")
            return ""
    
    async def get_events(
        self,
        simulation_id: str,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve simulation events.
        
        Args:
            simulation_id: Simulation ID to filter by
            event_type: Specific event type (optional)
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        try:
            if self.use_mock:
                events = []
                for event_data in self.mock_data['events'].values():
                    if event_data.get('simulation_id') == simulation_id:
                        if event_type is None or event_data.get('event_type') == event_type:
                            events.append(event_data)
                
                # Sort by timestamp descending
                events.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
                events = events[:limit]
                
                logger.info(f"Mock: Retrieved {len(events)} events for simulation {simulation_id}")
                return events
            
            collection_ref = self.db.collection(self.collections['events'])
            query = collection_ref.where('simulation_id', '==', simulation_id)
            
            if event_type:
                query = query.where('event_type', '==', event_type)
            
            query = query.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
            docs = await query.get()
            events = [doc.to_dict() for doc in docs]
            
            logger.info(f"Retrieved {len(events)} events for simulation {simulation_id}")
            return events
            
        except Exception as e:
            logger.error(f"Error retrieving events: {e}")
            return []
    
    async def update_realtime_metrics(
        self,
        simulation_id: str,
        metrics_data: Dict[str, Any]
    ) -> bool:
        """
        Update real-time simulation metrics.
        
        Args:
            simulation_id: Associated simulation ID
            metrics_data: Metrics data to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add metadata
            metrics_record = {
                'simulation_id': simulation_id,
                'timestamp': datetime.utcnow(),
                'metrics': metrics_data
            }
            
            if self.use_mock:
                # Use timestamp as unique key for mock storage
                key = f"{simulation_id}_{datetime.utcnow().isoformat()}"
                self.mock_data['metrics'][key] = metrics_record
                logger.info(f"Mock: Updated realtime metrics for simulation {simulation_id}")
                return True
            
            # Use subcollection for metrics to allow multiple entries per simulation
            doc_ref = (self.db.collection(self.collections['simulations'])
                      .document(simulation_id)
                      .collection('realtime_metrics')
                      .document())
            
            await doc_ref.set(metrics_record)
            
            # Also update the main simulation document with latest metrics
            sim_ref = self.db.collection(self.collections['simulations']).document(simulation_id)
            await sim_ref.set({
                'latest_metrics': metrics_data,
                'metrics_updated': datetime.utcnow()
            }, merge=True)
            
            logger.info(f"Updated realtime metrics for simulation {simulation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating realtime metrics: {e}")
            return False
    
    async def get_realtime_metrics(
        self,
        simulation_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent real-time metrics.
        
        Args:
            simulation_id: Simulation ID to filter by
            limit: Maximum number of metric records to return
            
        Returns:
            List of metrics records
        """
        try:
            if self.use_mock:
                metrics = []
                for key, data in self.mock_data['metrics'].items():
                    if data.get('simulation_id') == simulation_id:
                        metrics.append(data)
                
                # Sort by timestamp descending
                metrics.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
                metrics = metrics[:limit]
                
                logger.info(f"Mock: Retrieved {len(metrics)} metric records for simulation {simulation_id}")
                return metrics
            
            collection_ref = (self.db.collection(self.collections['simulations'])
                             .document(simulation_id)
                             .collection('realtime_metrics'))
            
            query = collection_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
            docs = await query.get()
            metrics = [doc.to_dict() for doc in docs]
            
            logger.info(f"Retrieved {len(metrics)} metric records for simulation {simulation_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error retrieving realtime metrics: {e}")
            return []
    
    async def delete_simulation_data(self, simulation_id: str) -> bool:
        """
        Delete all data associated with a simulation.
        
        Args:
            simulation_id: Simulation ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.use_mock:
                # Remove from all mock collections
                for collection in self.mock_data.values():
                    keys_to_remove = [
                        key for key, data in collection.items()
                        if (isinstance(data, dict) and data.get('simulation_id') == simulation_id) or
                           key.startswith(f"{simulation_id}_")
                    ]
                    for key in keys_to_remove:
                        del collection[key]
                
                logger.info(f"Mock: Deleted all data for simulation {simulation_id}")
                return True
            
            # Delete simulation document
            sim_ref = self.db.collection(self.collections['simulations']).document(simulation_id)
            await sim_ref.delete()
            
            # Delete agent states
            agent_query = self.db.collection(self.collections['agent_states']).where('simulation_id', '==', simulation_id)
            agent_docs = await agent_query.get()
            for doc in agent_docs:
                await doc.reference.delete()
            
            # Delete events
            event_query = self.db.collection(self.collections['events']).where('simulation_id', '==', simulation_id)
            event_docs = await event_query.get()
            for doc in event_docs:
                await doc.reference.delete()
            
            logger.info(f"Deleted all data for simulation {simulation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting simulation data: {e}")
            return False
    
    async def get_active_simulations(self) -> List[Dict[str, Any]]:
        """
        Get list of active simulations.
        
        Returns:
            List of active simulation data
        """
        try:
            if self.use_mock:
                active_sims = [
                    data for data in self.mock_data['simulations'].values()
                    if data.get('status') == 'active'
                ]
                logger.info(f"Mock: Retrieved {len(active_sims)} active simulations")
                return active_sims
            
            collection_ref = self.db.collection(self.collections['simulations'])
            query = collection_ref.where('status', '==', 'active')
            docs = await query.get()
            
            active_sims = [doc.to_dict() for doc in docs]
            logger.info(f"Retrieved {len(active_sims)} active simulations")
            return active_sims
            
        except Exception as e:
            logger.error(f"Error retrieving active simulations: {e}")
            return []
    
    async def batch_update_agent_states(
        self,
        simulation_id: str,
        agent_updates: List[Dict[str, Any]]
    ) -> bool:
        """
        Batch update multiple agent states for efficiency.
        
        Args:
            simulation_id: Associated simulation ID
            agent_updates: List of agent update data with 'agent_id' and state data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.use_mock:
                for update in agent_updates:
                    agent_id = update.get('agent_id')
                    if agent_id:
                        doc_id = f"{simulation_id}_{agent_id}"
                        update.update({
                            'simulation_id': simulation_id,
                            'last_updated': datetime.utcnow()
                        })
                        self.mock_data['agent_states'][doc_id] = update
                
                logger.info(f"Mock: Batch updated {len(agent_updates)} agent states")
                return True
            
            # Use Firestore batch writes for atomic updates
            batch = self.db.batch()
            
            for update in agent_updates:
                agent_id = update.get('agent_id')
                if agent_id:
                    doc_id = f"{simulation_id}_{agent_id}"
                    update.update({
                        'simulation_id': simulation_id,
                        'last_updated': datetime.utcnow()
                    })
                    
                    doc_ref = self.db.collection(self.collections['agent_states']).document(doc_id)
                    batch.set(doc_ref, update, merge=True)
            
            await batch.commit()
            logger.info(f"Batch updated {len(agent_updates)} agent states")
            return True
            
        except Exception as e:
            logger.error(f"Error batch updating agent states: {e}")
            return False
    
    async def subscribe_to_simulation_changes(
        self,
        simulation_id: str,
        callback: callable
    ):
        """
        Subscribe to real-time changes in simulation data.
        
        Args:
            simulation_id: Simulation ID to monitor
            callback: Function to call when changes occur
            
        Note:
            This is a placeholder for real-time subscriptions.
            In production, this would use Firestore real-time listeners.
        """
        if self.use_mock:
            logger.info(f"Mock: Subscribed to changes for simulation {simulation_id}")
            # Mock implementation - in real scenario, would set up polling or websockets
            return
        
        try:
            # Set up Firestore real-time listener
            doc_ref = self.db.collection(self.collections['simulations']).document(simulation_id)
            
            def on_snapshot(doc_snapshot, changes, read_time):
                for doc in doc_snapshot:
                    if doc.exists:
                        callback(simulation_id, doc.to_dict())
            
            # This would be the actual listener setup
            # doc_ref.on_snapshot(on_snapshot)
            logger.info(f"Subscribed to changes for simulation {simulation_id}")
            
        except Exception as e:
            logger.error(f"Error setting up subscription: {e}")
    
    async def cleanup_old_data(self, days_old: int = 30) -> bool:
        """
        Clean up old simulation data.
        
        Args:
            days_old: Delete data older than this many days
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            if self.use_mock:
                # Clean up mock data
                for collection in self.mock_data.values():
                    keys_to_remove = []
                    for key, data in collection.items():
                        if isinstance(data, dict):
                            timestamp = data.get('timestamp') or data.get('last_updated')
                            if timestamp and timestamp < cutoff_date:
                                keys_to_remove.append(key)
                    
                    for key in keys_to_remove:
                        del collection[key]
                
                logger.info(f"Mock: Cleaned up data older than {days_old} days")
                return True
            
            # Clean up Firestore collections
            collections_to_clean = ['events', 'agent_states']
            
            for collection_name in collections_to_clean:
                query = (self.db.collection(self.collections[collection_name])
                        .where('timestamp', '<', cutoff_date)
                        .limit(500))  # Batch delete to avoid timeout
                
                docs = await query.get()
                
                if docs:
                    batch = self.db.batch()
                    for doc in docs:
                        batch.delete(doc.reference)
                    await batch.commit()
                    
                    logger.info(f"Deleted {len(docs)} old records from {collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return False