"""
ERIS Communications Director Agent - Compact version
Public information management and media coordination
"""

import logging
from datetime import datetime
from typing import Dict, Any, List

from google.adk.agents import Agent as LlmAgent
from agents.base_agent import ERISAgentFactory
from services import CloudServices
from utils.time_utils import SimulationPhase

logger = logging.getLogger(__name__)

class CommunicationsDirectorAgent:
    def __init__(self, cloud_services: CloudServices):
        self.cloud = cloud_services
        self.agent_id = "communications_director"
        self.agent_type = "communications"
        
        # Communication channels
        self.channels = {
            "emergency_broadcast": {"reach": 400000, "reliability": 0.95},
            "social_media": {"reach": 250000, "reliability": 0.70},
            "local_media": {"reach": 125000, "reliability": 0.80},
            "mobile_alerts": {"reach": 350000, "reliability": 0.90}
        }
        
        # Communication metrics
        self.messages_sent = 0
        self.total_reach = 0
        self.public_trust = 0.8
        self.misinformation_counters = 0
        
        # Simulation state
        self.simulation_id = None
        self.current_phase = SimulationPhase.IMPACT
        self.disaster_type = None
        self.disaster_severity = 5
        
        self.adk_agent = self._create_agent()
    
    def _create_agent(self) -> LlmAgent:
        def send_public_message(message_type: str, content: str, audience: str = "general", priority: str = "medium") -> Dict[str, Any]:
            """Send public message through appropriate channels"""
            # Select channels based on priority
            if priority == "critical":
                selected_channels = ["emergency_broadcast", "mobile_alerts", "social_media"]
            elif priority == "high":
                selected_channels = ["mobile_alerts", "social_media", "local_media"]
            else:
                selected_channels = ["social_media", "local_media"]
            
            # Calculate reach
            total_reach = 0
            for channel in selected_channels:
                if channel in self.channels:
                    total_reach += int(self.channels[channel]["reach"] * 0.8)  # 80% effective reach
            
            self.messages_sent += 1
            self.total_reach += total_reach
            
            # Update trust based on message quality
            if priority == "critical" and "official" in content.lower():
                self.public_trust = min(1.0, self.public_trust + 0.02)
            
            return {
                "message_type": message_type,
                "priority": priority,
                "channels_used": selected_channels,
                "estimated_reach": total_reach,
                "message_id": f"msg_{self.messages_sent}",
                "status": "sent"
            }
        
        def coordinate_media(activity: str, key_messages: str = "", urgency: str = "standard") -> Dict[str, Any]:
            """Coordinate media relations"""
            message_list = [msg.strip() for msg in key_messages.split(',') if msg.strip()]
            if not message_list:
                message_list = ["Emergency response is coordinated", "Public safety is priority", "Updates will follow"]
            
            # Estimate media engagement
            if urgency == "immediate":
                outlets_engaged = 6
                estimated_reach = 500000
            elif urgency == "urgent":
                outlets_engaged = 4
                estimated_reach = 300000
            else:
                outlets_engaged = 3
                estimated_reach = 200000
            
            return {
                "activity": activity,
                "urgency": urgency,
                "outlets_engaged": outlets_engaged,
                "key_messages": message_list,
                "estimated_reach": estimated_reach,
                "media_engagement": "active"
            }
        
        def counter_misinformation(topic: str, correct_info: str, scope: str = "targeted") -> Dict[str, Any]:
            """Counter misinformation with accurate information"""
            self.misinformation_counters += 1
            
            # Select channels for counter-messaging
            if scope == "wide":
                channels = ["social_media", "local_media", "emergency_broadcast"]
                reach = 400000
            else:  # targeted
                channels = ["social_media"]
                reach = 150000
            
            # Estimate effectiveness
            effectiveness = 75 if self.public_trust > 0.7 else 60
            
            return {
                "topic": topic,
                "counter_id": f"counter_{self.misinformation_counters}",
                "channels": channels,
                "estimated_reach": reach,
                "effectiveness": effectiveness,
                "scope": scope,
                "status": "deployed"
            }
        
        def assess_communication_effectiveness() -> Dict[str, Any]:
            """Assess overall communication effectiveness"""
            avg_reach = self.total_reach / max(1, self.messages_sent)
            
            # Calculate effectiveness score
            reach_score = min(100, (avg_reach / 300000) * 100)  # Target 300k avg reach
            trust_score = self.public_trust * 100
            counter_score = min(100, self.misinformation_counters * 20)  # 20 points per counter
            
            overall_score = (reach_score + trust_score + counter_score) / 3
            
            return {
                "messages_sent": self.messages_sent,
                "total_reach": self.total_reach,
                "average_reach": int(avg_reach),
                "public_trust": round(self.public_trust, 2),
                "misinformation_counters": self.misinformation_counters,
                "effectiveness_score": round(overall_score, 1),
                "communication_status": "effective" if overall_score > 70 else "moderate"
            }
        
        return ERISAgentFactory.create_eris_agent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            custom_tools=[send_public_message, coordinate_media, 
                         counter_misinformation, assess_communication_effectiveness]
        )
    
    async def initialize_for_simulation(self, simulation_id: str, disaster_type: str, severity: int, location: str):
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.disaster_severity = severity
        
        # Reset communication state
        self.messages_sent = 0
        self.total_reach = 0
        self.public_trust = 0.8
        self.misinformation_counters = 0
        
        logger.info(f"Communications Director Agent initialized for {disaster_type}")
    
    async def process_phase(self, phase: SimulationPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        self.current_phase = phase
        
        # Simple phase processing
        if phase == SimulationPhase.IMPACT:
            # Send emergency alerts
            self.messages_sent = 2
            self.total_reach = 600000
            self.public_trust = 0.82
            result = {"action": "emergency_alerts", "alerts_sent": 2}
            
        elif phase == SimulationPhase.RESPONSE:
            # Regular updates and media coordination
            self.messages_sent = 5
            self.total_reach = 1200000
            self.misinformation_counters = 2
            self.public_trust = 0.85
            result = {"action": "sustained_communication", "media_engaged": True}
            
        else:  # RECOVERY
            # Recovery messaging
            self.messages_sent = 7
            self.total_reach = 1500000
            self.public_trust = 0.88
            result = {"action": "recovery_communication", "trust_maintained": True}
        
        # Generate metrics
        metrics = await self._generate_metrics()
        
        # Save state
        await self._save_state(metrics)
        
        return {
            "agent_id": self.agent_id,
            "phase": phase.value,
            "metrics": metrics,
            "actions": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _generate_metrics(self) -> Dict[str, Any]:
        avg_reach = self.total_reach / max(1, self.messages_sent)
        
        return {
            "messages_sent": self.messages_sent,
            "total_reach": self.total_reach,
            "average_reach": int(avg_reach),
            "public_trust": round(self.public_trust, 2),
            "misinformation_counters": self.misinformation_counters,
            "channels_available": len(self.channels),
            "communication_effectiveness": "high" if self.public_trust > 0.8 else "moderate"
        }
    
    async def _save_state(self, metrics: Dict[str, Any]):
        try:
            await self.cloud.firestore.save_agent_state(self.agent_id, self.simulation_id, {
                "metrics": metrics,
                "phase": self.current_phase.value,
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")


def create_communications_director_agent(cloud_services: CloudServices) -> CommunicationsDirectorAgent:
    return CommunicationsDirectorAgent(cloud_services)
