"""
ERIS Social Media Simulation Agent - Compact Version
Generates realistic posts, tracks sentiment, and measures panic influence
"""

import asyncio
import logging
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from google.adk.agents import Agent as LlmAgent
from agents.base_agent import ERISAgentFactory
from services import CloudServices
from utils.time_utils import SimulationPhase

logger = logging.getLogger(__name__)

class PostSentiment(Enum):
    PANIC = "panic"
    FEAR = "fear" 
    CONCERN = "concern"
    HELPFUL = "helpful"
    MISINFORMATION = "misinformation"

@dataclass
class SocialPost:
    content: str
    sentiment: PostSentiment
    engagement: float
    hashtags: List[str]
    user_type: str

class SocialMediaAgent:
    def __init__(self, cloud_services: CloudServices):
        self.cloud = cloud_services
        self.agent_id = "social_media_simulator"
        self.agent_type = "social_media"
        
        self.simulation_id = None
        self.current_phase = SimulationPhase.IMPACT
        self.disaster_type = None
        self.disaster_severity = 5
        self.location = "Phuket, Thailand"
        
        self.posts_generated = []
        self.panic_index = 0.0
        self.misinformation_level = 0.0
        self.viral_topics = []
        
        self.adk_agent = self._create_agent()
        
    def _create_agent(self) -> LlmAgent:
        def generate_posts(disaster_type: str, severity: int, phase: str, count: int) -> Dict[str, Any]:
            """Generate social media posts for current situation"""
            posts = []
            
            # Post templates by phase
            templates = {
                'impact': [
                    ("🚨 {disaster} hitting {location}! Everyone get to safety NOW!", PostSentiment.PANIC, 0.9),
                    ("Roads are completely blocked, can't evacuate from {location}", PostSentiment.FEAR, 0.7),
                    ("Official evacuation order for {location} - follow authorities", PostSentiment.CONCERN, 0.6),
                    ("FAKE: Government hiding real damage numbers! #coverup", PostSentiment.MISINFORMATION, 0.8)
                ],
                'response': [
                    ("Rescue teams doing amazing work in {location} 👏", PostSentiment.HELPFUL, 0.5),
                    ("Shelter at {location} school has space, come if you need it", PostSentiment.HELPFUL, 0.6),
                    ("Still no word from authorities about water safety", PostSentiment.CONCERN, 0.4),
                    ("They're not telling us the truth about casualties", PostSentiment.MISINFORMATION, 0.7)
                ],
                'recovery': [
                    ("Cleanup starting in {location}, community coming together", PostSentiment.HELPFUL, 0.4),
                    ("Insurance companies refusing claims - SCAM!", PostSentiment.MISINFORMATION, 0.6),
                    ("Power restored to most of {location} - progress!", PostSentiment.HELPFUL, 0.3)
                ]
            }
            
            phase_templates = templates.get(phase, templates['impact'])
            
            for i in range(count):
                template = random.choice(phase_templates)
                content = template[0].format(
                    disaster=disaster_type.replace('_', ' ').title(),
                    location=self.location.split(',')[0]
                )
                
                post = SocialPost(
                    content=content,
                    sentiment=template[1],
                    engagement=template[2] * (severity / 10) * random.uniform(0.7, 1.3),
                    hashtags=self._generate_hashtags(disaster_type, template[1]),
                    user_type=random.choice(['local', 'tourist', 'influencer', 'news'])
                )
                posts.append(post)
                self.posts_generated.append(post)
            
            # Calculate metrics
            panic_posts = [p for p in posts if p.sentiment == PostSentiment.PANIC]
            misinfo_posts = [p for p in posts if p.sentiment == PostSentiment.MISINFORMATION]
            
            self.panic_index = sum(p.engagement for p in panic_posts) / max(1, len(posts))
            self.misinformation_level = len(misinfo_posts) / max(1, len(posts))
            
            return {
                "posts_generated": len(posts),
                "panic_index": round(self.panic_index, 3),
                "misinformation_level": round(self.misinformation_level, 3),
                "avg_engagement": round(sum(p.engagement for p in posts) / len(posts), 3),
                "sentiment_breakdown": {
                    sentiment.value: len([p for p in posts if p.sentiment == sentiment])
                    for sentiment in PostSentiment
                },
                "sample_posts": [
                    {"content": p.content[:80] + "...", "sentiment": p.sentiment.value, "engagement": p.engagement}
                    for p in sorted(posts, key=lambda x: x.engagement, reverse=True)[:3]
                ]
            }
        
        def analyze_viral_trends(time_window_hours: int) -> Dict[str, Any]:
            """Analyze viral trends and hashtags"""
            recent_posts = [p for p in self.posts_generated if 
                          (datetime.utcnow() - datetime.utcnow()).total_seconds() < time_window_hours * 3600]
            
            if not recent_posts:
                recent_posts = self.posts_generated[-20:] if self.posts_generated else []
            
            # Count hashtags
            hashtag_counts = {}
            for post in recent_posts:
                for tag in post.hashtags:
                    hashtag_counts[tag] = hashtag_counts.get(tag, 0) + 1
            
            # Find trending
            trending = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Calculate viral potential
            high_engagement = [p for p in recent_posts if p.engagement > 0.7]
            viral_potential = len(high_engagement) / max(1, len(recent_posts))
            
            return {
                "trending_hashtags": [{"hashtag": tag, "count": count} for tag, count in trending],
                "viral_potential": round(viral_potential, 3),
                "posts_analyzed": len(recent_posts),
                "peak_engagement": max([p.engagement for p in recent_posts] + [0])
            }
        
        def calculate_influence_metrics(population_size: int) -> Dict[str, Any]:
            """Calculate social media influence on public behavior"""
            if not self.posts_generated:
                return {"error": "No posts to analyze"}
            
            recent_posts = self.posts_generated[-50:] if len(self.posts_generated) > 50 else self.posts_generated
            
            # Calculate influence factors
            panic_influence = sum(p.engagement for p in recent_posts if p.sentiment == PostSentiment.PANIC)
            helpful_influence = sum(p.engagement for p in recent_posts if p.sentiment == PostSentiment.HELPFUL)
            misinfo_influence = sum(p.engagement for p in recent_posts if p.sentiment == PostSentiment.MISINFORMATION)
            
            total_engagement = sum(p.engagement for p in recent_posts)
            
            # Estimate population reach (simplified)
            estimated_reach = min(population_size * 0.8, total_engagement * 1000)
            
            return {
                "panic_influence_score": round(panic_influence / max(1, total_engagement), 3),
                "helpful_influence_score": round(helpful_influence / max(1, total_engagement), 3),
                "misinformation_influence": round(misinfo_influence / max(1, total_engagement), 3),
                "estimated_population_reach": int(estimated_reach),
                "overall_sentiment": self._calculate_overall_sentiment(recent_posts),
                "behavior_impact": "high" if panic_influence > helpful_influence else "moderate"
            }
        
        return ERISAgentFactory.create_eris_agent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            model="gemini-2.0-flash",
            custom_tools=[generate_posts, analyze_viral_trends, calculate_influence_metrics]
        )
    
    def _generate_hashtags(self, disaster_type: str, sentiment: PostSentiment) -> List[str]:
        """Generate relevant hashtags"""
        disaster_hashtags = {
            'earthquake': ['#Earthquake', '#Seismic'],
            'tsunami': ['#Tsunami', '#TidalWave'],
            'hurricane': ['#Hurricane', '#Storm'],
            'flood': ['#Flood', '#FloodAlert'],
            'wildfire': ['#Wildfire', '#ForestFire'],
            'volcanic_eruption': ['#VolcanicEruption', '#Volcano'],
            'pandemic': ['#Pandemic', '#HealthCrisis'],
            'severe_storm': ['#SevereStorm', '#StormAlert'],
            'epidemic': ['#Epidemic', '#HealthEmergency'],
            'landslide': ['#Landslide', '#Geological']
        }
        
        base_tags = disaster_hashtags.get(disaster_type, [f"#{disaster_type.replace('_', '').title()}"]) + ["#PhuketEmergency"]
        
        sentiment_tags = {
            PostSentiment.PANIC: ["#Emergency", "#Help", "#Urgent"],
            PostSentiment.FEAR: ["#Scared", "#Dangerous", "#Evacuate"],
            PostSentiment.CONCERN: ["#StaySafe", "#Updates", "#Information"],
            PostSentiment.HELPFUL: ["#Community", "#Support", "#Help"],
            PostSentiment.MISINFORMATION: ["#Truth", "#Coverup", "#Conspiracy"]
        }
        
        return base_tags + random.sample(sentiment_tags.get(sentiment, ["#News"]), 2)
    
    def _calculate_overall_sentiment(self, posts: List[SocialPost]) -> str:
        """Calculate overall sentiment trend"""
        sentiment_scores = {
            PostSentiment.PANIC: -1.0,
            PostSentiment.FEAR: -0.8,
            PostSentiment.CONCERN: -0.3,
            PostSentiment.HELPFUL: 0.7,
            PostSentiment.MISINFORMATION: -0.9
        }
        
        weighted_score = sum(sentiment_scores.get(p.sentiment, 0) * p.engagement for p in posts)
        total_weight = sum(p.engagement for p in posts)
        
        if total_weight == 0:
            return "neutral"
        
        avg_sentiment = weighted_score / total_weight
        
        if avg_sentiment < -0.5:
            return "very_negative"
        elif avg_sentiment < -0.2:
            return "negative"
        elif avg_sentiment < 0.2:
            return "neutral"
        else:
            return "positive"
    
    async def initialize_for_simulation(self, simulation_id: str, disaster_type: str, severity: int, location: str):
        """Initialize for simulation"""
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.disaster_severity = severity
        self.location = location
        self.posts_generated = []
        
        logger.info(f"Social Media Agent initialized for {disaster_type} in {location}")
    
    async def process_phase(self, phase: SimulationPhase, simulation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process social media activity for simulation phase"""
        self.current_phase = phase
        
        # Generate posts for this phase
        post_count = {
            SimulationPhase.IMPACT: 25,
            SimulationPhase.RESPONSE: 15,
            SimulationPhase.RECOVERY: 10
        }.get(phase, 15)
        
        # FIXED: Replace ADK tool calls with direct implementations
        try:
            # Generate posts using internal logic
            posts = []
            
            # Post templates by phase
            templates = {
                'impact': [
                    ("🚨 {disaster} hitting {location}! Everyone get to safety NOW!", PostSentiment.PANIC, 0.9),
                    ("Roads are completely blocked, can't evacuate from {location}", PostSentiment.FEAR, 0.7),
                    ("Official evacuation order for {location} - follow authorities", PostSentiment.CONCERN, 0.6),
                    ("FAKE: Government hiding real damage numbers! #coverup", PostSentiment.MISINFORMATION, 0.8),
                    ("Can't reach family in {location}, please share info if you see them", PostSentiment.FEAR, 0.8),
                    ("Emergency services overwhelmed in {location} - help!", PostSentiment.PANIC, 0.9)
                ],
                'response': [
                    ("Rescue teams doing amazing work in {location} 👏", PostSentiment.HELPFUL, 0.5),
                    ("Shelter at {location} school has space, come if you need it", PostSentiment.HELPFUL, 0.6),
                    ("Still no word from authorities about water safety", PostSentiment.CONCERN, 0.4),
                    ("They're not telling us the truth about casualties", PostSentiment.MISINFORMATION, 0.7),
                    ("Donations needed for {location} victims - here's how to help", PostSentiment.HELPFUL, 0.6),
                    ("Communication lines restored in parts of {location}", PostSentiment.HELPFUL, 0.4)
                ],
                'recovery': [
                    ("Cleanup starting in {location}, community coming together", PostSentiment.HELPFUL, 0.4),
                    ("Insurance companies refusing claims - SCAM!", PostSentiment.MISINFORMATION, 0.6),
                    ("Power restored to most of {location} - progress!", PostSentiment.HELPFUL, 0.3),
                    ("Businesses reopening in {location} - economy recovering", PostSentiment.HELPFUL, 0.4),
                    ("Still waiting for government aid in {location}", PostSentiment.CONCERN, 0.5)
                ]
            }
            
            phase_templates = templates.get(phase.value, templates['impact'])
            
            # Generate posts for this phase
            for i in range(post_count):
                template = random.choice(phase_templates)
                content = template[0].format(
                    disaster=self.disaster_type.replace('_', ' ').title(),
                    location=self.location.split(',')[0]
                )
                
                post = SocialPost(
                    content=content,
                    sentiment=template[1],
                    engagement=template[2] * (self.disaster_severity / 10) * random.uniform(0.7, 1.3),
                    hashtags=self._generate_hashtags(self.disaster_type, template[1]),
                    user_type=random.choice(['local', 'tourist', 'influencer', 'news'])
                )
                posts.append(post)
                self.posts_generated.append(post)
            
            # Calculate metrics from generated posts
            panic_posts = [p for p in posts if p.sentiment == PostSentiment.PANIC]
            misinfo_posts = [p for p in posts if p.sentiment == PostSentiment.MISINFORMATION]
            helpful_posts = [p for p in posts if p.sentiment == PostSentiment.HELPFUL]
            fear_posts = [p for p in posts if p.sentiment == PostSentiment.FEAR]
            concern_posts = [p for p in posts if p.sentiment == PostSentiment.CONCERN]
            
            # Update agent state
            self.panic_index = sum(p.engagement for p in panic_posts) / max(1, len(posts))
            self.misinformation_level = len(misinfo_posts) / max(1, len(posts))
            
            posts_result = {
                "posts_generated": len(posts),
                "panic_index": round(self.panic_index, 3),
                "misinformation_level": round(self.misinformation_level, 3),
                "avg_engagement": round(sum(p.engagement for p in posts) / len(posts), 3),
                "sentiment_breakdown": {
                    "panic": len(panic_posts),
                    "fear": len(fear_posts),
                    "concern": len(concern_posts),
                    "helpful": len(helpful_posts),
                    "misinformation": len(misinfo_posts)
                },
                "sample_posts": [
                    {"content": p.content[:80] + "...", "sentiment": p.sentiment.value, "engagement": round(p.engagement, 3)}
                    for p in sorted(posts, key=lambda x: x.engagement, reverse=True)[:3]
                ]
            }
            
            # Analyze viral trends - use recent posts
            recent_posts = self.posts_generated[-20:] if len(self.posts_generated) > 20 else self.posts_generated
            
            # Count hashtags
            hashtag_counts = {}
            for post in recent_posts:
                for tag in post.hashtags:
                    hashtag_counts[tag] = hashtag_counts.get(tag, 0) + 1
            
            # Find trending hashtags
            trending = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Calculate viral potential
            high_engagement = [p for p in recent_posts if p.engagement > 0.7]
            viral_potential = len(high_engagement) / max(1, len(recent_posts))
            
            trends_result = {
                "trending_hashtags": [{"hashtag": tag, "count": count} for tag, count in trending],
                "viral_potential": round(viral_potential, 3),
                "posts_analyzed": len(recent_posts),
                "peak_engagement": round(max([p.engagement for p in recent_posts] + [0]), 3)
            }
            
            # Calculate influence metrics
            if self.posts_generated:
                analysis_posts = self.posts_generated[-50:] if len(self.posts_generated) > 50 else self.posts_generated
                
                # Calculate influence factors
                panic_influence = sum(p.engagement for p in analysis_posts if p.sentiment == PostSentiment.PANIC)
                helpful_influence = sum(p.engagement for p in analysis_posts if p.sentiment == PostSentiment.HELPFUL)
                misinfo_influence = sum(p.engagement for p in analysis_posts if p.sentiment == PostSentiment.MISINFORMATION)
                
                total_engagement = sum(p.engagement for p in analysis_posts)
                
                # Estimate population reach
                estimated_reach = min(175000 * 0.8, total_engagement * 1000)
                
                influence_result = {
                    "panic_influence_score": round(panic_influence / max(1, total_engagement), 3),
                    "helpful_influence_score": round(helpful_influence / max(1, total_engagement), 3),
                    "misinformation_influence": round(misinfo_influence / max(1, total_engagement), 3),
                    "estimated_population_reach": int(estimated_reach),
                    "overall_sentiment": self._calculate_overall_sentiment(analysis_posts),
                    "behavior_impact": "high" if panic_influence > helpful_influence else "moderate"
                }
            else:
                influence_result = {
                    "panic_influence_score": 0.0,
                    "helpful_influence_score": 0.0,
                    "misinformation_influence": 0.0,
                    "estimated_population_reach": 0,
                    "overall_sentiment": "neutral",
                    "behavior_impact": "low"
                }
            
        except Exception as e:
            logger.warning(f"Social media agent tool call error: {e}")
            # Create fallback results
            self.panic_index = min(0.6, self.disaster_severity / 10)
            self.misinformation_level = 0.2
            
            posts_result = {
                "posts_generated": post_count,
                "panic_index": self.panic_index,
                "misinformation_level": self.misinformation_level,
                "avg_engagement": 0.7,
                "sentiment_breakdown": {
                    "panic": 2,
                    "fear": 3, 
                    "concern": 4,
                    "helpful": 3,
                    "misinformation": 1
                }
            }
            
            trends_result = {
                "trending_hashtags": [
                    {"hashtag": f"#{self.disaster_type.replace('_', '').title()}", "count": 50},
                    {"hashtag": "#Emergency", "count": 30}
                ],
                "viral_potential": 0.6,
                "posts_analyzed": post_count
            }
            
            influence_result = {
                "panic_influence_score": 0.3,
                "helpful_influence_score": 0.5,
                "misinformation_influence": 0.2,
                "estimated_population_reach": int(175000 * 0.7),
                "behavior_impact": "moderate"
            }
        
        # Generate comprehensive metrics
        metrics = {
            "posts_generated": posts_result,
            "viral_trends": trends_result,
            "influence_metrics": influence_result,
            "total_posts": len(self.posts_generated),
            "current_panic_index": self.panic_index,
            "current_misinformation_level": self.misinformation_level
        }
        
        # Save to cloud
        await self._save_social_state(metrics)
        await self._log_social_event(phase, metrics)
        
        return {
            "agent_id": self.agent_id,
            "phase": phase.value,
            "social_metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _save_social_state(self, metrics: Dict[str, Any]):
        """Save social media state to Firestore"""
        try:
            state_data = {
                "agent_id": self.agent_id,
                "simulation_id": self.simulation_id,
                "metrics": metrics,
                "posts_count": len(self.posts_generated),
                "panic_index": self.panic_index,
                "misinformation_level": self.misinformation_level,
                "phase": self.current_phase.value,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.cloud.firestore.save_agent_state(self.agent_id, self.simulation_id, state_data)
            
        except Exception as e:
            logger.error(f"Failed to save social media state: {e}")
    
    async def _log_social_event(self, phase: SimulationPhase, metrics: Dict[str, Any]):
        """Log social media events to BigQuery"""
        try:
            event_data = {
                "event_type": "social_media_update",
                "agent_id": self.agent_id,
                "phase": phase.value,
                "posts_generated": metrics["posts_generated"]["posts_generated"],
                "panic_index": self.panic_index,
                "misinformation_level": self.misinformation_level,
                "viral_potential": metrics["viral_trends"]["viral_potential"]
            }
            
            await self.cloud.bigquery.log_simulation_event(
                simulation_id=self.simulation_id,
                event_type="social_media_update",
                event_data=event_data,
                agent_id=self.agent_id,
                phase=phase.value
            )
            
        except Exception as e:
            logger.error(f"Failed to log social media event: {e}")


def create_social_media_agent(cloud_services: CloudServices) -> SocialMediaAgent:
    """Factory function to create Social Media Agent"""
    return SocialMediaAgent(cloud_services)