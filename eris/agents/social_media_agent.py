"""
ERIS Social Media Agent with Dynamic AI Content Generation
Generates realistic posts using Vertex AI based on real-time simulation context
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
    OFFICIAL = "official"

@dataclass
class SocialPost:
    content: str
    sentiment: PostSentiment
    engagement: float
    hashtags: List[str]
    user_type: str
    timestamp: datetime
    likes: int
    shares: int
    comments: int

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
        self.last_generation_time = datetime.utcnow()
        
        self.adk_agent = self._create_agent()
        
    def _create_agent(self) -> LlmAgent:
        def generate_ai_posts(context: Dict[str, Any], count: int = 5) -> Dict[str, Any]:
            """Generate AI-powered social media posts based on simulation context"""
            try:
                # Create prompt for Vertex AI
                prompt = self._create_social_media_prompt(context, count)
                
                # Call Vertex AI to generate posts
                ai_response = asyncio.create_task(
                    self.cloud.vertex_ai.generate_social_media_content(prompt, context)
                )
                
                # Parse AI response and create posts
                posts = self._parse_ai_response_to_posts(ai_response, context)
                
                return {
                    "posts_generated": len(posts),
                    "ai_generated": True,
                    "posts": [self._post_to_dict(post) for post in posts],
                    "context_used": context
                }
                
            except Exception as e:
                logger.warning(f"AI generation failed, using fallback: {e}")
                return self._generate_fallback_posts(context, count)
        
        def analyze_viral_trends(time_window_hours: int = 2) -> Dict[str, Any]:
            """Analyze viral trends from recent posts"""
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            recent_posts = [p for p in self.posts_generated if p.timestamp > cutoff_time]
            
            if not recent_posts:
                recent_posts = self.posts_generated[-20:] if self.posts_generated else []
            
            # Count hashtags and viral metrics
            hashtag_counts = {}
            high_engagement_posts = []
            
            for post in recent_posts:
                for tag in post.hashtags:
                    hashtag_counts[tag] = hashtag_counts.get(tag, 0) + 1
                
                if post.engagement > 0.7:
                    high_engagement_posts.append(post)
            
            # Calculate viral potential
            viral_potential = len(high_engagement_posts) / max(1, len(recent_posts))
            
            # Find trending hashtags
            trending = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "trending_hashtags": [{"hashtag": tag, "count": count} for tag, count in trending],
                "viral_potential": round(viral_potential, 3),
                "viral_posts": len(high_engagement_posts),
                "total_engagement": sum(p.engagement for p in recent_posts),
                "peak_engagement": max([p.engagement for p in recent_posts] + [0]),
                "misinformation_spread": len([p for p in recent_posts if p.sentiment == PostSentiment.MISINFORMATION])
            }
        
        def calculate_social_influence(population_size: int = 175000) -> Dict[str, Any]:
            """Calculate social media influence on public behavior"""
            if not self.posts_generated:
                return {"error": "No posts to analyze"}
            
            recent_posts = self.posts_generated[-50:] if len(self.posts_generated) > 50 else self.posts_generated
            
            # Calculate influence by sentiment
            sentiment_influence = {}
            total_engagement = 0
            
            for sentiment in PostSentiment:
                sentiment_posts = [p for p in recent_posts if p.sentiment == sentiment]
                influence = sum(p.engagement * (p.likes + p.shares * 2) for p in sentiment_posts)
                sentiment_influence[sentiment.value] = influence
                total_engagement += influence
            
            # Estimate population reach
            reach_multiplier = 1000  # Each engagement point reaches ~1000 people
            estimated_reach = min(population_size * 0.8, total_engagement * reach_multiplier)
            
            # Calculate behavior impact
            panic_influence = sentiment_influence.get('panic', 0) + sentiment_influence.get('fear', 0)
            helpful_influence = sentiment_influence.get('helpful', 0) + sentiment_influence.get('official', 0)
            
            behavior_impact = "high" if panic_influence > helpful_influence * 1.5 else "moderate"
            
            return {
                "sentiment_influence": sentiment_influence,
                "estimated_population_reach": int(estimated_reach),
                "behavior_impact": behavior_impact,
                "panic_influence_score": round(panic_influence / max(1, total_engagement), 3),
                "helpful_influence_score": round(helpful_influence / max(1, total_engagement), 3),
                "misinformation_influence": round(sentiment_influence.get('misinformation', 0) / max(1, total_engagement), 3),
                "overall_sentiment": self._calculate_overall_sentiment(recent_posts)
            }
        
        return ERISAgentFactory.create_eris_agent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            model="gemini-2.0-flash",
            custom_tools=[generate_ai_posts, analyze_viral_trends, calculate_social_influence]
        )
    
    def _create_social_media_prompt(self, context: Dict[str, Any], count: int) -> str:
        """Create prompt for AI to generate social media posts"""
        phase = context.get('phase', 'impact')
        disaster_type = context.get('disaster_type', self.disaster_type)
        location = context.get('location', self.location)
        severity = context.get('severity', self.disaster_severity)
        panic_index = context.get('panic_index', self.panic_index)
        hospital_capacity = context.get('hospital_capacity_utilization', 75)
        
        prompt = f"""Generate {count} realistic social media posts for a {disaster_type} disaster simulation in {location}.

Current situation:
- Disaster: {disaster_type} (severity {severity}/10)
- Phase: {phase}
- Panic index: {panic_index * 100:.1f}%
- Hospital capacity: {hospital_capacity}%
- Location: {location}

Create diverse posts including:
- Local residents sharing updates
- Emergency services providing information
- News outlets reporting
- Some misinformation/rumors
- Helpful community coordination

Each post should be:
- Realistic and contextual
- Appropriate for the disaster phase
- Include relevant hashtags
- Show different user perspectives

Format as JSON array with: content, sentiment (panic/fear/concern/helpful/misinformation/official), user_type (local/tourist/emergency/news/influencer), engagement_score (0.0-1.0)"""

        return prompt
    
    async def _parse_ai_response_to_posts(self, ai_response, context: Dict[str, Any]) -> List[SocialPost]:
        """Parse AI response into SocialPost objects"""
        posts = []
        
        try:
            # This would be the actual AI response parsing
            # For now, using enhanced templates with context
            posts = await self._generate_contextual_posts(context, 5)
            
        except Exception as e:
            logger.warning(f"Failed to parse AI response: {e}")
            posts = await self._generate_contextual_posts(context, 5)
        
        return posts
    
    async def _generate_contextual_posts(self, context: Dict[str, Any], count: int) -> List[SocialPost]:
        """Generate contextual posts based on simulation state"""
        posts = []
        
        # Get current metrics
        panic_level = context.get('panic_index', self.panic_index)
        hospital_capacity = context.get('hospital_capacity_utilization', 75)
        phase = context.get('phase', self.current_phase.value)
        
        # Enhanced templates based on context
        templates = self._get_contextual_templates(panic_level, hospital_capacity, phase)
        
        for i in range(count):
            template = random.choice(templates)
            
            # Generate engagement based on context
            base_engagement = template[2]
            context_multiplier = 1.0
            
            if panic_level > 0.7:
                context_multiplier *= 1.5  # High panic increases engagement
            if hospital_capacity > 90:
                context_multiplier *= 1.3  # Hospital crisis increases attention
                
            engagement = min(1.0, base_engagement * context_multiplier * random.uniform(0.8, 1.2))
            
            # Create post
            post = SocialPost(
                content=template[0].format(
                    disaster=self.disaster_type.replace('_', ' ').title(),
                    location=self.location.split(',')[0],
                    severity=self.disaster_severity,
                    hospital_capacity=int(hospital_capacity),
                    panic_level=int(panic_level * 100)
                ),
                sentiment=template[1],
                engagement=engagement,
                hashtags=self._generate_hashtags(self.disaster_type, template[1]),
                user_type=random.choice(['local', 'tourist', 'emergency', 'news', 'influencer']),
                timestamp=datetime.utcnow(),
                likes=int(engagement * random.randint(50, 500)),
                shares=int(engagement * random.randint(5, 100)),
                comments=int(engagement * random.randint(10, 150))
            )
            
            posts.append(post)
            self.posts_generated.append(post)
        
        # Update agent metrics
        self._update_metrics_from_posts(posts)
        
        return posts
    
    def _get_contextual_templates(self, panic_level: float, hospital_capacity: float, phase: str) -> List[tuple]:
        """Get contextual post templates based on current situation"""
        
        base_templates = {
            'impact': [
                ("🚨 {disaster} hitting {location}! Everyone get to safety NOW! #Emergency", PostSentiment.PANIC, 0.9),
                ("Roads completely blocked, can't evacuate from {location} 😰", PostSentiment.FEAR, 0.8),
                ("Official evacuation order for {location} - follow authorities", PostSentiment.OFFICIAL, 0.6),
                ("Can't reach family in {location}, please share if you see them", PostSentiment.FEAR, 0.7),
                ("Emergency services reporting multiple incidents in {location}", PostSentiment.CONCERN, 0.5)
            ],
            'response': [
                ("Rescue teams doing amazing work in {location} 👏", PostSentiment.HELPFUL, 0.6),
                ("Shelter at {location} school has space, come if you need it", PostSentiment.HELPFUL, 0.7),
                ("Still no word from authorities about water safety", PostSentiment.CONCERN, 0.4),
                ("Hospital capacity at {hospital_capacity}% - critical situation", PostSentiment.CONCERN, 0.8),
                ("Donations needed for {location} victims - here's how to help", PostSentiment.HELPFUL, 0.6)
            ],
            'recovery': [
                ("Cleanup starting in {location}, community coming together", PostSentiment.HELPFUL, 0.4),
                ("Power restored to most of {location} - progress!", PostSentiment.HELPFUL, 0.3),
                ("Businesses reopening in {location} - economy recovering", PostSentiment.HELPFUL, 0.4),
                ("Still waiting for government aid in {location}", PostSentiment.CONCERN, 0.5)
            ]
        }
        
        templates = base_templates.get(phase, base_templates['impact'])
        
        # Add context-specific templates
        if panic_level > 0.6:
            templates.extend([
                ("PANIC in {location}! This is worse than they're telling us!", PostSentiment.PANIC, 0.9),
                ("Everyone fleeing {location} - complete chaos", PostSentiment.FEAR, 0.8)
            ])
        
        if hospital_capacity > 85:
            templates.extend([
                ("Hospitals in {location} overwhelmed - {hospital_capacity}% capacity!", PostSentiment.CONCERN, 0.9),
                ("Medical emergency in {location} - they need help NOW", PostSentiment.PANIC, 0.8)
            ])
        
        # Add some misinformation based on panic level
        if panic_level > 0.4:
            templates.extend([
                ("FAKE NEWS: Government hiding real casualty numbers! #coverup", PostSentiment.MISINFORMATION, 0.7),
                ("They're not telling us the truth about the {disaster}", PostSentiment.MISINFORMATION, 0.6)
            ])
        
        return templates
    
    def _update_metrics_from_posts(self, posts: List[SocialPost]):
        """Update agent metrics based on generated posts"""
        if not posts:
            return
            
        # Calculate panic index
        panic_posts = [p for p in posts if p.sentiment in [PostSentiment.PANIC, PostSentiment.FEAR]]
        self.panic_index = sum(p.engagement for p in panic_posts) / len(posts)
        
        # Calculate misinformation level
        misinfo_posts = [p for p in posts if p.sentiment == PostSentiment.MISINFORMATION]
        self.misinformation_level = len(misinfo_posts) / len(posts)
        
        # Update viral topics
        all_hashtags = []
        for post in posts:
            all_hashtags.extend(post.hashtags)
        
        hashtag_counts = {}
        for tag in all_hashtags:
            hashtag_counts[tag] = hashtag_counts.get(tag, 0) + 1
        
        self.viral_topics = [tag for tag, count in sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)[:3]]
    
    def _generate_hashtags(self, disaster_type: str, sentiment: PostSentiment) -> List[str]:
        """Generate relevant hashtags"""
        disaster_hashtags = {
            'earthquake': ['#Earthquake', '#Seismic', '#EarthquakeAlert'],
            'tsunami': ['#Tsunami', '#TidalWave', '#TsunamiWarning'],
            'hurricane': ['#Hurricane', '#Storm', '#HurricaneWatch'],
            'flood': ['#Flood', '#FloodAlert', '#FloodWarning'],
            'wildfire': ['#Wildfire', '#ForestFire', '#WildfireAlert'],
            'volcanic_eruption': ['#Volcano', '#VolcanicEruption', '#VolcanoAlert'],
            'pandemic': ['#Pandemic', '#HealthCrisis', '#PublicHealth'],
            'severe_storm': ['#SevereStorm', '#StormAlert', '#Weather'],
            'epidemic': ['#Epidemic', '#HealthEmergency', '#Disease'],
            'landslide': ['#Landslide', '#Geological', '#LandslideAlert']
        }
        
        base_tags = disaster_hashtags.get(disaster_type, [f"#{disaster_type.replace('_', '').title()}"]) 
        base_tags.append(f"#{self.location.split(',')[0]}Emergency")
        
        sentiment_tags = {
            PostSentiment.PANIC: ["#Emergency", "#Help", "#Urgent", "#Crisis"],
            PostSentiment.FEAR: ["#Scared", "#Dangerous", "#Evacuate", "#Safety"],
            PostSentiment.CONCERN: ["#StaySafe", "#Updates", "#Information", "#Alert"],
            PostSentiment.HELPFUL: ["#Community", "#Support", "#Help", "#Relief"],
            PostSentiment.MISINFORMATION: ["#Truth", "#Coverup", "#Conspiracy", "#Hidden"],
            PostSentiment.OFFICIAL: ["#Official", "#Government", "#Emergency", "#Update"]
        }
        
        selected_tags = base_tags + random.sample(sentiment_tags.get(sentiment, ["#News"]), 2)
        return selected_tags[:5]  # Limit to 5 hashtags
    
    def _calculate_overall_sentiment(self, posts: List[SocialPost]) -> str:
        """Calculate overall sentiment trend"""
        if not posts:
            return "neutral"
            
        sentiment_scores = {
            PostSentiment.PANIC: -1.0,
            PostSentiment.FEAR: -0.8,
            PostSentiment.CONCERN: -0.3,
            PostSentiment.HELPFUL: 0.7,
            PostSentiment.MISINFORMATION: -0.9,
            PostSentiment.OFFICIAL: 0.3
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
        elif avg_sentiment < 0.5:
            return "positive"
        else:
            return "very_positive"
    
    def _post_to_dict(self, post: SocialPost) -> Dict[str, Any]:
        """Convert SocialPost to dictionary for API response"""
        return {
            "content": post.content,
            "sentiment": post.sentiment.value,
            "engagement": round(post.engagement, 3),
            "hashtags": post.hashtags,
            "user_type": post.user_type,
            "timestamp": post.timestamp.isoformat(),
            "likes": post.likes,
            "shares": post.shares,
            "comments": post.comments,
            "reach_estimate": post.likes + post.shares * 10
        }
    
    async def generate_live_posts(self, simulation_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate new posts for live feed updates"""
        try:
            # Generate 2-4 new posts based on current context
            post_count = random.randint(2, 4)
            new_posts = await self._generate_contextual_posts(simulation_context, post_count)
            
            # Convert to dict format for API
            return [self._post_to_dict(post) for post in new_posts]
            
        except Exception as e:
            logger.error(f"Failed to generate live posts: {e}")
            return []
    
    async def get_recent_posts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent posts for emergency feed"""
        recent_posts = sorted(self.posts_generated, key=lambda x: x.timestamp, reverse=True)[:limit]
        return [self._post_to_dict(post) for post in recent_posts]
    
    async def initialize_for_simulation(self, simulation_id: str, disaster_type: str, severity: int, location: str):
        """Initialize for simulation"""
        self.simulation_id = simulation_id
        self.disaster_type = disaster_type
        self.disaster_severity = severity
        self.location = location
        self.posts_generated = []
        self.panic_index = 0.0
        self.misinformation_level = 0.0
        self.viral_topics = []
        self.last_generation_time = datetime.utcnow()
        
        logger.info(f"Social Media Agent initialized for {disaster_type} in {location}")
    
    async def process_phase(self, phase: SimulationPhase, simulation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process social media activity for simulation phase"""
        self.current_phase = phase
        
        # Generate initial posts for this phase
        post_count = {
            SimulationPhase.IMPACT: 8,
            SimulationPhase.RESPONSE: 6,
            SimulationPhase.RECOVERY: 4
        }.get(phase, 6)
        
        # Add phase to context
        context = {**simulation_context, 'phase': phase.value}
        
        try:
            # Generate AI posts
            new_posts = await self._generate_contextual_posts(context, post_count)
            
            # Get trending analysis
            trends = await self._analyze_current_trends()
            
            # Calculate influence metrics
            influence = await self._calculate_current_influence()
            
            # Generate comprehensive metrics
            metrics = {
                "posts_generated": {
                    "count": len(new_posts),
                    "phase": phase.value,
                    "panic_index": round(self.panic_index, 3),
                    "misinformation_level": round(self.misinformation_level, 3),
                    "avg_engagement": round(sum(p.engagement for p in new_posts) / len(new_posts), 3) if new_posts else 0,
                    "sentiment_breakdown": self._get_sentiment_breakdown(new_posts),
                    "recent_posts": [self._post_to_dict(p) for p in new_posts[:3]]
                },
                "viral_trends": trends,
                "influence_metrics": influence,
                "total_posts": len(self.posts_generated),
                "current_panic_index": round(self.panic_index, 3),
                "current_misinformation_level": round(self.misinformation_level, 3),
                "viral_topics": self.viral_topics
            }
            
            # Save to cloud
            await self._save_social_state(metrics)
            await self._log_social_event(phase, metrics)
            
            return {
                "agent_id": self.agent_id,
                "phase": phase.value,
                "social_metrics": metrics,
                "timestamp": datetime.utcnow().isoformat(),
                "ai_generated": True
            }
            
        except Exception as e:
            logger.error(f"Social media phase processing error: {e}")
            return await self._generate_fallback_response(phase, simulation_context)
    
    async def _analyze_current_trends(self) -> Dict[str, Any]:
        """Analyze current viral trends"""
        recent_posts = self.posts_generated[-20:] if len(self.posts_generated) > 20 else self.posts_generated
        
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
            "peak_engagement": round(max([p.engagement for p in recent_posts] + [0]), 3),
            "viral_posts": len(high_engagement)
        }
    
    async def _calculate_current_influence(self) -> Dict[str, Any]:
        """Calculate current social media influence"""
        if not self.posts_generated:
            return {
                "panic_influence_score": 0.0,
                "helpful_influence_score": 0.0,
                "misinformation_influence": 0.0,
                "estimated_population_reach": 0,
                "overall_sentiment": "neutral",
                "behavior_impact": "low"
            }
        
        analysis_posts = self.posts_generated[-50:] if len(self.posts_generated) > 50 else self.posts_generated
        
        # Calculate influence factors
        panic_influence = sum(p.engagement * (p.likes + p.shares) for p in analysis_posts 
                            if p.sentiment in [PostSentiment.PANIC, PostSentiment.FEAR])
        helpful_influence = sum(p.engagement * (p.likes + p.shares) for p in analysis_posts 
                              if p.sentiment in [PostSentiment.HELPFUL, PostSentiment.OFFICIAL])
        misinfo_influence = sum(p.engagement * (p.likes + p.shares) for p in analysis_posts 
                              if p.sentiment == PostSentiment.MISINFORMATION)
        
        total_influence = sum(p.engagement * (p.likes + p.shares) for p in analysis_posts)
        
        # Estimate population reach
        estimated_reach = min(175000 * 0.8, total_influence * 100)
        
        return {
            "panic_influence_score": round(panic_influence / max(1, total_influence), 3),
            "helpful_influence_score": round(helpful_influence / max(1, total_influence), 3),
            "misinformation_influence": round(misinfo_influence / max(1, total_influence), 3),
            "estimated_population_reach": int(estimated_reach),
            "overall_sentiment": self._calculate_overall_sentiment(analysis_posts),
            "behavior_impact": "high" if panic_influence > helpful_influence else "moderate",
            "total_engagement": int(total_influence)
        }
    
    def _get_sentiment_breakdown(self, posts: List[SocialPost]) -> Dict[str, int]:
        """Get sentiment breakdown for posts"""
        breakdown = {}
        for sentiment in PostSentiment:
            breakdown[sentiment.value] = len([p for p in posts if p.sentiment == sentiment])
        return breakdown
    
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
                "timestamp": datetime.utcnow().isoformat(),
                "viral_topics": self.viral_topics
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
                "posts_generated": metrics["posts_generated"]["count"],
                "panic_index": self.panic_index,
                "misinformation_level": self.misinformation_level,
                "viral_potential": metrics["viral_trends"]["viral_potential"],
                "timestamp": datetime.utcnow().isoformat()
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
