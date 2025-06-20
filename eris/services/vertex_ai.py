"""
Vertex AI service integration for ERIS disaster simulation platform.
Handles AI-powered content generation for social media, news, and official statements.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

try:
    from google.cloud import aiplatform
    from vertexai.preview.generative_models import GenerativeModel
    import vertexai
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False

import os

logger = logging.getLogger(__name__)

# Simple settings for this service
class Settings:
    def __init__(self):
        self.GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "eris-simulation-project")
        self.GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

def get_settings():
    return Settings()

class VertexAIService:
    """
    Service for generating AI content using Google Cloud Vertex AI.
    Supports both production and mock modes for development.
    """
    
    def __init__(self, use_mock: bool = False):
        self.settings = get_settings()
        self.use_mock = use_mock or not VERTEX_AI_AVAILABLE
        self.project_id = self.settings.GOOGLE_CLOUD_PROJECT_ID
        self.location = self.settings.GOOGLE_CLOUD_LOCATION
        self.model_name = "gemini-pro"
        
        if not self.use_mock:
            try:
                vertexai.init(project=self.project_id, location=self.location)
                self.model = GenerativeModel(self.model_name)
                logger.info(f"Initialized Vertex AI with project {self.project_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize Vertex AI: {e}. Using mock mode.")
                self.use_mock = True
    
    async def generate_social_media_posts(
        self,
        disaster_context: Dict[str, Any],
        simulation_phase: str,
        num_posts: int = 5,
        platform: str = "twitter"
    ) -> List[Dict[str, Any]]:
        """
        Generate realistic social media posts for disaster simulation.
        
        Args:
            disaster_context: Context about the disaster (type, location, severity)
            simulation_phase: Current phase (initial, response, recovery)
            num_posts: Number of posts to generate
            platform: Social media platform (twitter, facebook, instagram)
            
        Returns:
            List of generated social media posts with metadata
        """
        if self.use_mock:
            return self._generate_mock_social_posts(disaster_context, simulation_phase, num_posts, platform)
        
        try:
            prompt = self._build_social_media_prompt(disaster_context, simulation_phase, platform, num_posts)
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "max_output_tokens": 2048
                }
            )
            
            posts = self._parse_social_media_response(response.text, platform)
            
            logger.info(f"Generated {len(posts)} social media posts for {disaster_context.get('type', 'unknown')} disaster")
            return posts
            
        except Exception as e:
            logger.error(f"Error generating social media posts: {e}")
            return self._generate_mock_social_posts(disaster_context, simulation_phase, num_posts, platform)
    
    async def generate_news_content(
        self,
        disaster_context: Dict[str, Any],
        simulation_phase: str,
        content_type: str = "breaking_news"
    ) -> Dict[str, Any]:
        """
        Generate news content for disaster simulation.
        
        Args:
            disaster_context: Context about the disaster
            simulation_phase: Current simulation phase
            content_type: Type of news content (breaking_news, update, analysis)
            
        Returns:
            Generated news article with metadata
        """
        if self.use_mock:
            return self._generate_mock_news_content(disaster_context, simulation_phase, content_type)
        
        try:
            prompt = self._build_news_content_prompt(disaster_context, simulation_phase, content_type)
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "max_output_tokens": 1024
                }
            )
            
            article = self._parse_news_response(response.text, content_type)
            
            logger.info(f"Generated {content_type} news content for {disaster_context.get('type', 'unknown')} disaster")
            return article
            
        except Exception as e:
            logger.error(f"Error generating news content: {e}")
            return self._generate_mock_news_content(disaster_context, simulation_phase, content_type)
    
    async def generate_official_statements(
        self,
        disaster_context: Dict[str, Any],
        simulation_phase: str,
        agency: str = "emergency_management",
        statement_type: str = "public_advisory"
    ) -> Dict[str, Any]:
        """
        Generate official government/agency statements for disaster simulation.
        
        Args:
            disaster_context: Context about the disaster
            simulation_phase: Current simulation phase
            agency: Issuing agency (emergency_management, mayor, governor, federal)
            statement_type: Type of statement (public_advisory, evacuation_order, press_release)
            
        Returns:
            Generated official statement with metadata
        """
        if self.use_mock:
            return self._generate_mock_official_statement(disaster_context, simulation_phase, agency, statement_type)
        
        try:
            prompt = self._build_official_statement_prompt(disaster_context, simulation_phase, agency, statement_type)
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.5,
                    "top_p": 0.7,
                    "max_output_tokens": 1024
                }
            )
            
            statement = self._parse_official_statement_response(response.text, agency, statement_type)
            
            logger.info(f"Generated {statement_type} for {agency} regarding {disaster_context.get('type', 'unknown')} disaster")
            return statement
            
        except Exception as e:
            logger.error(f"Error generating official statement: {e}")
            return self._generate_mock_official_statement(disaster_context, simulation_phase, agency, statement_type)
    
    def _build_social_media_prompt(self, disaster_context: Dict, phase: str, platform: str, num_posts: int) -> str:
        """Build prompt for social media post generation."""
        disaster_type = disaster_context.get("type", "natural disaster")
        location = disaster_context.get("location", "the affected area")
        severity = disaster_context.get("severity", "moderate")
        
        char_limit = {"twitter": 280, "facebook": 500, "instagram": 300}.get(platform, 280)
        
        return f"""
Generate {num_posts} realistic social media posts for {platform} about a {severity} {disaster_type} in {location} during the {phase} phase.

Context:
- Disaster: {disaster_type}
- Location: {location}
- Severity: {severity}
- Phase: {phase}
- Platform: {platform} (max {char_limit} characters per post)

Include a mix of:
- Eyewitness accounts
- Calls for help or assistance
- Updates on conditions
- Safety information sharing
- Community support messages

Return as JSON array with format:
[{{"content": "post text", "author_type": "resident|official|responder", "timestamp": "relative time", "engagement_level": "high|medium|low"}}]
"""
    
    def _build_news_content_prompt(self, disaster_context: Dict, phase: str, content_type: str) -> str:
        """Build prompt for news content generation."""
        disaster_type = disaster_context.get("type", "natural disaster")
        location = disaster_context.get("location", "the affected area")
        severity = disaster_context.get("severity", "moderate")
        
        return f"""
Generate a {content_type} news article about a {severity} {disaster_type} in {location} during the {phase} phase.

Context:
- Disaster: {disaster_type}
- Location: {location}
- Severity: {severity}
- Phase: {phase}
- Content Type: {content_type}

Include:
- Compelling headline
- Key facts and impact
- Official quotes
- Safety information
- Current status updates

Return as JSON with format:
{{"headline": "news headline", "content": "article body", "byline": "reporter name", "timestamp": "publication time", "urgency": "high|medium|low"}}
"""
    
    def _build_official_statement_prompt(self, disaster_context: Dict, phase: str, agency: str, statement_type: str) -> str:
        """Build prompt for official statement generation."""
        disaster_type = disaster_context.get("type", "natural disaster")
        location = disaster_context.get("location", "the affected area")
        severity = disaster_context.get("severity", "moderate")
        
        return f"""
Generate an official {statement_type} from {agency} about a {severity} {disaster_type} in {location} during the {phase} phase.

Context:
- Disaster: {disaster_type}
- Location: {location}
- Severity: {severity}
- Phase: {phase}
- Agency: {agency}
- Statement Type: {statement_type}

Include:
- Official tone and language
- Clear action items or advisories
- Contact information
- Authority and credibility markers
- Appropriate urgency level

Return as JSON with format:
{{"title": "statement title", "content": "statement body", "agency": "{agency}", "official": "official name", "timestamp": "issue time", "priority": "high|medium|low"}}
"""
    
    def _parse_social_media_response(self, response: str, platform: str) -> List[Dict[str, Any]]:
        """Parse AI response into structured social media posts."""
        try:
            # Try to extract JSON from response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                posts = json.loads(json_str)
                
                # Add metadata
                for post in posts:
                    post.update({
                        "platform": platform,
                        "generated_at": datetime.utcnow().isoformat(),
                        "simulation_generated": True
                    })
                
                return posts
        except Exception as e:
            logger.warning(f"Failed to parse social media response: {e}")
        
        # Fallback to mock data
        return self._generate_mock_social_posts({}, "response", 3, platform)
    
    def _parse_news_response(self, response: str, content_type: str) -> Dict[str, Any]:
        """Parse AI response into structured news content."""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                article = json.loads(json_str)
                
                article.update({
                    "content_type": content_type,
                    "generated_at": datetime.utcnow().isoformat(),
                    "simulation_generated": True
                })
                
                return article
        except Exception as e:
            logger.warning(f"Failed to parse news response: {e}")
        
        return self._generate_mock_news_content({}, "response", content_type)
    
    def _parse_official_statement_response(self, response: str, agency: str, statement_type: str) -> Dict[str, Any]:
        """Parse AI response into structured official statement."""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                statement = json.loads(json_str)
                
                statement.update({
                    "agency": agency,
                    "statement_type": statement_type,
                    "generated_at": datetime.utcnow().isoformat(),
                    "simulation_generated": True
                })
                
                return statement
        except Exception as e:
            logger.warning(f"Failed to parse official statement response: {e}")
        
        return self._generate_mock_official_statement({}, "response", agency, statement_type)
    
    def _generate_mock_social_posts(self, disaster_context: Dict, phase: str, num_posts: int, platform: str) -> List[Dict[str, Any]]:
        """Generate mock social media posts for development."""
        mock_posts = [
            {
                "content": "Power is out in downtown area. Anyone know what's happening? #emergency",
                "author_type": "resident",
                "timestamp": "2 minutes ago",
                "engagement_level": "high",
                "platform": platform,
                "generated_at": datetime.utcnow().isoformat(),
                "simulation_generated": True
            },
            {
                "content": "Emergency services are responding to reports in the area. Please stay safe and follow official guidance.",
                "author_type": "official",
                "timestamp": "5 minutes ago",
                "engagement_level": "medium",
                "platform": platform,
                "generated_at": datetime.utcnow().isoformat(),
                "simulation_generated": True
            },
            {
                "content": "Offering shelter to neighbors who need a safe place. Send me a message if you need help. #community",
                "author_type": "resident",
                "timestamp": "8 minutes ago",
                "engagement_level": "medium",
                "platform": platform,
                "generated_at": datetime.utcnow().isoformat(),
                "simulation_generated": True
            }
        ]
        
        return mock_posts[:num_posts]
    
    def _generate_mock_news_content(self, disaster_context: Dict, phase: str, content_type: str) -> Dict[str, Any]:
        """Generate mock news content for development."""
        return {
            "headline": "Breaking: Emergency Response Underway in Local Area",
            "content": "Emergency services are responding to a developing situation in the downtown area. Residents are advised to stay indoors and monitor official channels for updates. More information will be provided as it becomes available.",
            "byline": "Sarah Johnson, News Reporter",
            "timestamp": datetime.utcnow().isoformat(),
            "urgency": "high",
            "content_type": content_type,
            "generated_at": datetime.utcnow().isoformat(),
            "simulation_generated": True
        }
    
    def _generate_mock_official_statement(self, disaster_context: Dict, phase: str, agency: str, statement_type: str) -> Dict[str, Any]:
        """Generate mock official statement for development."""
        return {
            "title": "Public Safety Advisory",
            "content": "The Emergency Management Office is monitoring the current situation and coordinating with local agencies. Residents are advised to remain calm and follow safety protocols. Emergency services are on scene and responding appropriately.",
            "agency": agency,
            "official": "Director of Emergency Management",
            "timestamp": datetime.utcnow().isoformat(),
            "priority": "high",
            "statement_type": statement_type,
            "generated_at": datetime.utcnow().isoformat(),
            "simulation_generated": True
        }
