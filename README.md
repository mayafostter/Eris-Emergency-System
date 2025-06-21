ERIS (Emergency Response Intelligence System)

AI-Powered Disaster Simulation & Coordination Platform  
Agent Orchestrator with Gemini 2.0 Flash

Overview

ERIS is a sophisticated emergency response intelligence system that leverages 10 specialized AI agents to simulate, coordinate, and optimize disaster response scenarios. Built for emergency preparedness testing, training, and strategic planning.

## Live System Access

| Component | URL | Status |
|-----------|-----|--------|
| **Main Dashboard** | [eris-emergency-system.vercel.app](https://eris-emergency-system.vercel.app/) | ✅ Active |
| **API Backend** | [eris-backend-621360763676.us-central1.run.app](https://eris-backend-621360763676.us-central1.run.app/) | ✅ Active |
| **API Documentation** | [API Docs](https://eris-backend-621360763676.us-central1.run.app/docs) | ✅ Active |
| **Fallback Dashboard** | [System Dashboard](https://eris-backend-621360763676.us-central1.run.app/dashboard) | ✅ Active |
| **GitHub Repository** | [github.com/mayafostter/Eris-Emergency-System](https://github.com/mayafostter/Eris-Emergency-System) | ✅ Active |

## AI Orchestrator Architecture

**ERIS deploys a sophisticated 10-agent orchestrator powered by Google's Gemini 2.0 Flash** for coordinated emergency response simulation.

### Agent Architecture Overview

```
ERIS Orchestrator (Gemini 2.0 Flash)
├── ADK Agents (6) - Google Agent Development Kit
│   ├── Emergency Response Coordinator
│   ├── Public Health Manager
│   ├── Infrastructure Manager
│   ├── Logistics Coordinator
│   ├── Communications Director
│   └── Recovery Coordinator
└── Enhanced Agents (4) - Specialized AI Modules
    ├── Hospital Load Modeler
    ├── Public Behavior Simulator
    ├── Social Media Dynamics
    └── News Coverage Simulator
```

## Core Capabilities

### Multi-Phase Disaster Simulation
- **Impact Phase**: Initial disaster response and damage assessment
- **Response Phase**: Coordinated emergency operations and resource deployment
- **Recovery Phase**: Long-term restoration and lessons learned

### Cross-Agent Coordination
- **Context Sharing**: Real-time simulation state synchronization across all 10 agents
- **Phase-Based Execution**: Orchestrated agent activation based on disaster timeline
- **Dynamic Resource Allocation**: AI-driven resource optimization across response systems

### Real-Time Intelligence
- **Live Metrics Dashboard**: Hospital capacity, panic indices, infrastructure status
- **WebSocket Streaming**: Real-time updates from all agent systems
- **Composite Scoring**: AI-calculated emergency response effectiveness

## Technology Stack

### AI & Orchestration
- **Gemini 2.0 Flash**: Core AI model for agent orchestration and decision making
- **Google ADK**: Agent Development Kit for standardized AI agent creation
- **Vertex AI**: Advanced AI capabilities for content generation and analysis

### Backend Infrastructure
- **FastAPI**: High-performance async API framework
- **Google Cloud Run**: Serverless container deployment
- **Firestore**: Real-time NoSQL database for simulation state
- **BigQuery**: Analytics and metrics data warehouse

### Frontend & Integration
- **React + Vite**: Modern frontend framework with fast development
- **CSS**: Utility-first styling framework
- **WebSocket**: Real-time bidirectional communication
- **Vercel**: Edge-optimized frontend deployment

## Detailed Agent Descriptions

### ADK Agents (Google Agent Development Kit)

#### 1. Emergency Response Coordinator
- **Purpose**: Overall emergency response coordination and first responder management
- **AI Model**: Gemini 2.0 Flash
- **Capabilities**: Resource dispatch, emergency prioritization, multi-agency coordination
- **Key Metrics**: Response time, resource utilization, coordination effectiveness

#### 2. Public Health Manager
- **Purpose**: Health system coordination and medical response management
- **AI Model**: Gemini 2.0 Flash
- **Capabilities**: Medical resource allocation, health advisory generation, disease monitoring
- **Key Metrics**: Hospital capacity, medical supply levels, health advisory reach

#### 3. Infrastructure Manager
- **Purpose**: Critical infrastructure assessment and restoration coordination
- **AI Model**: Gemini 2.0 Flash
- **Capabilities**: Damage assessment, repair prioritization, utility coordination
- **Key Metrics**: Infrastructure damage percentage, restoration progress, service availability

#### 4. Logistics Coordinator
- **Purpose**: Supply chain management and resource distribution
- **AI Model**: Gemini 2.0 Flash
- **Capabilities**: Supply tracking, distribution optimization, vendor coordination
- **Key Metrics**: Supply chain efficiency, distribution coverage, inventory levels

#### 5. Communications Director
- **Purpose**: Public information management and media coordination
- **AI Model**: Gemini 2.0 Flash
- **Capabilities**: Public messaging, media relations, communication strategy
- **Key Metrics**: Message reach, public trust levels, communication effectiveness

#### 6. Recovery Coordinator
- **Purpose**: Long-term recovery planning and community rebuilding
- **AI Model**: Gemini 2.0 Flash
- **Capabilities**: Recovery planning, community engagement, economic restoration
- **Key Metrics**: Recovery progress, community resilience, economic indicators

### Enhanced Agents (Specialized AI Modules)

#### 7. Hospital Load Modeler
- **Purpose**: Real-time hospital capacity and medical resource simulation
- **AI Model**: Gemini 2.0 Flash + Specialized algorithms
- **Capabilities**: Patient surge modeling, resource optimization, capacity forecasting
- **Key Metrics**: Bed occupancy, ICU capacity, medical supply availability, staff utilization

#### 8. Public Behavior Simulator
- **Purpose**: Population behavior modeling during emergency scenarios
- **AI Model**: Gemini 2.0 Flash + Behavioral algorithms
- **Capabilities**: Panic index calculation, evacuation compliance modeling, social dynamics
- **Key Metrics**: Panic level, evacuation rates, compliance with emergency orders

#### 9. Social Media Dynamics
- **Purpose**: Social media sentiment and information spread simulation
- **AI Model**: Gemini 2.0 Flash + NLP processing
- **Capabilities**: Social sentiment analysis, misinformation tracking, viral content identification
- **Key Metrics**: Social media activity, sentiment trends, misinformation levels

#### 10. News Coverage Simulator
- **Purpose**: News media coverage and public information dissemination modeling
- **AI Model**: Gemini 2.0 Flash + Media analysis
- **Capabilities**: News story generation, press briefing simulation, media influence tracking
- **Key Metrics**: Media coverage quality, public trust in official information, news reach

## Quick Start Guide

### Start a Simulation

**Default Demo Scenario: Phuket Flood Emergency**

# Using the live API - Phuket Flood Simulation
curl -X POST "https://eris-backend-621360763676.us-central1.run.app/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "disaster_type": "flood",
    "location": "Phuket, Thailand",
    "severity": 7,
    "duration": 4
  }'

**Alternative scenarios:**

# Earthquake scenario
curl -X POST "https://eris-backend-621360763676.us-central1.run.app/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "disaster_type": "earthquake",
    "location": "San Francisco, CA",
    "severity": 9,
    "duration": 12
  }'
```

### Monitor Real-time Metrics

```javascript
// Connect to WebSocket for live updates
const ws = new WebSocket('wss://eris-backend-621360763676.us-central1.run.app/ws/metrics/{simulation_id}');

ws.onmessage = (event) => {
  const metrics = JSON.parse(event.data);
  console.log('Real-time metrics:', metrics.dashboard_metrics);
  console.log('Orchestrator status:', metrics.orchestrator);
};
```

### Access Dashboard

**Quick Demo Setup:**
1. Visit: [https://eris-emergency-system.vercel.app/](https://eris-emergency-system.vercel.app/)
2. **Default Settings** (pre-configured for demo):
   - **Disaster Type**: Flood
   - **Location**: Phuket, Thailand
   - **Severity**: 7
   - **Duration**: 4 hours
3. Click **"Start Simulation"** to launch the 10-agent orchestrator

**Web Dashboard Features:**
- Real-time agent coordination display
- Live metrics from all 10 agents
- Gemini 2.0 Flash orchestrator status
- Cross-agent context sharing visualization

## System Metrics & Monitoring

### Key Performance Indicators

| Metric | Description | Range | AI Source |
|--------|-------------|-------|-----------|
| **Alert Level** | Overall emergency status | GREEN/YELLOW/RED | Orchestrator |
| **Panic Index** | Population stress level | 0-100 | Public Behavior Agent |
| **Hospital Capacity** | Medical system utilization | 0-100% | Hospital Load Agent |
| **Infrastructure Status** | Critical systems functionality | 0-100% | Infrastructure Agent |
| **Emergency Response** | Response effectiveness score | 0-100 | Emergency Response Agent |
| **Public Trust** | Trust in official communications | 0-100% | Communications Agent |
| **Evacuation Compliance** | Population evacuation adherence | 0-100% | Public Behavior Agent |

### Real-time Data Streams

- **Agent Status**: Live status from all 10 agents
- **Cross-Agent Context**: Shared simulation state
- **Phase Progression**: Impact → Response → Recovery
- **Resource Allocation**: Dynamic resource optimization
- **Public Sentiment**: Social media and news analysis

## Deployment Guide - Prerequisites

# Install required tools
npm install -g vercel
pip install -r requirements.txt

# Authenticate with Google Cloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

### Backend Deployment (Google Cloud Run)

# Set default region to avoid prompts
gcloud config set run/region us-central1

# Deploy the 10-agent orchestrator
gcloud run deploy eris-backend \
  --source . \
  --platform=managed \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --set-env-vars="ORCHESTRATOR_VERSION=0.5.0,TOTAL_AGENTS=10,AI_MODEL=gemini-2.0-flash"


### Frontend Deployment (Vercel)

# Deploy React frontend
cd frontend
echo "VITE_API_BASE_URL=https://your-backend-url" > .env.production
vercel --prod

### Verification

# Test the deployment
curl https://your-backend-url/health
curl https://your-backend-url/system/info
curl https://your-backend-url/agents/health

## API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/simulate` | POST | Start new disaster simulation |
| `/status/{simulation_id}` | GET | Get simulation status |
| `/metrics/{simulation_id}` | GET | Get agent metrics |
| `/metrics/dashboard/{simulation_id}` | GET | Get dashboard metrics |
| `/orchestrator/{simulation_id}` | GET | Get orchestrator status |
| `/orchestrator/{simulation_id}/agents` | GET | Get all agent info |
| `/ws/metrics/{simulation_id}` | WebSocket | Real-time metrics stream |

### Agent Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/agents/health` | GET | Check all agent systems |
| `/system/info` | GET | Get system capabilities |
| `/health` | GET | Overall system health |

### Example API Responses

#### Simulation Status
```json
{
  "simulation_id": "abc123",
  "status": "active",
  "current_phase": "response",
  "orchestrator": {
    "ai_model": "Gemini 2.0 Flash",
    "total_agents": 10,
    "coordination_active": true
  }
}
```

#### Dashboard Metrics
```json
{
  "dashboard_data": {
    "alert_level": "YELLOW",
    "panic_index": 35,
    "hospital_capacity": 78,
    "infrastructure_failures": 2,
    "emergency_response": 87,
    "public_trust": 82
  },
  "orchestrator_info": {
    "ai_model": "Gemini 2.0 Flash",
    "current_phase": "response",
    "total_agents": 10
  }
}
```

## Advanced Features

### AI Orchestration Engine

The ERIS orchestrator uses **Gemini 2.0 Flash** as the core intelligence for:

- **Dynamic Agent Coordination**: Real-time decision making for agent activation and resource allocation
- **Context-Aware Processing**: Understanding disaster scenarios and adapting agent behavior
- **Cross-Agent Communication**: Facilitating intelligent information sharing between specialized agents
- **Predictive Analysis**: Forecasting disaster impacts and response effectiveness

### Cross-Agent Context Sharing

```python
# Example context shared across all agents
simulation_context = {
    'disaster_type': 'earthquake',
    'severity': 7,
    'infrastructure_damage': 45,
    'hospital_capacity_utilization': 85,
    'panic_index': 0.35,
    'evacuation_compliance': 0.78,
    'public_trust_level': 0.82,
    'social_media_activity': 0.9
}
```

### Phase-Based Execution

1. **Impact Phase** (0-6 hours)
   - Initial damage assessment
   - Emergency response activation
   - Public safety measures

2. **Response Phase** (6-48 hours)
   - Coordinated emergency operations
   - Resource deployment
   - Public communication

3. **Recovery Phase** (48+ hours)
   - Infrastructure restoration
   - Community support
   - Long-term planning

## Demo Scenarios

### **Primary Demo: Phuket Flood Emergency**
```bash
curl -X POST "https://eris-backend-621360763676.us-central1.run.app/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "disaster_type": "flood",
    "location": "Phuket, Thailand",
    "severity": 7,
    "duration": 4
  }'
```
**Showcases**: Hospital load modeling, evacuation coordination, social media panic tracking, tourism impact assessment

### Tsunami Response
```bash
curl -X POST "https://eris-backend-621360763676.us-central1.run.app/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "disaster_type": "tsunami",
    "location": "Coastal California",
    "severity": 8,
    "duration": 96
  }'
```

### Wildfire Emergency
```bash
curl -X POST "https://eris-backend-621360763676.us-central1.run.app/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "disaster_type": "wildfire",
    "location": "Los Angeles County",
    "severity": 6,
    "duration": 120
  }'
```

## Project Structure

```
eris/                              # Backend Root
├── api/
│   └── main.py                    # FastAPI Server (v0.5.0)
├── orchestrator/
│   └── orchestrator.py            # 10-Agent Orchestrator
├── agents/
│   ├── base_agent.py              # ADK Agent Factory
│   ├── emergency_response_agent.py
│   ├── public_health_agent.py
│   ├── infrastructure_manager_agent.py
│   ├── logistics_coordinator_agent.py
│   ├── communications_director_agent.py
│   ├── recovery_coordinator_agent.py
│   ├── hospital_load_agent.py
│   ├── public_behavior_agent.py
│   ├── social_media_agent.py
│   └── news_simulation_agent.py
├── services/
│   ├── firestore_service.py
│   ├── bigquery_service.py
│   ├── vertex_ai_service.py
│   └── metrics_collector.py
├── utils/
│   └── time_utils.py
├── config.py
├── requirements.txt
└── fallback_dashboard.html

frontend/                          # React Frontend
├── src/
│   ├── App.jsx                    # Main Dashboard
│   └── components/
├── .env.production
└── package.json
```

## 🏆 Innovation Highlights

1. **10-Agent AI Orchestrator**: Unprecedented coordination of specialized AI agents
2. **Gemini 2.0 Flash Integration**: Cutting-edge AI model for emergency response
3. **Real-time Cross-Agent Context**: Dynamic information sharing across all agents
4. **Hospital Load Modeling**: Specialized healthcare system simulation
5. **Social Media Dynamics**: AI-powered social sentiment and misinformation tracking
6. **Phase-Based Execution**: Realistic disaster timeline progression


### Troubleshooting - Common Issues

1. **CORS Errors**: Ensure frontend URL is in backend CORS configuration
2. **Agent Timeout**: Increase Cloud Run timeout for complex simulations
3. **Memory Issues**: Scale up Cloud Run memory allocation for large simulations
4. **Region Selection**: Always use the same region (us-central1) to maintain consistent URLs

#### Performance Optimization

- **Concurrent Agents**: All 10 agents run concurrently for optimal performance
- **Context Caching**: Simulation context is cached for faster agent access
- **WebSocket Optimization**: Real-time updates use efficient WebSocket connections


## Acknowledgments

- **Google Cloud**: For Gemini 2.0 Flash and cloud infrastructure
- **Google ADK**: For agent development framework
- **Vercel**: For frontend deployment platform
- **FastAPI**: For high-performance API framework
- **React Community**: For modern frontend development tools

---

ERIS Emergency Response Intelligence System 
Powered by Gemini 2.0 Flash + Google ADK
