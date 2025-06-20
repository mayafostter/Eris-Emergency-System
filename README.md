ERIS (Emergency Response Intelligence System)

AI-Powered Disaster Simulation Platform with Multi-Agent Coordination

Overview

ERIS is a sophisticated emergency response intelligence system that leverages 10 specialized AI agents to simulate, coordinate, and optimize disaster response scenarios.
Built for emergency preparedness testing, training, and strategic planning.

Key Features

- 10 AI Agents: 6 Google ADK + 4 Enhanced specialized agents
- Real-time Dashboard: Live metrics with WebSocket streaming
- Multi-Disaster Support: 7+ disaster types with realistic simulations
- Phase-based Execution: Impact → Response → Recovery phases
- Cloud Integration: Firestore, BigQuery, and Vertex AI
- Dual Interface: Professional CLI + Modern Web Dashboard
- Live Coordination: Real-time cross-agent communication

System Architecture
![Diagram](https://github.com/user-attachments/assets/c7531da7-f00e-4292-ac47-6bd1cc612ef2)

AI Agent System

Google ADK Agents (Core Emergency Response)

| Agent | Role | Capabilities |
|-------|------|-------------|
| Emergency Response Coordinator | Central command and control | Resource allocation, priority setting, cross-department coordination |
| Public Health Official | Health crisis management | Medical resource distribution, health advisories, epidemic tracking |
| Infrastructure Manager | Critical systems oversight | Power grids, transportation, communications infrastructure |
| Logistics Coordinator | Supply chain management | Resource routing, inventory management, distribution optimization |
| Communications Director | Information dissemination | Public messaging, media coordination, emergency broadcasts |
| Recovery Coordinator | Long-term restoration | Recovery planning, rebuilding coordination, community support |

Enhanced Specialized Agents

| Agent | Focus Area | Key Metrics |
|-------|------------|-------------|
| Hospital Load Coordinator | Healthcare capacity | ICU utilization, patient flow, medical supply tracking |
| Public Behavior Analyst | Population response | Evacuation compliance, panic levels, crowd dynamics |
| Social Media Monitor | Digital sentiment | Misinformation tracking, public sentiment, viral content analysis |
| News Simulation Agent | Media landscape | Press coverage, public trust, information accuracy |

Quick Start

Prerequisites

- Python 3.9+ with pip
- Node.js 16+ with npm
- Git for cloning the repository

1. Backend Setup

Clone and navigate to backend
git clone <repository-url>
cd eris

Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Start FastAPI backend
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload


2. Frontend Setup

Navigate to frontend (in new terminal)
cd eris-frontend

Install dependencies
npm install

Build and start production server
npm run build
npm run preview


3. CLI Usage

In backend directory with venv activated
cd eris

View system status
python main.py status

List available disasters
python main.py disasters

Start simulation with monitoring
python main.py simulate --type flood --location "Phuket, Thailand" --severity 7 --duration 4 --monitor 30

Monitor existing simulation
python main.py monitor <simulation_id> --duration 60


Access Points

Once running, access ERIS through:

Web Dashboard: http://localhost:4173
API Documentation: http://127.0.0.1:8000/docs
CLI Interface: python main.py --help

Demo Scenarios

Scenario 1: Coastal Flood Emergency

python main.py simulate --type flood --location "Phuket, Thailand" --severity 8 --duration 6 --monitor 30

Showcases: Hospital overflow, evacuation coordination, social media panic tracking

Scenario 2: Urban Earthquake Response

python main.py simulate --type earthquake --location "San Francisco, CA" --severity 7 --duration 12 --monitor 60

Showcases: Infrastructure damage assessment, emergency resource allocation, public communications

Scenario 3: Wildfire Evacuation

python main.py simulate --type wildfire --location "Los Angeles, CA" --severity 6 --duration 8 --monitor 45

Showcases: Mass evacuation logistics, air quality monitoring, news coverage simulation

Key Metrics & Outputs

Real-time Dashboard Metrics
- Alert Level: GREEN → YELLOW → RED → CRITICAL
- Panic Index: Public stress and anxiety levels (0-100%)
- Hospital Capacity: ICU utilization and medical resource strain
- Population Affected: Number of people impacted by disaster
- Infrastructure Failures: Critical system outages and damage
- Emergency Response: Overall response effectiveness (0-100%)

Agent Performance Indicators
- Efficiency Rating: Individual agent performance (85-100%)
- Progress Tracking: Task completion across simulation phases
- Coordination Score: Cross-agent communication effectiveness
- Resource Utilization: Optimal allocation of emergency resources

Enhanced Analytics
- Social Media Sentiment: Real-time public mood analysis
- News Coverage Impact: Media influence on public perception
- Behavioral Patterns: Population response to emergency directives
- Recovery Metrics: Long-term restoration progress indicators

Technology Stack

Backend
- FastAPI: High-performance async web framework
- Google ADK 1.2.1: Advanced AI agent development kit
- Pydantic: Data validation and settings management
- AsyncIO: Concurrent agent coordination
- WebSockets: Real-time data streaming

Frontend
- React 18: Modern UI framework with hooks
- Vite: Lightning-fast build tool and dev server
- Tailwind CSS: Utility-first styling framework
- Lucide React: Professional icon library
- WebSocket Client: Real-time data connection

Cloud & AI
- Google Cloud Firestore: Real-time database
- Google BigQuery: Analytics and data warehousing
- Google Vertex AI: AI content generation
- Google ADK: Agent orchestration platform

Development
- Python 3.9+: Backend runtime
- Node.js 16+: Frontend runtime
- TypeScript: Type-safe frontend development
- ESLint: Code quality enforcement

API Endpoints

Core Simulation
- POST /simulate - Start new disaster simulation
- GET /status/{simulation_id} - Get simulation status
- GET /health - System health check

Metrics & Analytics
- GET /metrics/dashboard/{simulation_id} - Real-time dashboard data
- GET /extended-metrics/{simulation_id} - Enhanced agent metrics
- WS /ws/metrics/{simulation_id} - WebSocket metrics stream

Agent Management
- GET /orchestrator/{simulation_id}/agents - All agent information
- GET /enhanced-agents/{simulation_id} - Enhanced agent details
- GET /agents/health - Agent system status

System Information
- GET /system/info - Comprehensive system capabilities
- GET /ping - Quick connectivity test

Performance

System Requirements
- CPU: 4+ cores recommended for optimal agent coordination
- Memory: 8GB+ RAM for full feature set
- Network: Broadband connection for cloud services
- Storage: 2GB+ free space for logs and data

Benchmarks
- Simulation Startup: <5 seconds
- Agent Response Time: <500ms average
- Dashboard Update Frequency: 2-second intervals
- Concurrent Simulations: Up to 5 recommended

Code Style
- Python: Follow PEP 8, use type hints
- JavaScript: Follow ESLint configuration
- Documentation: Update README for new features

Acknowledgments

- Google Cloud Platform for ADK and cloud services
- FastAPI team for excellent async framework
- React community for robust frontend tools
- Emergency Management Community for domain expertise

Links

- Live Demo: [Demo URL when available]
- API Documentation: http://127.0.0.1:8000/docs
- Technical Blog: [Blog URL when available]
- Presentation Slides: [Slides URL when available]
