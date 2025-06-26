import React from 'react';
import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Play,
  Square,
  AlertTriangle,
  MapPin,
  TrendingUp,
  MessageCircle,
  Cloud,
  Zap,
  Activity,
  Heart,
  Settings,
  BarChart3,
  Radio,
  Wifi,
  WifiOff
} from 'lucide-react';

// API Service with WebSocket support
const apiService = {
  baseURL: import.meta.env.VITE_API_URL || 'https://eris-backend-621360763676.us-central1.run.app',

  async getSystemInfo() {
    const response = await fetch(`${this.baseURL}/system/info`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  },

  async startSimulation(config) {
    const response = await fetch(`${this.baseURL}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        disaster_type: config.disaster_type,
        location: config.location,
        severity: config.severity,
        duration: config.duration || 24
      })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  },

  async getDashboardMetrics(simulationId) {
    const response = await fetch(`${this.baseURL}/metrics/dashboard/${simulationId}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  },

  async getAllAgents(simulationId) {
    const response = await fetch(`${this.baseURL}/orchestrator/${simulationId}/agents`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  },

  async getOrchestratorInfo(simulationId) {
    const response = await fetch(`${this.baseURL}/orchestrator/${simulationId}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  },

  async getLiveFeed(simulationId, limit = 10) {
    const response = await fetch(`${this.baseURL}/live-feed/${simulationId}?limit=${limit}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  },

  // WebSocket URL helper
  getWebSocketURL(simulationId) {
    const wsProtocol = this.baseURL.startsWith('https') ? 'wss' : 'ws';
    const wsBaseURL = this.baseURL.replace(/^https?/, wsProtocol);
    return `${wsBaseURL}/ws/metrics/${simulationId}`;
  }
};

// Disaster Types
const DISASTER_TYPES = [
  { type: "earthquake", name: "Earthquake", severity_scale: { min: 1, max: 9, unit: 'Richter Scale', default: 7 } },
  { type: "hurricane", name: "Hurricane", severity_scale: { min: 1, max: 5, unit: 'Category', default: 3 } },
  { type: "flood", name: "Flood", severity_scale: { min: 1, max: 10, unit: 'Flood Index', default: 7 } },
  { type: "wildfire", name: "Wildfire", severity_scale: { min: 1, max: 10, unit: 'Fire Index', default: 6 } },
  { type: "tsunami", name: "Tsunami", severity_scale: { min: 1, max: 10, unit: 'Wave Height (m)', default: 8 } },
  { type: "volcanic_eruption", name: "Volcanic Eruption", severity_scale: { min: 1, max: 10, unit: 'VEI Scale', default: 5 } },
  { type: "severe_storm", name: "Severe Storm", severity_scale: { min: 1, max: 10, unit: 'Storm Index', default: 6 } },
  { type: "epidemic", name: "Epidemic", severity_scale: { min: 1, max: 10, unit: 'Infection Rate', default: 6 } },
  { type: "pandemic", name: "Pandemic", severity_scale: { min: 1, max: 10, unit: 'Global Impact', default: 7 } },
  { type: "landslide", name: "Landslide", severity_scale: { min: 1, max: 10, unit: 'Risk Level', default: 6 } }
];

// Agent Configuration
const createAgentConfig = () => [
  { id: 'emergency_response_coordinator', name: 'Emergency Response Coordinator', icon: AlertTriangle, color: 'from-red-500 to-orange-500', type: 'adk' },
  { id: 'public_health_manager', name: 'Public Health Manager', icon: Heart, color: 'from-pink-500 to-rose-500', type: 'adk' },
  { id: 'infrastructure_manager', name: 'Infrastructure Manager', icon: Settings, color: 'from-yellow-500 to-amber-500', type: 'adk' },
  { id: 'logistics_coordinator', name: 'Logistics Coordinator', icon: BarChart3, color: 'from-green-500 to-emerald-500', type: 'adk' },
  { id: 'communications_director', name: 'Communications Director', icon: Radio, color: 'from-purple-500 to-violet-500', type: 'adk' },
  { id: 'recovery_coordinator', name: 'Recovery Coordinator', icon: TrendingUp, color: 'from-blue-500 to-cyan-500', type: 'adk' },
  { id: 'hospital_load_modeler', name: 'Hospital Load Modeler', icon: Activity, color: 'from-teal-500 to-cyan-500', type: 'enhanced' },
  { id: 'public_behavior_simulator', name: 'Public Behavior Simulator', icon: MessageCircle, color: 'from-indigo-500 to-blue-500', type: 'enhanced' },
  { id: 'social_media_dynamics', name: 'Social Media Dynamics', icon: Zap, color: 'from-orange-500 to-red-500', type: 'enhanced' },
  { id: 'news_coverage_simulator', name: 'News Coverage Simulator', icon: Cloud, color: 'from-slate-500 to-gray-500', type: 'enhanced' }
];

export default function ERISDashboard() {
  // State management
  const [isRunning, setIsRunning] = useState(false);
  const [currentSimulationId, setCurrentSimulationId] = useState(null);
  const [error, setError] = useState(null);
  const [systemInfo, setSystemInfo] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [orchestratorInfo, setOrchestratorInfo] = useState(null);
  const [emergencyFeed, setEmergencyFeed] = useState([]);
  const [socialFeed, setSocialFeed] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');

  const [simulationForm, setSimulationForm] = useState({
    disaster_type: 'flood',
    location: 'Phuket, Thailand',
    severity: 7,
    duration: 4
  });

  const [agents, setAgents] = useState(createAgentConfig());
  const [metrics, setMetrics] = useState({
    alert_level: 'GREEN',
    panic_index: 0,
    hospital_capacity: 65,
    population_affected: 0,
    infrastructure_failures: 0,
    emergency_response: 95,
    public_trust: 80,
    evacuation_compliance: 75
  });

  // WebSocket management
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const pollingIntervalRef = useRef(null);

  const currentDisaster = DISASTER_TYPES.find(d => d.type === simulationForm.disaster_type);

  // Cleanup function
  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }
  }, []);

  // Initialize system
  useEffect(() => {
    const testConnection = async () => {
      try {
        setError(null);
        const info = await apiService.getSystemInfo();
        setSystemInfo(info);
        setIsConnected(true);
        setConnectionStatus('connected');
      } catch (error) {
        setError(`Connection failed: ${error.message}`);
        setIsConnected(false);
        setConnectionStatus('disconnected');
      }
    };
    testConnection();

    // Cleanup on unmount
    return cleanup;
  }, [cleanup]);

  // WebSocket connection management
  const connectWebSocket = useCallback((simulationId) => {
    if (!simulationId) return;

    try {
      const wsUrl = apiService.getWebSocketURL(simulationId);
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        setConnectionStatus('websocket_connected');
        console.log('WebSocket connected for real-time updates');
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      wsRef.current.onclose = () => {
        setConnectionStatus('connected');
        console.log('WebSocket disconnected - falling back to polling');

        // Attempt to reconnect after delay
        if (isRunning) {
          reconnectTimeoutRef.current = setTimeout(() => {
            connectWebSocket(simulationId);
          }, 5000);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
    }
  }, [isRunning]);

  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback((data) => {
    switch (data.type) {
      case 'initial_state':
      case 'metrics_update':
        if (data.dashboard_metrics) {
          setMetrics(prevMetrics => ({
            ...prevMetrics,
            ...data.dashboard_metrics
          }));
        }
        if (data.orchestrator_info || data.orchestrator) {
          setOrchestratorInfo(data.orchestrator_info || data.orchestrator);
        }
        break;

      case 'social_media_update':
        if (data.new_posts && Array.isArray(data.new_posts)) {
          setSocialFeed(prevFeed => [
            ...data.new_posts.map(post => ({
              id: Date.now() + Math.random(),
              user: `@${post.user_type || 'User'}${Math.floor(Math.random() * 1000)}`,
              time: new Date(post.timestamp).toLocaleTimeString(),
              message: post.content,
              type: post.sentiment === 'panic' ? 'urgent' :
                post.sentiment === 'concern' ? 'warning' : 'info',
              engagement: post.engagement || {}
            })),
            ...prevFeed
          ].slice(0, 20)); // Keep last 20 items
        }
        break;

      case 'emergency_update':
        if (data.updates && Array.isArray(data.updates)) {
          setEmergencyFeed(prevFeed => [
            ...data.updates.map(update => ({
              id: Date.now() + Math.random(),
              user: update.source || '@EmergencySystem',
              time: new Date(update.timestamp).toLocaleTimeString(),
              message: update.content,
              type: update.priority === 'high' ? 'urgent' :
                update.priority === 'medium' ? 'warning' : 'info'
            })),
            ...prevFeed
          ].slice(0, 15)); // Keep last 15 items
        }
        break;

      case 'heartbeat':
        // Keep connection alive
        break;

      default:
        console.log('Unknown WebSocket message type:', data.type);
    }
  }, []);

  // Fallback polling when WebSocket is not available
  const startPolling = useCallback((simulationId) => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }

    pollingIntervalRef.current = setInterval(async () => {
      if (connectionStatus !== 'websocket_connected') {
        try {
          // Poll dashboard metrics
          const dashboardData = await apiService.getDashboardMetrics(simulationId);
          setMetrics(prev => ({ ...prev, ...dashboardData.dashboard_data }));

          if (dashboardData.orchestrator_info) {
            setOrchestratorInfo(dashboardData.orchestrator_info);
          }

          // Poll live feed
          try {
            const feedData = await apiService.getLiveFeed(simulationId, 10);
            if (feedData.feed_items && feedData.feed_items.length > 0) {
              setEmergencyFeed(feedData.feed_items.map(item => ({
                id: Date.now() + Math.random(),
                user: item.source || '@EmergencySystem',
                time: new Date(item.timestamp).toLocaleTimeString(),
                message: item.content,
                type: item.type === 'official_update' ? 'urgent' : 'info'
              })));
            }
          } catch (feedError) {
            console.log('Live feed polling failed:', feedError);
          }

        } catch (error) {
          console.error('Polling error:', error);
        }
      }
    }, 5000); // Poll every 5 seconds
  }, [connectionStatus]);

  // Update agent progress simulation
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setAgents(prev => prev.map(agent => ({
        ...agent,
        status: 'active',
        progress: Math.min(100, (agent.progress || 0) + Math.random() * 3),
        efficiency: Math.max(85, Math.min(100, (agent.efficiency || 95) + (Math.random() - 0.5) * 1))
      })));
    }, 3000);

    return () => clearInterval(interval);
  }, [isRunning]);

  const handleStart = async () => {
    if (!isConnected) {
      setError('Backend not connected. Please check your server connection.');
      return;
    }

    try {
      setError(null);
      const response = await apiService.startSimulation(simulationForm);
      setCurrentSimulationId(response.simulation_id);
      setIsRunning(true);

      // Set orchestrator info from response
      if (response.orchestrator_info) {
        setOrchestratorInfo(response.orchestrator_info);
      }

      // Initialize agents
      setAgents(prev => prev.map(agent => ({
        ...agent,
        status: 'active',
        progress: Math.floor(Math.random() * 20) + 10,
        efficiency: Math.floor(Math.random() * 8) + 92
      })));

      // Clear previous feeds
      setEmergencyFeed([]);
      setSocialFeed([]);

      // Connect WebSocket for real-time updates
      connectWebSocket(response.simulation_id);

      // Start polling as fallback
      startPolling(response.simulation_id);

    } catch (error) {
      setError(`Failed to start simulation: ${error.message}`);
    }
  };

  const handleStop = () => {
    setIsRunning(false);
    setCurrentSimulationId(null);
    setError(null);
    setOrchestratorInfo(null);

    // Cleanup connections
    cleanup();

    // Reset agents
    setAgents(createAgentConfig());

    // Reset metrics
    setMetrics({
      alert_level: 'GREEN',
      panic_index: 0,
      hospital_capacity: 65,
      population_affected: 0,
      infrastructure_failures: 0,
      emergency_response: 95,
      public_trust: 80,
      evacuation_compliance: 75
    });

    // Clear feeds
    setEmergencyFeed([]);
    setSocialFeed([]);
  };

  const getAlertColor = (level) => {
    switch (level) {
      case 'CRITICAL':
      case 'RED': return 'bg-red-500/20 text-red-300 border-red-500/50';
      case 'ORANGE': return 'bg-orange-500/20 text-orange-300 border-orange-500/50';
      case 'YELLOW': return 'bg-yellow-500/15 text-yellow-300 border-yellow-500/40';
      case 'GREEN': return 'bg-green-500/15 text-green-300 border-green-500/40';
      default: return 'bg-gray-500/15 text-gray-300 border-gray-500/40';
    }
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'websocket_connected': return 'bg-green-500/10 border-green-500/30 text-green-400';
      case 'connected': return 'bg-blue-500/10 border-blue-500/30 text-blue-400';
      default: return 'bg-red-500/10 border-red-500/30 text-red-400';
    }
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'websocket_connected': return 'Real-time Active';
      case 'connected': return 'Connected';
      default: return 'Disconnected';
    }
  };

  const getConnectionIcon = () => {
    return connectionStatus === 'disconnected' ? WifiOff : Wifi;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      {/* Header */}
      <div className="max-w-7xl mx-auto px-4 py-6">
        <header className="glass-panel">
          <div className="px-8 py-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-6">
                <div>
                  <div className="flex items-center gap-4">
                    <h1 className="text-4xl font-bold bg-gradient-to-r from-white to-cyan-300 bg-clip-text text-transparent">
                      ERIS
                    </h1>
                    {systemInfo && (
                      <span className="px-3 py-1 bg-purple-500/20 text-purple-300 rounded-lg text-sm font-medium border border-purple-500/30">
                        v{systemInfo.eris_version}
                      </span>
                    )}
                  </div>
                  <p className="text-gray-400 mt-1">
                    Emergency Response Intelligence System - {systemInfo?.orchestrator?.ai_model || 'Gemini 2.0 Flash'} Orchestrator
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-4">
                <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border ${getConnectionStatusColor()}`}>
                  {React.createElement(getConnectionIcon(), { className: "w-4 h-4" })}
                  <span className="font-medium">
                    {getConnectionStatusText()}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </header>
      </div>

      <div className="max-w-7xl mx-auto px-4 space-y-6">
        {/* Alert Banner */}
        {isRunning && (
          <div className={`glass-panel p-6 border-2 ${getAlertColor(metrics.alert_level)}`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <AlertTriangle className="w-8 h-8" />
                <div>
                  <h3 className="text-xl font-bold">Alert Level: {metrics.alert_level}</h3>
                  <p className="text-sm opacity-90 mt-1">
                    Active simulation: {currentSimulationId?.slice(0, 8)}... |
                    {connectionStatus === 'websocket_connected' ? ' Real-time updates active' : ' Polling mode active'}
                  </p>
                </div>
              </div>
              <div className="text-right">
                <div className="text-3xl font-bold">{Math.round(metrics.panic_index || 0)}%</div>
                <div className="text-sm opacity-75">Panic Index</div>
              </div>
            </div>
          </div>
        )}

        {/* Configuration Panel */}
        <div className="glass-panel p-8">
          <div className="flex items-center gap-3 mb-6">
            <MapPin className="w-6 h-6 text-blue-400" />
            <h2 className="text-2xl font-bold">Simulation Configuration</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="space-y-3">
              <label className="block text-sm font-medium text-gray-300">Disaster Type</label>
              <select
                value={simulationForm.disaster_type}
                onChange={(e) => setSimulationForm(prev => ({ ...prev, disaster_type: e.target.value }))}
                disabled={isRunning}
                className="w-full px-4 py-3 bg-white/10 rounded-lg text-white border border-white/20 focus:border-cyan-400 focus:outline-none disabled:opacity-50"
              >
                {DISASTER_TYPES.map(disaster => (
                  <option key={disaster.type} value={disaster.type} className="bg-slate-800">
                    {disaster.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-3">
              <label className="block text-sm font-medium text-gray-300">Location</label>
              <input
                type="text"
                value={simulationForm.location}
                onChange={(e) => setSimulationForm(prev => ({ ...prev, location: e.target.value }))}
                disabled={isRunning}
                className="w-full px-4 py-3 bg-white/10 rounded-lg text-white border border-white/20 focus:border-cyan-400 focus:outline-none disabled:opacity-50"
                placeholder="e.g., Phuket, Thailand"
              />
            </div>

            <div className="space-y-3">
              <label className="block text-sm font-medium text-gray-300">
                Severity: {simulationForm.severity} {currentDisaster?.severity_scale.unit}
              </label>
              <input
                type="range"
                min={currentDisaster?.severity_scale.min || 1}
                max={currentDisaster?.severity_scale.max || 10}
                value={simulationForm.severity}
                onChange={(e) => setSimulationForm(prev => ({ ...prev, severity: parseInt(e.target.value) }))}
                disabled={isRunning}
                className="w-full h-3 bg-white/20 rounded-lg appearance-none cursor-pointer slider"
              />
            </div>

            <div className="space-y-3">
              <label className="block text-sm font-medium text-gray-300">Duration (hours)</label>
              <input
                type="number"
                min="1"
                max="24"
                value={simulationForm.duration}
                onChange={(e) => setSimulationForm(prev => ({ ...prev, duration: parseInt(e.target.value) }))}
                disabled={isRunning}
                className="w-full px-4 py-3 bg-white/10 rounded-lg text-white border border-white/20 focus:border-cyan-400 focus:outline-none disabled:opacity-50"
              />
            </div>
          </div>

          {/* Control Panel */}
          <div className="flex items-center justify-between mt-8 pt-6 border-t border-white/10">
            <div className="flex items-center gap-4">
              {currentSimulationId && (
                <div className="px-4 py-2 bg-white/10 rounded-lg border border-white/20">
                  <span className="text-white font-mono text-sm">{currentSimulationId.slice(0, 8)}...</span>
                </div>
              )}
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${isRunning ? 'bg-green-400 animate-pulse' : 'bg-gray-400'}`} />
                <span className="text-gray-300 font-medium">{isRunning ? 'Running' : 'Ready'}</span>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {error && (
                <div className="text-red-400 text-sm max-w-xs truncate bg-red-500/10 px-3 py-2 rounded border border-red-500/30" title={error}>
                  {error}
                </div>
              )}

              <button
                onClick={isRunning ? handleStop : handleStart}
                disabled={!isConnected}
                className={`${isRunning ? 'btn-stop' : 'btn-start'} flex items-center gap-3 px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed text-white`}
              >
                {isRunning ? <Square className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                {isRunning ? 'Stop Simulation' : 'Start Simulation'}
              </button>
            </div>
          </div>
        </div>

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Agents Panel */}
          <div className="lg:col-span-2">
            <div className="glass-panel p-8 h-full">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold">AI Agent Network</h2>
                  <p className="text-cyan-300 mt-1">
                    {systemInfo ? `${systemInfo.total_agent_types} agents` : 'Loading...'} •
                    {systemInfo?.agent_system ? ` ${systemInfo.agent_system.adk_agents} ADK + ${systemInfo.agent_system.enhanced_agents} Enhanced` : ' 6 ADK + 4 Enhanced'} •
                    {orchestratorInfo?.ai_model || systemInfo?.orchestrator?.ai_model || 'Gemini 2.0 Flash'}
                  </p>
                </div>
                <div className="px-4 py-2 bg-green-500/10 rounded-lg border border-green-500/30">
                  <span className="text-green-300 font-medium">
                    {agents.filter(a => a.status === 'active').length}/10 Active
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {agents.map((agent) => {
                  const IconComponent = agent.icon;
                  const isActive = isRunning && agent.status === 'active';

                  return (
                    <div key={agent.id} className="bg-white/5 rounded-xl p-4 border border-white/10 hover:border-white/20 transition-all duration-300 metric-card">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-lg transition-all duration-300 ${isActive ? `bg-gradient-to-r ${agent.color} bg-opacity-30` : 'bg-gray-500/20'}`}>
                            <IconComponent className="w-4 h-4 text-white" />
                          </div>
                          <div>
                            <div className="font-semibold text-white text-sm">{agent.name}</div>
                            <div className="text-xs text-gray-400">
                              {agent.type === 'adk' ? 'Google ADK' : 'Enhanced'} • {orchestratorInfo?.ai_model || 'Gemini 2.0 Flash'} • {Math.round(agent.efficiency || 95)}% efficiency
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className={`px-2 py-1 rounded text-xs font-medium ${isActive ? 'bg-green-500/20 text-green-300' : 'bg-gray-500/20 text-gray-400'}`}>
                            {agent.status || 'standby'}
                          </div>
                        </div>
                      </div>

                      {/* Progress Bar */}
                      <div className="mb-2">
                        <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
                          <span>Progress</span>
                          <span>{Math.round(agent.progress || 0)}% complete</span>
                        </div>
                        <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-700 ${isActive ? `bg-gradient-to-r ${agent.color}` : 'bg-gray-600'}`}
                            style={{ width: `${agent.progress || 0}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Metrics & Feed Sidebar */}
          <div className="space-y-6">
            {/* Key Metrics */}
            <div className="glass-panel p-6">
              <h3 className="text-xl font-bold mb-4">Key Metrics</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Hospital Capacity</span>
                  <span className={`font-bold ${metrics.hospital_capacity > 80 ? 'text-red-400' : metrics.hospital_capacity > 60 ? 'text-yellow-400' : 'text-green-400'}`}>
                    {Math.round(metrics.hospital_capacity || 0)}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Population Affected</span>
                  <span className="font-bold text-orange-400">{(metrics.population_affected || 0).toLocaleString()}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Infrastructure Failures</span>
                  <span className="font-bold text-red-400">{metrics.infrastructure_failures || 0}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Emergency Response</span>
                  <span className="font-bold text-green-400">{Math.round(metrics.emergency_response || 0)}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Public Trust</span>
                  <span className={`font-bold ${metrics.public_trust > 70 ? 'text-green-400' : metrics.public_trust > 50 ? 'text-yellow-400' : 'text-red-400'}`}>
                    {Math.round(metrics.public_trust || 0)}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Evacuation Compliance</span>
                  <span className={`font-bold ${metrics.evacuation_compliance > 70 ? 'text-green-400' : metrics.evacuation_compliance > 50 ? 'text-yellow-400' : 'text-red-400'}`}>
                    {Math.round(metrics.evacuation_compliance || 0)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Emergency Feed */}
            <div className="glass-panel p-6">
              <div className="flex items-center gap-3 mb-4">
                <AlertTriangle className="w-5 h-5 text-red-400" />
                <h3 className="text-lg font-bold">Emergency Feed</h3>
              </div>
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {emergencyFeed.length === 0 ? (
                  <div className="text-gray-500 text-sm text-center py-4">
                    {isRunning ? 'Waiting for emergency updates...' : 'No emergency alerts'}
                  </div>
                ) : (
                  emergencyFeed.map((item) => (
                    <div key={item.id} className={`p-3 rounded-lg border-l-4 ${item.type === 'urgent' ? 'bg-red-500/10 border-red-500' :
                      item.type === 'warning' ? 'bg-yellow-500/10 border-yellow-500' :
                        'bg-blue-500/10 border-blue-500'
                      }`}>
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-sm">{item.user}</span>
                        <span className="text-xs text-gray-400">{item.time}</span>
                      </div>
                      <p className="text-sm text-gray-300">{item.message}</p>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Social Media Feed */}
            <div className="glass-panel p-6">
              <div className="flex items-center gap-3 mb-4">
                <MessageCircle className="w-5 h-5 text-blue-400" />
                <h3 className="text-lg font-bold">Social Media Feed</h3>
              </div>
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {socialFeed.length === 0 ? (
                  <div className="text-gray-500 text-sm text-center py-4">
                    {isRunning ? 'Monitoring social media...' : 'No social activity'}
                  </div>
                ) : (
                  socialFeed.map((post) => (
                    <div key={post.id} className={`p-3 rounded-lg border-l-4 ${post.type === 'urgent' ? 'bg-red-500/10 border-red-500' :
                      post.type === 'warning' ? 'bg-yellow-500/10 border-yellow-500' :
                        'bg-purple-500/10 border-purple-500'
                      }`}>
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-sm">{post.user}</span>
                        <span className="text-xs text-gray-400">{post.time}</span>
                      </div>
                      <p className="text-sm text-gray-300">{post.message}</p>
                      {post.engagement && (
                        <div className="flex items-center gap-3 mt-2 text-xs text-gray-500">
                          {post.engagement.likes && <span>❤️ {post.engagement.likes}</span>}
                          {post.engagement.shares && <span>🔄 {post.engagement.shares}</span>}
                          {post.engagement.comments && <span>💬 {post.engagement.comments}</span>}
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
