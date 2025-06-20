import { useState, useEffect } from 'react';
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

// API Service
const apiService = {
  baseURL: 'http://127.0.0.1:8000',

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
  }
};

// Disaster Types
const DISASTER_TYPES = [
  { type: "earthquake", name: "Earthquake", severity_scale: { min: 1, max: 9, unit: 'Richter Scale', default: 7 } },
  { type: "hurricane", name: "Hurricane", severity_scale: { min: 1, max: 5, unit: 'Category', default: 3 } },
  { type: "flood", name: "Flood", severity_scale: { min: 1, max: 10, unit: 'Flood Index', default: 7 } },
  { type: "wildfire", name: "Wildfire", severity_scale: { min: 1, max: 10, unit: 'Fire Index', default: 6 } },
  { type: "tsunami", name: "Tsunami", severity_scale: { min: 1, max: 10, unit: 'Wave Height (m)', default: 8 } }
];

// Agent Configuration
const createAgentConfig = () => [
  { id: 'emergency_response_coordinator', name: 'Emergency Response Coordinator', icon: AlertTriangle, color: 'from-red-500 to-orange-500', type: 'adk' },
  { id: 'public_health_official', name: 'Public Health Official', icon: Heart, color: 'from-pink-500 to-rose-500', type: 'adk' },
  { id: 'infrastructure_manager', name: 'Infrastructure Manager', icon: Settings, color: 'from-yellow-500 to-amber-500', type: 'adk' },
  { id: 'logistics_coordinator', name: 'Logistics Coordinator', icon: BarChart3, color: 'from-green-500 to-emerald-500', type: 'adk' },
  { id: 'communications_director', name: 'Communications Director', icon: Radio, color: 'from-purple-500 to-violet-500', type: 'adk' },
  { id: 'recovery_coordinator', name: 'Recovery Coordinator', icon: TrendingUp, color: 'from-blue-500 to-cyan-500', type: 'adk' },
  { id: 'hospital_load_coordinator', name: 'Hospital Load Coordinator', icon: Activity, color: 'from-teal-500 to-cyan-500', type: 'enhanced' },
  { id: 'public_behavior_analyst', name: 'Public Behavior Analyst', icon: MessageCircle, color: 'from-indigo-500 to-blue-500', type: 'enhanced' },
  { id: 'social_media_monitor', name: 'Social Media Monitor', icon: Zap, color: 'from-orange-500 to-red-500', type: 'enhanced' },
  { id: 'news_simulation_agent', name: 'News Simulation Agent', icon: Cloud, color: 'from-slate-500 to-gray-500', type: 'enhanced' }
];

export default function ERISDashboard() {
  // State management
  const [isRunning, setIsRunning] = useState(false);
  const [currentSimulationId, setCurrentSimulationId] = useState(null);
  const [error, setError] = useState(null);
  const [systemInfo, setSystemInfo] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  const [simulationForm, setSimulationForm] = useState({
    disaster_type: 'flood',
    location: 'Phuket, Thailand',
    severity: 7,
    duration: 4
  });

  const [agents, setAgents] = useState(createAgentConfig());
  const [metrics, setMetrics] = useState({
    alert_level: 'GREEN',
    panic_index: 15,
    hospital_capacity: 85,
    population_affected: 15000,
    infrastructure_failures: 3,
    emergency_response: 94
  });

  const currentDisaster = DISASTER_TYPES.find(d => d.type === simulationForm.disaster_type);

  // Initialize system
  useEffect(() => {
    const testConnection = async () => {
      try {
        setError(null);
        const info = await apiService.getSystemInfo();
        setSystemInfo(info);
        setIsConnected(true);
      } catch (error) {
        setError(`Connection failed: ${error.message}`);
        setIsConnected(false);
      }
    };
    testConnection();
  }, []);

  // Real-time polling
  useEffect(() => {
    if (!currentSimulationId || !isRunning) return;

    const pollInterval = setInterval(async () => {
      try {
        const dashboardData = await apiService.getDashboardMetrics(currentSimulationId);
        setMetrics(prev => ({ ...prev, ...dashboardData }));

        setAgents(prev => prev.map(agent => ({
          ...agent,
          status: 'active',
          progress: Math.min(100, agent.progress + Math.random() * 5),
          efficiency: Math.max(85, Math.min(100, agent.efficiency + (Math.random() - 0.5) * 2))
        })));
      } catch (error) {
        console.error('Polling error:', error);
      }
    }, 3000);

    return () => clearInterval(pollInterval);
  }, [currentSimulationId, isRunning]);

  const handleStart = async () => {
    if (!isConnected) {
      setError('Backend not connected. Please check your FastAPI server.');
      return;
    }

    try {
      setError(null);
      const response = await apiService.startSimulation(simulationForm);
      setCurrentSimulationId(response.simulation_id);
      setIsRunning(true);

      setAgents(prev => prev.map(agent => ({
        ...agent,
        status: 'active',
        progress: Math.floor(Math.random() * 20) + 10,
        efficiency: Math.floor(Math.random() * 8) + 92
      })));
    } catch (error) {
      setError(`Failed to start simulation: ${error.message}`);
    }
  };

  const handleStop = () => {
    setIsRunning(false);
    setCurrentSimulationId(null);
    setError(null);
    setAgents(createAgentConfig());
  };

  const getAlertColor = (level) => {
    switch (level) {
      case 'CRITICAL': return 'bg-red-500/20 text-red-300 border-red-500/50';
      case 'RED': return 'bg-red-500/15 text-red-300 border-red-500/40';
      case 'YELLOW': return 'bg-yellow-500/15 text-yellow-300 border-yellow-500/40';
      case 'GREEN': return 'bg-green-500/15 text-green-300 border-green-500/40';
      default: return 'bg-gray-500/15 text-gray-300 border-gray-500/40';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      <style jsx>{`
        .glass-panel {
          background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
          backdrop-filter: blur(16px);
          border-radius: 20px;
          border: 1px solid rgba(255, 255, 255, 0.15);
          box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.3);
        }
        .metric-card {
          transition: all 0.3s ease;
        }
        .metric-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4);
        }
      `}</style>

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
                  <p className="text-gray-400 mt-1">Emergency Response Intelligence System</p>
                </div>
              </div>

              <div className="flex items-center gap-4">
                <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border ${isConnected ? 'bg-green-500/10 border-green-500/30' : 'bg-red-500/10 border-red-500/30'
                  }`}>
                  {isConnected ? <Wifi className="w-4 h-4 text-green-400" /> : <WifiOff className="w-4 h-4 text-red-400" />}
                  <span className={`font-medium ${isConnected ? 'text-green-300' : 'text-red-300'}`}>
                    {isConnected ? 'Connected' : 'Disconnected'}
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
                  <p className="text-sm opacity-90 mt-1">Active simulation: {currentSimulationId?.slice(0, 8)}...</p>
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
              {currentDisaster && (
                <p className="text-xs text-gray-400">{currentDisaster.description}</p>
              )}
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
                className={`btn-${isRunning ? 'stop' : 'start'} flex items-center gap-3 px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed text-white`}
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
                    {systemInfo ? `${systemInfo.total_agent_types} agents` : 'Loading...'} • 6 Google ADK + 4 Enhanced
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
                          <div className={`p-2 rounded-lg transition-all duration-300 ${isActive ? `bg-gradient-to-r ${agent.color} bg-opacity-30` : 'bg-gray-500/20'
                            }`}>
                            <IconComponent className="w-4 h-4 text-white" />
                          </div>
                          <div>
                            <div className="font-semibold text-white text-sm">{agent.name}</div>
                            <div className="text-xs text-gray-400">
                              {agent.type === 'adk' ? '🔷 Google ADK' : '⚡ Enhanced'} • {Math.round(agent.efficiency || 95)}% efficiency
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className={`px-2 py-1 rounded text-xs font-medium ${isActive ? 'bg-green-500/20 text-green-300' : 'bg-gray-500/20 text-gray-400'
                            }`}>
                            {agent.status || 'standby'}
                          </div>
                        </div>
                      </div>

                      {/* Progress Bar - NEW ADDITION */}
                      <div className="mb-2">
                        <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
                          <span>Progress</span>
                          <span>{Math.round(agent.progress || 0)}% complete</span>
                        </div>
                        <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-700 ${isActive ? `bg-gradient-to-r ${agent.color}` : 'bg-gray-600'
                              }`}
                            style={{ width: `${agent.progress || 0}%` }}
                          />
                        </div>
                      </div>

                      {/* Efficiency Bar - NEW ADDITION */}
                      <div>
                        <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
                          <span>Efficiency</span>
                          <span>{Math.round(agent.efficiency || 95)}%</span>
                        </div>
                        <div className="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${isActive ? 'bg-gradient-to-r from-cyan-500 to-blue-500' : 'bg-gray-600'
                              }`}
                            style={{ width: `${agent.efficiency || 95}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* System Status Panel */}
          <div className="space-y-6">
            <div className="glass-panel p-6">
              <div className="flex items-center gap-3 mb-4">
                <Cloud className="w-5 h-5 text-blue-400" />
                <h3 className="text-xl font-bold">System Status</h3>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Backend Connection</span>
                  <span className={`text-sm font-medium ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
                    {isConnected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Active Simulation</span>
                  <span className={`text-sm font-medium ${isRunning ? 'text-green-400' : 'text-gray-400'}`}>
                    {isRunning ? 'Running' : 'Standby'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Agents Online</span>
                  <span className="text-sm font-medium text-green-400">
                    {agents.filter(a => a.status === 'active').length}/10
                  </span>
                </div>
                {systemInfo && (
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">ERIS Version</span>
                    <span className="text-sm font-medium text-cyan-400">
                      v{systemInfo.eris_version}
                    </span>
                  </div>
                )}
              </div>
            </div>
            {/* Live Metrics */}
            <div className="glass-panel p-6">
              <div className="flex items-center gap-3 mb-4">
                <TrendingUp className="w-5 h-5 text-cyan-400" />
                <h3 className="text-xl font-bold">Live Metrics</h3>
              </div>

              <div className="space-y-4">
                {[
                  { label: 'Hospital Capacity', value: metrics.hospital_capacity, color: 'red', unit: '%' },
                  { label: 'Population Affected', value: Math.round(metrics.population_affected / 1000), color: 'yellow', unit: 'K' },
                  { label: 'Emergency Response', value: metrics.emergency_response, color: 'green', unit: '%' },
                  { label: 'Infrastructure Failures', value: metrics.infrastructure_failures, color: 'orange', unit: '' }
                ].map((metric, index) => (
                  <div key={index} className="bg-white/5 rounded-lg p-3 border border-white/10">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-gray-300 text-sm">{metric.label}</span>
                      <span className={`font-bold text-${metric.color}-400`}>
                        {metric.value}{metric.unit}
                      </span>
                    </div>
                    <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full bg-gradient-to-r from-${metric.color}-500 to-${metric.color}-400 rounded-full transition-all duration-700`}
                        style={{ width: `${Math.min(100, typeof metric.value === 'number' ? metric.value : 0)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Emergency Feed */}
            <div className="glass-panel p-6">
              <div className="flex items-center gap-3 mb-4">
                <MessageCircle className="w-5 h-5 text-cyan-400" />
                <h3 className="text-xl font-bold">Emergency Feed</h3>
              </div>

              <div className="space-y-3 max-h-60 overflow-y-auto">
                {[
                  { user: 'EmergencyPhuket', time: '2 min ago', message: `Hospital overflow detected. ICU at ${Math.round(metrics.hospital_capacity)}% capacity.`, type: 'urgent' },
                  { user: 'PublicHealthTH', time: '5 min ago', message: `Public panic index at ${Math.round(metrics.panic_index)}%. Coordinating crowd management.`, type: 'warning' },
                  { user: 'ERISSystem', time: '1 min ago', message: `All ${agents.filter(a => a.status === 'active').length} AI agents operational. Emergency response at ${Math.round(metrics.emergency_response)}%.`, type: 'info' }
                ].map((post, index) => (
                  <div key={index} className={`p-3 rounded-lg border-l-4 ${post.type === 'urgent' ? 'border-red-500 bg-red-500/10' :
                    post.type === 'warning' ? 'border-yellow-500 bg-yellow-500/10' :
                      'border-blue-500 bg-blue-500/10'
                    }`}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-blue-300">@{post.user}</span>
                      <span className="text-xs text-gray-400">{post.time}</span>
                    </div>
                    <p className="text-sm text-white leading-relaxed">{post.message}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Debug Panel */}
      <div className="fixed bottom-4 right-4 glass-panel p-4 max-w-xs">
        <div className="text-xs text-gray-400 mb-2">Debug Info</div>
        <div className="text-xs text-white space-y-1">
          <div>Connected: {isConnected ? '✅' : '❌'}</div>
          <div>Running: {isRunning ? '🟢' : '⭕'}</div>
          <div>Agents: {agents.filter(a => a.status === 'active').length}/10</div>
          {currentSimulationId && (
            <div>ID: {currentSimulationId.slice(0, 8)}...</div>
          )}
        </div>
      </div>
    </div>
  );
}
