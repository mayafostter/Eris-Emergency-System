import React, { useState, useEffect, useRef, useCallback } from 'react';
import { AlertTriangle, Shield, Activity, Users, Building, Radio, Heart, TrendingUp, Clock, Map, Search, BarChart3, AlertCircle, Play, Square, BookOpen, Database, Brain, Target, MessageSquare, MessageCircle, Newspaper, Globe, Filter, Calendar, MapPin, ChevronRight, Eye, Lightbulb, CheckCircle, XCircle, Settings, Zap, Cloud, Wifi, WifiOff } from 'lucide-react';

// Import data and services
import { HISTORICAL_DISASTERS } from '../data/historicalDisasters';
import { CURRENT_NEWS_FEED, SOCIAL_INTELLIGENCE, BEST_PRACTICES } from '../data/threatIntelligence';
import { createAgentConfig, DISASTER_TYPES } from '../data/agentConfig';
import { apiService } from '../services/apiService';

const EmergencyIntelligencePlatform = () => {
    const [activeMode, setActiveMode] = useState('learn'); // learn, monitor, respond
    const [selectedDisaster, setSelectedDisaster] = useState(null);
    const [systemStatus, setSystemStatus] = useState('monitoring');
    const [searchQuery, setSearchQuery] = useState('');
    const [filterType, setFilterType] = useState('all');
    const [currentTime, setCurrentTime] = useState(new Date());
    const [currentPage, setCurrentPage] = useState(1);
    const disastersPerPage = 3;

    // Real-time emergency response state
    const [isEmergencyActive, setIsEmergencyActive] = useState(false);
    const [currentEmergencyId, setCurrentEmergencyId] = useState(null);
    const [error, setError] = useState(null);
    const [systemInfo, setSystemInfo] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const [orchestratorInfo, setOrchestratorInfo] = useState(null);
    const [emergencyFeed, setEmergencyFeed] = useState([]);
    const [socialFeed, setSocialFeed] = useState([]);
    const [connectionStatus, setConnectionStatus] = useState('disconnected');

    // Emergency response configuration
    const [emergencyConfig, setEmergencyConfig] = useState({
        disaster_type: 'flood',
        location: 'Phuket, Thailand',
        severity: 7,
        duration: 4
    });

    // Intelligence agents and metrics
    const [agents, setAgents] = useState(createAgentConfig());
    const [emergencyMetrics, setEmergencyMetrics] = useState({
        alert_level: 'GREEN',
        threat_index: 0,
        hospital_capacity: 65,
        population_at_risk: 0,
        infrastructure_status: 95,
        response_readiness: 95,
        public_confidence: 80,
        evacuation_readiness: 75
    });

    // WebSocket management
    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);
    const pollingIntervalRef = useRef(null);

    const currentDisaster = DISASTER_TYPES.find(d => d.type === emergencyConfig.disaster_type);

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
        const timer = setInterval(() => setCurrentTime(new Date()), 1000);
        return () => clearInterval(timer);
    }, []);

    useEffect(() => {
        const testConnection = async () => {
            try {
                setError(null);
                const info = await apiService.getSystemInfo();
                setSystemInfo(info);
                setIsConnected(true);
                setConnectionStatus('connected');
            } catch (error) {
                setError(`Intelligence system offline: ${error.message}`);
                setIsConnected(false);
                setConnectionStatus('disconnected');
            }
        };
        testConnection();

        return cleanup;
    }, [cleanup]);

    // Pagination and filtering
    const filteredDisasters = HISTORICAL_DISASTERS.filter(disaster => {
        const matchesSearch = disaster.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            disaster.location.toLowerCase().includes(searchQuery.toLowerCase());
        const matchesFilter = filterType === 'all' ||
            disaster.type.toLowerCase().replace(/\s+/g, '_') === filterType.toLowerCase() ||
            disaster.type.toLowerCase() === filterType.toLowerCase();
        return matchesSearch && matchesFilter;
    });

    // Pagination calculations
    const totalPages = Math.ceil(filteredDisasters.length / disastersPerPage);
    const startIndex = (currentPage - 1) * disastersPerPage;
    const paginatedDisasters = filteredDisasters.slice(startIndex, startIndex + disastersPerPage);

    // Reset to page 1 when filters change
    useEffect(() => {
        setCurrentPage(1);
    }, [searchQuery, filterType]);

    // WebSocket connection management
    const connectWebSocket = useCallback((emergencyId) => {
        if (!emergencyId) return;

        try {
            const wsUrl = apiService.getWebSocketURL(emergencyId);
            wsRef.current = new WebSocket(wsUrl);

            wsRef.current.onopen = () => {
                setConnectionStatus('real_time_active');
                console.log('Real-time emergency intelligence activated');
            };

            wsRef.current.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    handleIntelligenceUpdate(data);
                } catch (error) {
                    console.error('Error processing intelligence update:', error);
                }
            };

            wsRef.current.onclose = () => {
                setConnectionStatus('connected');
                console.log('Real-time intelligence disconnected - using polling');

                if (isEmergencyActive) {
                    reconnectTimeoutRef.current = setTimeout(() => {
                        connectWebSocket(emergencyId);
                    }, 5000);
                }
            };

            wsRef.current.onerror = (error) => {
                console.error('Intelligence system error:', error);
            };

        } catch (error) {
            console.error('Failed to connect real-time intelligence:', error);
        }
    }, [isEmergencyActive]);

    // Handle real-time intelligence updates
    const handleIntelligenceUpdate = useCallback((data) => {
        switch (data.type) {
            case 'initial_state':
            case 'metrics_update':
                if (data.dashboard_metrics) {
                    setEmergencyMetrics(prevMetrics => ({
                        alert_level: data.dashboard_metrics.alert_level || prevMetrics.alert_level,
                        threat_index: data.dashboard_metrics.panic_index || prevMetrics.threat_index,
                        hospital_capacity: data.dashboard_metrics.hospital_capacity || prevMetrics.hospital_capacity,
                        population_at_risk: data.dashboard_metrics.population_affected || prevMetrics.population_at_risk,
                        infrastructure_status: 100 - (data.dashboard_metrics.infrastructure_failures * 10) || prevMetrics.infrastructure_status,
                        response_readiness: data.dashboard_metrics.emergency_response || prevMetrics.response_readiness,
                        public_confidence: data.dashboard_metrics.public_trust || prevMetrics.public_confidence,
                        evacuation_readiness: data.dashboard_metrics.evacuation_compliance || prevMetrics.evacuation_readiness
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
                            user: `@${post.user_type || 'Citizen'}${Math.floor(Math.random() * 1000)}`,
                            time: new Date(post.timestamp).toLocaleTimeString(),
                            message: post.content,
                            type: post.sentiment === 'panic' ? 'urgent' :
                                post.sentiment === 'concern' ? 'warning' : 'info',
                            engagement: post.engagement || {}
                        })),
                        ...prevFeed
                    ].slice(0, 20));
                }
                break;

            case 'emergency_update':
                if (data.updates && Array.isArray(data.updates)) {
                    setEmergencyFeed(prevFeed => [
                        ...data.updates.map(update => ({
                            id: Date.now() + Math.random(),
                            user: update.source || '@EmergencyHQ',
                            time: new Date(update.timestamp).toLocaleTimeString(),
                            message: update.content,
                            type: update.priority === 'high' ? 'urgent' :
                                update.priority === 'medium' ? 'warning' : 'info'
                        })),
                        ...prevFeed
                    ].slice(0, 15));
                }
                break;
        }
    }, []);

    // Fallback polling for intelligence updates
    const startIntelligencePolling = useCallback((emergencyId) => {
        if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
        }

        pollingIntervalRef.current = setInterval(async () => {
            if (connectionStatus !== 'real_time_active') {
                try {
                    const dashboardData = await apiService.getDashboardMetrics(emergencyId);
                    setEmergencyMetrics(prev => ({
                        alert_level: dashboardData.dashboard_data?.alert_level || prev.alert_level,
                        threat_index: dashboardData.dashboard_data?.panic_index || prev.threat_index,
                        hospital_capacity: dashboardData.dashboard_data?.hospital_capacity || prev.hospital_capacity,
                        population_at_risk: dashboardData.dashboard_data?.population_affected || prev.population_at_risk,
                        infrastructure_status: 100 - (dashboardData.dashboard_data?.infrastructure_failures * 10) || prev.infrastructure_status,
                        response_readiness: dashboardData.dashboard_data?.emergency_response || prev.response_readiness,
                        public_confidence: dashboardData.dashboard_data?.public_trust || prev.public_confidence,
                        evacuation_readiness: dashboardData.dashboard_data?.evacuation_compliance || prev.evacuation_readiness
                    }));

                    if (dashboardData.orchestrator_info) {
                        setOrchestratorInfo(dashboardData.orchestrator_info);
                    }

                    try {
                        const feedData = await apiService.getLiveFeed(emergencyId, 10);
                        if (feedData.feed_items && feedData.feed_items.length > 0) {
                            setEmergencyFeed(feedData.feed_items.map(item => ({
                                id: Date.now() + Math.random(),
                                user: item.source || '@EmergencyHQ',
                                time: new Date(item.timestamp).toLocaleTimeString(),
                                message: item.content,
                                type: item.type === 'official_update' ? 'urgent' : 'info'
                            })));
                        }
                    } catch (feedError) {
                        console.log('Emergency feed polling failed:', feedError);
                    }

                } catch (error) {
                    console.error('Intelligence polling error:', error);
                }
            }
        }, 5000);
    }, [connectionStatus]);

    // Update agent progress for active emergency response
    useEffect(() => {
        if (!isEmergencyActive) return;

        const interval = setInterval(() => {
            setAgents(prev => prev.map(agent => ({
                ...agent,
                status: 'analyzing',
                progress: Math.min(100, (agent.progress || 0) + Math.random() * 3),
                efficiency: Math.max(85, Math.min(100, (agent.efficiency || 95) + (Math.random() - 0.5) * 1))
            })));
        }, 3000);

        return () => clearInterval(interval);
    }, [isEmergencyActive]);

    const handleStudyDisaster = (disaster) => {
        setSelectedDisaster(disaster);
        setActiveMode('learn');
    };

    const handleActivateIntelligence = async (location, threatType) => {
        try {
            const response = await apiService.startSimulation({
                disaster_type: threatType.toLowerCase(),
                location: location,
                severity: 5,
                duration: 24
            });

            setCurrentEmergencyId(response.simulation_id);
            setIsEmergencyActive(true);
            setSystemStatus('intelligence_active');
            setActiveMode('respond');

            // Initialize intelligence agents
            setAgents(prev => prev.map(agent => ({
                ...agent,
                status: 'analyzing',
                progress: Math.floor(Math.random() * 20) + 10,
                efficiency: Math.floor(Math.random() * 8) + 92
            })));

            // Clear previous intelligence feeds
            setEmergencyFeed([]);
            setSocialFeed([]);

            // Activate real-time intelligence
            connectWebSocket(response.simulation_id);
            startIntelligencePolling(response.simulation_id);

        } catch (error) {
            setError('Failed to activate emergency intelligence system');
        }
    };

    const handleActivateEmergencyResponse = async () => {
        if (!isConnected) {
            setError('Intelligence system offline. Please check system connection.');
            return;
        }

        try {
            setError(null);
            const response = await apiService.startSimulation(emergencyConfig);
            setCurrentEmergencyId(response.simulation_id);
            setIsEmergencyActive(true);

            if (response.orchestrator_info) {
                setOrchestratorInfo(response.orchestrator_info);
            }

            // Initialize intelligence agents
            setAgents(prev => prev.map(agent => ({
                ...agent,
                status: 'analyzing',
                progress: Math.floor(Math.random() * 20) + 10,
                efficiency: Math.floor(Math.random() * 8) + 92
            })));

            setEmergencyFeed([]);
            setSocialFeed([]);

            connectWebSocket(response.simulation_id);
            startIntelligencePolling(response.simulation_id);

        } catch (error) {
            setError(`Failed to activate emergency response: ${error.message}`);
        }
    };

    const handleStopEmergencyResponse = () => {
        setIsEmergencyActive(false);
        setCurrentEmergencyId(null);
        setError(null);
        setOrchestratorInfo(null);

        cleanup();

        setAgents(createAgentConfig());

        setEmergencyMetrics({
            alert_level: 'GREEN',
            threat_index: 0,
            hospital_capacity: 65,
            population_at_risk: 0,
            infrastructure_status: 95,
            response_readiness: 95,
            public_confidence: 80,
            evacuation_readiness: 75
        });

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
            case 'real_time_active': return 'bg-green-500/10 border-green-500/30 text-green-400';
            case 'connected': return 'bg-blue-500/10 border-blue-500/30 text-blue-400';
            default: return 'bg-red-500/10 border-red-500/30 text-red-400';
        }
    };

    const getConnectionStatusText = () => {
        switch (connectionStatus) {
            case 'real_time_active': return 'Real-time Intelligence Active';
            case 'connected': return 'Intelligence System Online';
            default: return 'System Offline';
        }
    };

    const getConnectionIcon = () => {
        return connectionStatus === 'disconnected' ? WifiOff : Wifi;
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
            {/* Main Header */}
            <div className="border-b border-slate-700 bg-slate-800/50 backdrop-blur">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-6">
                            <div>
                                <h1 className="text-2xl font-bold text-white">ERIS</h1>
                                <p className="text-sm text-slate-300">Emergency Response Intelligence System</p>
                            </div>

                            {/* Mode Switcher */}
                            <div className="flex bg-slate-700 rounded-lg p-1">
                                <button
                                    onClick={() => setActiveMode('learn')}
                                    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${activeMode === 'learn' ? 'bg-blue-600 text-white' : 'text-slate-300 hover:text-white'
                                        }`}
                                >
                                    <BookOpen className="w-4 h-4 inline-block mr-2" />
                                    Learn from History
                                </button>
                                <button
                                    onClick={() => setActiveMode('monitor')}
                                    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${activeMode === 'monitor' ? 'bg-orange-600 text-white' : 'text-slate-300 hover:text-white'
                                        }`}
                                >
                                    <Eye className="w-4 h-4 inline-block mr-2" />
                                    Monitor Threats
                                </button>
                                <button
                                    onClick={() => setActiveMode('respond')}
                                    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${activeMode === 'respond' ? 'bg-red-600 text-white' : 'text-slate-300 hover:text-white'
                                        }`}
                                >
                                    <Target className="w-4 h-4 inline-block mr-2" />
                                    Emergency Intelligence
                                </button>
                            </div>
                        </div>

                        <div className="flex items-center space-x-4">
                            <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border ${getConnectionStatusColor()}`}>
                                {React.createElement(getConnectionIcon(), { className: "w-4 h-4" })}
                                <span className="font-medium">
                                    {getConnectionStatusText()}
                                </span>
                            </div>
                            <div className="text-sm text-slate-400">
                                {currentTime.toLocaleString()}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="max-w-7xl mx-auto px-6 py-6">
                {/* Active Emergency Alert Banner */}
                {isEmergencyActive && (
                    <div className={`mb-6 p-6 rounded-lg border-2 ${getAlertColor(emergencyMetrics.alert_level)}`}>
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-4">
                                <AlertTriangle className="w-8 h-8" />
                                <div>
                                    <h3 className="text-xl font-bold">Emergency Intelligence Active: {emergencyMetrics.alert_level}</h3>
                                    <p className="text-sm opacity-90 mt-1">
                                        Live emergency response for {emergencyConfig.disaster_type.replace('_', ' ')} in {emergencyConfig.location} |
                                        Intelligence ID: {currentEmergencyId?.slice(0, 8)}... |
                                        {connectionStatus === 'real_time_active' ? ' Real-time intelligence streaming' : ' Standard monitoring active'}
                                    </p>
                                </div>
                            </div>
                            <div className="text-right">
                                <div className="text-3xl font-bold">{Math.round(emergencyMetrics.threat_index || 0)}%</div>
                                <div className="text-sm opacity-75">Threat Index</div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Learn Mode - Historical Disaster Analysis */}
                {activeMode === 'learn' && (
                    <div className="space-y-6">
                        <div className="text-center mb-8">
                            <h2 className="text-3xl font-bold text-white mb-2">Learn from Every Disaster in History</h2>
                            <p className="text-slate-400 text-lg">Study real disasters, understand what worked, and prepare better responses</p>
                        </div>

                        {/* Search and Filter */}
                        <div className="bg-slate-800 rounded-lg p-6">
                            <div className="flex items-center space-x-4 mb-4">
                                <div className="flex-1 relative">
                                    <Search className="w-5 h-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" />
                                    <input
                                        type="text"
                                        placeholder="Search disasters by name, location, or type..."
                                        value={searchQuery}
                                        onChange={(e) => setSearchQuery(e.target.value)}
                                        className="w-full pl-10 pr-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white focus:border-blue-400 focus:outline-none"
                                    />
                                </div>
                                <select
                                    value={filterType}
                                    onChange={(e) => setFilterType(e.target.value)}
                                    className="px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white focus:border-blue-400 focus:outline-none"
                                >
                                    <option value="all">All Types</option>
                                    <option value="severe_storm">Severe Storm</option>
                                    <option value="earthquake">Earthquake</option>
                                    <option value="hurricane">Hurricane</option>
                                    <option value="flood">Flood</option>
                                    <option value="wildfire">Wildfire</option>
                                    <option value="tsunami">Tsunami</option>
                                    <option value="landslide">Landslide</option>
                                    <option value="volcanic_eruption">Volcanic Eruption</option>
                                    <option value="epidemic">Epidemic</option>
                                    <option value="pandemic">Pandemic</option>
                                </select>
                            </div>
                        </div>

                        {/* Historical Disasters Grid */}
                        {!selectedDisaster ? (
                            <div className="space-y-6">
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                                    {paginatedDisasters.map((disaster) => (
                                        <div key={disaster.id} className="bg-slate-800 rounded-lg p-6 hover:bg-slate-700 transition-colors cursor-pointer border border-slate-700" onClick={() => handleStudyDisaster(disaster)}>
                                            <div className="flex items-start justify-between mb-4">
                                                <div>
                                                    <h3 className="text-lg font-semibold text-white mb-1">{disaster.name}</h3>
                                                    <p className="text-slate-400 text-sm">{disaster.location} • {disaster.year}</p>
                                                </div>
                                                <span className="px-2 py-1 bg-blue-500/20 text-blue-400 text-xs rounded-full">
                                                    {disaster.type}
                                                </span>
                                            </div>

                                            <div className="space-y-2 mb-4">
                                                <div className="flex justify-between text-sm">
                                                    <span className="text-slate-400">Casualties:</span>
                                                    <span className="text-red-400 font-medium">{disaster.casualties.toLocaleString()}</span>
                                                </div>
                                                <div className="flex justify-between text-sm">
                                                    <span className="text-slate-400">Economic Damage:</span>
                                                    <span className="text-yellow-400 font-medium">{disaster.damage}</span>
                                                </div>
                                                <div className="flex justify-between text-sm">
                                                    <span className="text-slate-400">Response Duration:</span>
                                                    <span className="text-blue-400 font-medium">{disaster.timeline}</span>
                                                </div>
                                            </div>

                                            <div className="mb-4">
                                                <p className="text-xs text-slate-400 mb-2">Key Lessons Learned:</p>
                                                <ul className="space-y-1">
                                                    {disaster.lessons.slice(0, 2).map((lesson, index) => (
                                                        <li key={index} className="text-xs text-slate-300 flex items-start">
                                                            <Lightbulb className="w-3 h-3 text-yellow-400 mt-0.5 mr-1 flex-shrink-0" />
                                                            {lesson}
                                                        </li>
                                                    ))}
                                                </ul>
                                            </div>

                                            <button className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 rounded-lg text-sm font-medium transition-colors flex items-center justify-center">
                                                <BookOpen className="w-4 h-4 mr-2" />
                                                Study This Disaster
                                            </button>
                                        </div>
                                    ))}
                                </div>

                                {/* Pagination Controls */}
                                {totalPages > 1 && (
                                    <div className="flex items-center justify-between bg-slate-800 rounded-lg p-4">
                                        <div className="text-sm text-slate-400">
                                            Showing {startIndex + 1}-{Math.min(startIndex + disastersPerPage, filteredDisasters.length)} of {filteredDisasters.length} disasters
                                        </div>

                                        <div className="flex items-center space-x-2">
                                            <button
                                                onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                                                disabled={currentPage === 1}
                                                className="px-3 py-1 rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed bg-slate-700 text-white hover:bg-slate-600"
                                            >
                                                <ChevronRight className="w-4 h-4 rotate-180" />
                                            </button>

                                            <div className="flex items-center space-x-1">
                                                {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
                                                    <button
                                                        key={page}
                                                        onClick={() => setCurrentPage(page)}
                                                        className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${currentPage === page
                                                            ? 'bg-blue-600 text-white'
                                                            : 'bg-slate-700 text-slate-300 hover:bg-slate-600 hover:text-white'
                                                            }`}
                                                    >
                                                        {page}
                                                    </button>
                                                ))}
                                            </div>

                                            <button
                                                onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                                                disabled={currentPage === totalPages}
                                                className="px-3 py-1 rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed bg-slate-700 text-white hover:bg-slate-600"
                                            >
                                                <ChevronRight className="w-4 h-4" />
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        ) : (
                            /* Detailed Study View - Now appears in place of the grid */
                            <div className="space-y-6">
                                {/* Back to Grid Button */}
                                <div className="flex items-center justify-between">
                                    <button
                                        onClick={() => setSelectedDisaster(null)}
                                        className="flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors"
                                    >
                                        <ChevronRight className="w-4 h-4 rotate-180" />
                                        Back to Disaster List
                                    </button>
                                    <span className="text-slate-400 text-sm">
                                        {filteredDisasters.length} disasters found • Page {currentPage} of {totalPages}
                                    </span>
                                </div>

                                {/* Study View */}
                                <div className="bg-slate-800 rounded-lg p-6 border border-blue-500">
                                    <div className="flex items-center justify-between mb-6">
                                        <h2 className="text-2xl font-bold text-white">Studying: {selectedDisaster.name}</h2>
                                        <button
                                            onClick={() => setSelectedDisaster(null)}
                                            className="text-slate-400 hover:text-white text-xl font-bold w-8 h-8 flex items-center justify-center rounded-full hover:bg-slate-700 transition-colors"
                                        >
                                            ✕
                                        </button>
                                    </div>

                                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                        <div>
                                            <h3 className="text-lg font-semibold text-white mb-4">What Worked</h3>
                                            <div className="space-y-3">
                                                {selectedDisaster.lessons.map((lesson, index) => (
                                                    <div key={index} className="flex items-start p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                                                        <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 mr-3 flex-shrink-0" />
                                                        <span className="text-green-100">{lesson}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>

                                        <div>
                                            <h3 className="text-lg font-semibold text-white mb-4">Apply to Current Situation</h3>
                                            <div className="space-y-3">
                                                <button
                                                    onClick={() => handleActivateIntelligence(selectedDisaster.location, selectedDisaster.type)}
                                                    className="w-full p-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-left transition-colors"
                                                >
                                                    <div className="flex items-center justify-between">
                                                        <span>Activate intelligence monitoring for {selectedDisaster.location}</span>
                                                        <ChevronRight className="w-4 h-4" />
                                                    </div>
                                                </button>
                                                <button className="w-full p-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-left transition-colors">
                                                    <div className="flex items-center justify-between">
                                                        <span>Generate evacuation plan based on this case</span>
                                                        <ChevronRight className="w-4 h-4" />
                                                    </div>
                                                </button>
                                                <button className="w-full p-3 bg-orange-600 hover:bg-orange-700 text-white rounded-lg text-left transition-colors">
                                                    <div className="flex items-center justify-between">
                                                        <span>Create training scenario for your team</span>
                                                        <ChevronRight className="w-4 h-4" />
                                                    </div>
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Monitor Mode - Threat Intelligence */}
                {activeMode === 'monitor' && (
                    <div className="space-y-6">
                        <div className="text-center mb-8">
                            <h2 className="text-3xl font-bold text-white mb-2">Global Threat Intelligence</h2>
                            <p className="text-slate-400 text-lg">Monitor emerging threats and activate response protocols</p>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            {/* Real-time News Intelligence */}
                            <div className="lg:col-span-2 bg-slate-800 rounded-lg p-6">
                                <div className="flex items-center space-x-3 mb-4">
                                    <Newspaper className="w-5 h-5 text-blue-400" />
                                    <h3 className="text-lg font-semibold text-white">Global Emergency Intelligence</h3>
                                </div>
                                <div className="space-y-4">
                                    {CURRENT_NEWS_FEED.map((news, index) => (
                                        <div key={index} className={`p-4 rounded-lg border-l-4 ${news.priority === 'high' ? 'bg-red-500/10 border-red-500' :
                                            news.priority === 'medium' ? 'bg-yellow-500/10 border-yellow-500' :
                                                'bg-blue-500/10 border-blue-500'
                                            }`}>
                                            <div className="flex items-start justify-between mb-2">
                                                <h4 className="font-semibold text-white">{news.headline}</h4>
                                                <span className="text-xs text-slate-400">{news.time}</span>
                                            </div>
                                            <p className="text-slate-300 text-sm mb-2">{news.content}</p>
                                            <div className="flex items-center space-x-2">
                                                <span className="text-xs text-slate-400">{news.source}</span>
                                                <span className={`px-2 py-1 text-xs rounded-full ${news.type === 'weather_alert' ? 'bg-orange-500/20 text-orange-400' :
                                                    news.type === 'health_alert' ? 'bg-red-500/20 text-red-400' :
                                                        'bg-blue-500/20 text-blue-400'
                                                    }`}>
                                                    {news.type.replace('_', ' ')}
                                                </span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Social Intelligence */}
                            <div className="bg-slate-800 rounded-lg p-6">
                                <div className="flex items-center space-x-3 mb-4">
                                    <MessageSquare className="w-5 h-5 text-purple-400" />
                                    <h3 className="text-lg font-semibold text-white">Social Intelligence</h3>
                                </div>
                                <div className="space-y-3">
                                    {SOCIAL_INTELLIGENCE.map((post, index) => (
                                        <div key={index} className="p-3 bg-slate-700 rounded-lg">
                                            <div className="flex items-start justify-between mb-2">
                                                <div className="flex items-center space-x-2">
                                                    <span className="text-xs text-slate-400">{post.platform}</span>
                                                    <span className="text-xs text-slate-500">•</span>
                                                    <span className="text-xs text-slate-400">{post.location}</span>
                                                </div>
                                                <span className="text-xs text-slate-400">{post.time}</span>
                                            </div>
                                            <p className="text-sm text-slate-200 mb-2">{post.content}</p>
                                            <div className="flex items-center justify-between">
                                                <span className={`text-xs px-2 py-1 rounded-full ${post.sentiment === 'concern' ? 'bg-yellow-500/20 text-yellow-400' :
                                                    post.sentiment === 'neutral' ? 'bg-gray-500/20 text-gray-400' :
                                                        'bg-blue-500/20 text-blue-400'
                                                    }`}>
                                                    {post.sentiment}
                                                </span>
                                                <span className="text-xs text-slate-400">{post.engagement} interactions</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Threat Activation Panel */}
                        <div className="bg-slate-800 rounded-lg p-6">
                            <h3 className="text-lg font-semibold text-white mb-4">Activate Intelligence Monitoring</h3>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                <button
                                    onClick={() => handleActivateIntelligence('Southeast Asia', 'Flood')}
                                    className="p-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                                >
                                    <Globe className="w-6 h-6 mb-2" />
                                    <div className="text-sm font-medium">Monitor Flood Risk</div>
                                    <div className="text-xs opacity-75">Southeast Asia Region</div>
                                </button>
                                <button
                                    onClick={() => handleActivateIntelligence('Pacific Region', 'Tsunami')}
                                    className="p-4 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                                >
                                    <Activity className="w-6 h-6 mb-2" />
                                    <div className="text-sm font-medium">Monitor Tsunami Risk</div>
                                    <div className="text-xs opacity-75">Pacific Region</div>
                                </button>
                                <button
                                    onClick={() => handleActivateIntelligence('Global', 'Pandemic')}
                                    className="p-4 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
                                >
                                    <Heart className="w-6 h-6 mb-2" />
                                    <div className="text-sm font-medium">Monitor Health Risks</div>
                                    <div className="text-xs opacity-75">Global Coverage</div>
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {/* Emergency Intelligence Mode - Live Response */}
                {activeMode === 'respond' && (
                    <div className="space-y-6">
                        <div className="text-center mb-8">
                            <h2 className="text-3xl font-bold text-white mb-2">Live Emergency Intelligence</h2>
                            <p className="text-slate-400 text-lg">Real-time AI-powered emergency response coordination</p>
                        </div>

                        {!isEmergencyActive ? (
                            <>
                                {/* Emergency Response Configuration */}
                                <div className="bg-slate-800 rounded-lg p-8">
                                    <div className="flex items-center gap-3 mb-6">
                                        <MapPin className="w-6 h-6 text-blue-400" />
                                        <h2 className="text-2xl font-bold">Emergency Response Configuration</h2>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                                        <div className="space-y-3">
                                            <label className="block text-sm font-medium text-gray-300">Emergency Type</label>
                                            <select
                                                value={emergencyConfig.disaster_type}
                                                onChange={(e) => setEmergencyConfig(prev => ({ ...prev, disaster_type: e.target.value }))}
                                                className="w-full px-4 py-3 bg-white/10 rounded-lg text-white border border-white/20 focus:border-cyan-400 focus:outline-none"
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
                                                value={emergencyConfig.location}
                                                onChange={(e) => setEmergencyConfig(prev => ({ ...prev, location: e.target.value }))}
                                                className="w-full px-4 py-3 bg-white/10 rounded-lg text-white border border-white/20 focus:border-cyan-400 focus:outline-none"
                                                placeholder="e.g., Phuket, Thailand"
                                            />
                                        </div>

                                        <div className="space-y-3">
                                            <label className="block text-sm font-medium text-gray-300">
                                                Severity: {emergencyConfig.severity} {currentDisaster?.severity_scale.unit}
                                            </label>
                                            <input
                                                type="range"
                                                min={currentDisaster?.severity_scale.min || 1}
                                                max={currentDisaster?.severity_scale.max || 10}
                                                value={emergencyConfig.severity}
                                                onChange={(e) => setEmergencyConfig(prev => ({ ...prev, severity: parseInt(e.target.value) }))}
                                                className="w-full h-3 bg-white/20 rounded-lg appearance-none cursor-pointer slider"
                                            />
                                        </div>

                                        <div className="space-y-3">
                                            <label className="block text-sm font-medium text-gray-300">Duration (hours)</label>
                                            <input
                                                type="number"
                                                min="1"
                                                max="24"
                                                value={emergencyConfig.duration}
                                                onChange={(e) => setEmergencyConfig(prev => ({ ...prev, duration: parseInt(e.target.value) }))}
                                                className="w-full px-4 py-3 bg-white/10 rounded-lg text-white border border-white/20 focus:border-cyan-400 focus:outline-none"
                                            />
                                        </div>
                                    </div>

                                    {/* Control Panel */}
                                    <div className="flex items-center justify-between mt-8 pt-6 border-t border-white/10">
                                        <div className="flex items-center gap-4">
                                            <div className="flex items-center gap-2">
                                                <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-400' : 'bg-gray-400'}`} />
                                                <span className="text-gray-300 font-medium">{isConnected ? 'Intelligence System Ready' : 'System Offline'}</span>
                                            </div>
                                        </div>

                                        <div className="flex items-center gap-4">
                                            {error && (
                                                <div className="text-red-400 text-sm max-w-xs truncate bg-red-500/10 px-3 py-2 rounded border border-red-500/30" title={error}>
                                                    {error}
                                                </div>
                                            )}

                                            <button
                                                onClick={handleActivateEmergencyResponse}
                                                disabled={!isConnected}
                                                className="bg-red-600 hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-3 px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-300 text-white"
                                            >
                                                <Play className="w-5 h-5" />
                                                Activate Emergency Intelligence
                                            </button>
                                        </div>
                                    </div>
                                </div>

                                {/* System Status Display */}
                                <div className="bg-slate-800 rounded-lg p-8 text-center">
                                    <Shield className="w-12 h-12 text-green-400 mx-auto mb-4" />
                                    <h3 className="text-xl font-semibold text-white mb-2">Emergency Intelligence Standby</h3>
                                    <p className="text-slate-400">Configure emergency parameters above to activate live intelligence coordination</p>
                                    {systemInfo && (
                                        <div className="mt-4 text-sm text-slate-500">
                                            System Version: {systemInfo.eris_version} | AI Model: {systemInfo.orchestrator?.ai_model || 'Gemini 2.0 Flash'} | {systemInfo.agent_system?.total_agents || 10} Intelligence Agents Ready
                                        </div>
                                    )}
                                </div>
                            </>
                        ) : (
                            <>
                                {/* Active Emergency Response Dashboard */}
                                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                    {/* Intelligence Agent Network */}
                                    <div className="lg:col-span-2">
                                        <div className="bg-slate-800 rounded-lg p-8 h-full">
                                            <div className="flex items-center justify-between mb-6">
                                                <div>
                                                    <h2 className="text-2xl font-bold">AI Intelligence Network</h2>
                                                    <p className="text-cyan-300 mt-1">
                                                        {systemInfo ? `${systemInfo.agent_system?.total_agents || 10} intelligence agents` : 'Loading...'} •
                                                        {systemInfo?.agent_system ? ` ${systemInfo.agent_system.adk_agents} Strategic + ${systemInfo.agent_system.enhanced_agents} Tactical` : ' 6 Strategic + 4 Tactical'} •
                                                        {orchestratorInfo?.ai_model || systemInfo?.orchestrator?.ai_model || 'Gemini 2.0 Flash'}
                                                    </p>
                                                </div>
                                                <div className="px-4 py-2 bg-green-500/10 rounded-lg border border-green-500/30">
                                                    <span className="text-green-300 font-medium">
                                                        {agents.filter(a => a.status === 'analyzing').length}/10 Analyzing
                                                    </span>
                                                </div>
                                            </div>

                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                                {agents.map((agent) => {
                                                    const IconComponent = agent.icon;
                                                    const isActive = isEmergencyActive && agent.status === 'analyzing';

                                                    return (
                                                        <div key={agent.id} className="bg-white/5 rounded-xl p-4 border border-white/10 hover:border-white/20 transition-all duration-300">
                                                            <div className="flex items-center justify-between mb-3">
                                                                <div className="flex items-center gap-3">
                                                                    <div className={`p-2 rounded-lg transition-all duration-300 ${isActive ? `bg-gradient-to-r ${agent.color} bg-opacity-30` : 'bg-gray-500/20'}`}>
                                                                        <IconComponent className="w-4 h-4 text-white" />
                                                                    </div>
                                                                    <div>
                                                                        <div className="font-semibold text-white text-sm">{agent.name}</div>
                                                                        <div className="text-xs text-gray-400">
                                                                            {agent.type === 'adk' ? 'Strategic' : 'Tactical'} • {orchestratorInfo?.ai_model || 'Gemini 2.0 Flash'} • {Math.round(agent.efficiency || 95)}% efficiency
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                                <div className="text-right">
                                                                    <div className={`px-2 py-1 rounded text-xs font-medium ${isActive ? 'bg-green-500/20 text-green-300' : 'bg-gray-500/20 text-gray-400'}`}>
                                                                        {agent.status || 'standby'}
                                                                    </div>
                                                                </div>
                                                            </div>

                                                            <div className="mb-2">
                                                                <div className="text-xs text-gray-300 mb-1">{agent.description}</div>
                                                            </div>

                                                            {/* Progress Bar */}
                                                            <div className="mb-2">
                                                                <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
                                                                    <span>Analysis Progress</span>
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

                                            {/* Emergency Response Control */}
                                            <div className="mt-6 pt-6 border-t border-white/10">
                                                <div className="flex items-center justify-between">
                                                    <div className="flex items-center gap-4">
                                                        <div className="text-sm text-gray-300">
                                                            Emergency ID: <span className="font-mono text-blue-400">{currentEmergencyId?.slice(0, 8)}...</span>
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <div className="w-3 h-3 rounded-full bg-red-400 animate-pulse" />
                                                            <span className="text-gray-300 font-medium">Live Intelligence Active</span>
                                                        </div>
                                                    </div>
                                                    <button
                                                        onClick={handleStopEmergencyResponse}
                                                        className="bg-gray-600 hover:bg-gray-700 text-white px-6 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
                                                    >
                                                        <Square className="w-4 h-4" />
                                                        End Emergency Response
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Emergency Metrics & Feeds Sidebar */}
                                    <div className="space-y-6">
                                        {/* Key Emergency Metrics */}
                                        <div className="bg-slate-800 rounded-lg p-6">
                                            <h3 className="text-xl font-bold mb-4">Emergency Metrics</h3>
                                            <div className="space-y-4">
                                                <div className="flex items-center justify-between">
                                                    <span className="text-gray-400">Hospital Capacity</span>
                                                    <span className={`font-bold ${emergencyMetrics.hospital_capacity > 80 ? 'text-red-400' : emergencyMetrics.hospital_capacity > 60 ? 'text-yellow-400' : 'text-green-400'}`}>
                                                        {Math.round(emergencyMetrics.hospital_capacity || 0)}%
                                                    </span>
                                                </div>
                                                <div className="flex items-center justify-between">
                                                    <span className="text-gray-400">Population at Risk</span>
                                                    <span className="font-bold text-orange-400">{(emergencyMetrics.population_at_risk || 0).toLocaleString()}</span>
                                                </div>
                                                <div className="flex items-center justify-between">
                                                    <span className="text-gray-400">Infrastructure Status</span>
                                                    <span className="font-bold text-blue-400">{Math.round(emergencyMetrics.infrastructure_status || 0)}%</span>
                                                </div>
                                                <div className="flex items-center justify-between">
                                                    <span className="text-gray-400">Response Readiness</span>
                                                    <span className="font-bold text-green-400">{Math.round(emergencyMetrics.response_readiness || 0)}%</span>
                                                </div>
                                                <div className="flex items-center justify-between">
                                                    <span className="text-gray-400">Public Confidence</span>
                                                    <span className={`font-bold ${emergencyMetrics.public_confidence > 70 ? 'text-green-400' : emergencyMetrics.public_confidence > 50 ? 'text-yellow-400' : 'text-red-400'}`}>
                                                        {Math.round(emergencyMetrics.public_confidence || 0)}%
                                                    </span>
                                                </div>
                                                <div className="flex items-center justify-between">
                                                    <span className="text-gray-400">Evacuation Readiness</span>
                                                    <span className={`font-bold ${emergencyMetrics.evacuation_readiness > 70 ? 'text-green-400' : emergencyMetrics.evacuation_readiness > 50 ? 'text-yellow-400' : 'text-red-400'}`}>
                                                        {Math.round(emergencyMetrics.evacuation_readiness || 0)}%
                                                    </span>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Emergency Command Feed */}
                                        <div className="bg-slate-800 rounded-lg p-6">
                                            <div className="flex items-center gap-3 mb-4">
                                                <AlertTriangle className="w-5 h-5 text-red-400" />
                                                <h3 className="text-lg font-bold">Emergency Command</h3>
                                            </div>
                                            <div className="space-y-3 max-h-64 overflow-y-auto">
                                                {emergencyFeed.length === 0 ? (
                                                    <div className="text-gray-500 text-sm text-center py-4">
                                                        {isEmergencyActive ? 'Monitoring for emergency updates...' : 'No emergency alerts'}
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

                                        {/* Social Intelligence Feed */}
                                        <div className="bg-slate-800 rounded-lg p-6">
                                            <div className="flex items-center gap-3 mb-4">
                                                <MessageSquare className="w-5 h-5 text-blue-400" />
                                                <h3 className="text-lg font-bold">Social Intelligence</h3>
                                            </div>
                                            <div className="space-y-3 max-h-64 overflow-y-auto">
                                                {socialFeed.length === 0 ? (
                                                    <div className="text-gray-500 text-sm text-center py-4">
                                                        {isEmergencyActive ? 'Monitoring social media...' : 'No social activity'}
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

                                {/* Active Emergency Actions */}
                                {emergencyMetrics.alert_level !== 'GREEN' && (
                                    <div className="bg-gradient-to-r from-red-900 to-orange-900 rounded-lg p-6 border border-red-500">
                                        <div className="flex items-center space-x-3 mb-4">
                                            <AlertTriangle className="w-8 h-8 text-red-400" />
                                            <div>
                                                <h3 className="text-xl font-bold text-white">Active Emergency: {emergencyConfig.disaster_type.replace('_', ' ')}</h3>
                                                <p className="text-red-200">{emergencyConfig.location} - Alert Level: {emergencyMetrics.alert_level}</p>
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                            <div>
                                                <h4 className="font-semibold text-white mb-3">Recommended Actions</h4>
                                                <div className="space-y-2">
                                                    {BEST_PRACTICES[emergencyConfig.disaster_type]?.map((action, index) => (
                                                        <div key={index} className="flex items-start p-2 bg-white/10 rounded">
                                                            <Target className="w-4 h-4 text-orange-400 mt-1 mr-2 flex-shrink-0" />
                                                            <span className="text-white text-sm">{action}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>

                                            <div>
                                                <h4 className="font-semibold text-white mb-3">Similar Historical Cases</h4>
                                                <div className="space-y-2">
                                                    {HISTORICAL_DISASTERS
                                                        .filter(d => d.type.toLowerCase().includes(emergencyConfig.disaster_type.split('_')[0]))
                                                        .slice(0, 2)
                                                        .map((disaster, index) => (
                                                            <div key={index} className="p-2 bg-white/10 rounded">
                                                                <div className="text-sm font-medium text-white">{disaster.name}</div>
                                                                <div className="text-xs text-orange-200">{disaster.location} • {disaster.year}</div>
                                                            </div>
                                                        ))}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                )}

                {error && (
                    <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
                        <div className="flex items-center space-x-2">
                            <AlertCircle className="w-5 h-5 text-red-400" />
                            <span className="text-red-400">{error}</span>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default EmergencyIntelligencePlatform;
