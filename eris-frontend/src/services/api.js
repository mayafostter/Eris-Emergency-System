import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

class ERISApiService {
    constructor() {
        this.client = axios.create({
            baseURL: API_BASE_URL,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
            },
        });

        // Add response interceptor for error handling
        this.client.interceptors.response.use(
            (response) => response,
            (error) => {
                console.error('API Error:', error.response?.data || error.message);
                throw error;
            }
        );
    }

    // EXACT MATCH: Your /simulate endpoint (FIXED: 24-hour duration)
    async startSimulation(config) {
        try {
            const response = await this.client.post('/simulate', {
                disaster_type: config.disaster_type,
                location: config.location,
                severity: config.severity,
                duration: config.duration || 24  // ✅ FIXED: 24 hours default (was 72)
            });
            return response.data;
        } catch (error) {
            console.error('Failed to start simulation:', error);
            throw error;
        }
    }

    // EXACT MATCH: Your /status/{simulation_id} endpoint
    async getSimulationStatus(simulationId) {
        try {
            const response = await this.client.get(`/status/${simulationId}`);
            return response.data;
        } catch (error) {
            console.error(`Failed to get simulation status for ${simulationId}:`, error);
            throw error;
        }
    }

    // EXACT MATCH: Your /metrics/dashboard/{simulation_id} endpoint ⭐
    async getDashboardMetrics(simulationId) {
        try {
            const response = await this.client.get(`/metrics/dashboard/${simulationId}`);
            return response.data;
        } catch (error) {
            console.error(`Failed to get dashboard metrics for ${simulationId}:`, error);
            throw error;
        }
    }

    // EXACT MATCH: Your /extended-metrics/{simulation_id} endpoint
    async getExtendedMetrics(simulationId) {
        try {
            const response = await this.client.get(`/extended-metrics/${simulationId}`);
            return response.data;
        } catch (error) {
            console.error(`Failed to get extended metrics for ${simulationId}:`, error);
            throw error;
        }
    }

    // EXACT MATCH: Your /orchestrator/{simulation_id}/agents endpoint
    async getAllAgents(simulationId) {
        try {
            const response = await this.client.get(`/orchestrator/${simulationId}/agents`);
            return response.data;
        } catch (error) {
            console.error(`Failed to get agents for ${simulationId}:`, error);
            throw error;
        }
    }

    // EXACT MATCH: Your /agents/health endpoint
    async getAgentsHealth() {
        try {
            const response = await this.client.get('/agents/health');
            return response.data;
        } catch (error) {
            console.error('Failed to get agents health:', error);
            throw error;
        }
    }

    // EXACT MATCH: Your /system/info endpoint
    async getSystemInfo() {
        try {
            const response = await this.client.get('/system/info');
            return response.data;
        } catch (error) {
            console.error('Failed to get system info:', error);
            throw error;
        }
    }

    // EXACT MATCH: Your /health endpoint
    async getSystemHealth() {
        try {
            const response = await this.client.get('/health');
            return response.data;
        } catch (error) {
            console.error('Failed to get system health:', error);
            throw error;
        }
    }

    // EXACT MATCH: Your /metrics/{simulation_id} endpoint
    async getBasicMetrics(simulationId) {
        try {
            const response = await this.client.get(`/metrics/${simulationId}`);
            return response.data;
        } catch (error) {
            console.error(`Failed to get basic metrics for ${simulationId}:`, error);
            throw error;
        }
    }

    // Helper method to get WebSocket URL
    getWebSocketUrl(simulationId) {
        const wsBaseUrl = import.meta.env.VITE_WS_URL || 'ws://127.0.0.1:8000';
        return `${wsBaseUrl}/ws/metrics/${simulationId}`;
    }

    // Test backend connection
    async testConnection() {
        try {
            const health = await this.getSystemHealth();
            const systemInfo = await this.getSystemInfo();
            return {
                connected: true,
                health,
                systemInfo
            };
        } catch (error) {
            return {
                connected: false,
                error: error.message
            };
        }
    }
}

export default new ERISApiService();
