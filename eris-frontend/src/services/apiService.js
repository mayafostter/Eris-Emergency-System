// API Service with WebSocket support
export const apiService = {
    baseURL: 'https://eris-backend-621360763676.us-central1.run.app',

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

    getWebSocketURL(simulationId) {
        const wsProtocol = this.baseURL.startsWith('https') ? 'wss' : 'ws';
        const wsBaseURL = this.baseURL.replace(/^https?/, wsProtocol);
        return `${wsBaseURL}/ws/metrics/${simulationId}`;
    }
};
