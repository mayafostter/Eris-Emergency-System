import { AlertTriangle, Heart, Settings, BarChart3, Radio, TrendingUp, Activity, MessageCircle, Zap, Cloud } from 'lucide-react';

// Emergency Intelligence Agent Configuration
export const createAgentConfig = () => [
    {
        id: 'emergency_response_coordinator',
        name: 'Emergency Response Coordinator',
        icon: AlertTriangle,
        color: 'from-red-500 to-orange-500',
        type: 'adk',
        description: 'Coordinating all emergency response operations'
    },
    {
        id: 'public_health_manager',
        name: 'Public Health Manager',
        icon: Heart,
        color: 'from-pink-500 to-rose-500',
        type: 'adk',
        description: 'Managing health system response and capacity'
    },
    {
        id: 'infrastructure_manager',
        name: 'Infrastructure Manager',
        icon: Settings,
        color: 'from-yellow-500 to-amber-500',
        type: 'adk',
        description: 'Assessing critical infrastructure status'
    },
    {
        id: 'logistics_coordinator',
        name: 'Logistics Coordinator',
        icon: BarChart3,
        color: 'from-green-500 to-emerald-500',
        type: 'adk',
        description: 'Managing resource allocation and supply chains'
    },
    {
        id: 'communications_director',
        name: 'Communications Director',
        icon: Radio,
        color: 'from-purple-500 to-violet-500',
        type: 'adk',
        description: 'Coordinating public communications and alerts'
    },
    {
        id: 'recovery_coordinator',
        name: 'Recovery Coordinator',
        icon: TrendingUp,
        color: 'from-blue-500 to-cyan-500',
        type: 'adk',
        description: 'Planning and managing recovery operations'
    },
    {
        id: 'hospital_load_modeler',
        name: 'Hospital Load Intelligence',
        icon: Activity,
        color: 'from-teal-500 to-cyan-500',
        type: 'enhanced',
        description: 'Real-time hospital capacity monitoring'
    },
    {
        id: 'public_behavior_simulator',
        name: 'Population Behavior Analysis',
        icon: MessageCircle,
        color: 'from-indigo-500 to-blue-500',
        type: 'enhanced',
        description: 'Analyzing public response patterns'
    },
    {
        id: 'social_media_dynamics',
        name: 'Social Media Intelligence',
        icon: Zap,
        color: 'from-orange-500 to-red-500',
        type: 'enhanced',
        description: 'Monitoring social media trends and misinformation'
    },
    {
        id: 'news_coverage_simulator',
        name: 'Media Coverage Intelligence',
        icon: Cloud,
        color: 'from-slate-500 to-gray-500',
        type: 'enhanced',
        description: 'Tracking news coverage and public trust'
    }
];

// Disaster Types for emergency response
export const DISASTER_TYPES = [
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
