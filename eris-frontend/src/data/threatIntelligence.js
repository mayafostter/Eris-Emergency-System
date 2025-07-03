// Current threat intelligence feeds
export const CURRENT_NEWS_FEED = [
    {
        time: '2 hours ago',
        source: 'National Weather Service',
        headline: 'Severe Weather Warning: Potential flooding in Southeast Asia',
        content: 'Monsoon systems developing. Emergency managers should review flood response protocols.',
        priority: 'high',
        type: 'weather_alert'
    },
    {
        time: '4 hours ago',
        source: 'FEMA Updates',
        headline: 'New Hospital Evacuation Guidelines Released',
        content: 'Updated protocols based on lessons learned from recent hurricane responses.',
        priority: 'medium',
        type: 'best_practice'
    },
    {
        time: '6 hours ago',
        source: 'WHO Emergency',
        headline: 'Disease Outbreak Preparedness: Key Indicators to Monitor',
        content: 'Early warning signs and response triggers for epidemic situations.',
        priority: 'medium',
        type: 'health_alert'
    },
    {
        time: '8 hours ago',
        source: 'NOAA Climate Center',
        headline: 'Extreme Weather Patterns Intensifying Globally',
        content: 'Climate data shows increasing frequency of severe weather events requiring enhanced preparedness.',
        priority: 'medium',
        type: 'climate_alert'
    },
    {
        time: '12 hours ago',
        source: 'International Red Cross',
        headline: 'Updated Mass Casualty Response Protocols',
        content: 'New guidelines for coordinating multi-agency response to large-scale emergencies.',
        priority: 'medium',
        type: 'best_practice'
    }
];

// Social intelligence monitoring
export const SOCIAL_INTELLIGENCE = [
    {
        time: '15 min ago',
        platform: 'Twitter',
        location: 'Phuket, Thailand',
        sentiment: 'concern',
        content: 'Local residents reporting unusual weather patterns. Emergency services on standby.',
        engagement: 245,
        verified: true
    },
    {
        time: '32 min ago',
        platform: 'Facebook',
        location: 'Bangkok, Thailand',
        sentiment: 'neutral',
        content: 'Hospital emergency drill completed successfully. All systems operational.',
        engagement: 89,
        verified: true
    },
    {
        time: '1 hour ago',
        platform: 'Local News',
        location: 'Regional',
        sentiment: 'informative',
        content: 'Emergency managers gathering for quarterly preparedness review.',
        engagement: 156,
        verified: true
    },
    {
        time: '1.5 hours ago',
        platform: 'Instagram',
        location: 'Manila, Philippines',
        sentiment: 'concern',
        content: 'Heavy rains causing minor flooding in downtown areas. Traffic advisories issued.',
        engagement: 312,
        verified: false
    },
    {
        time: '2 hours ago',
        platform: 'Twitter',
        location: 'Jakarta, Indonesia',
        sentiment: 'neutral',
        content: 'Earthquake preparedness drill scheduled for next week. Community participation encouraged.',
        engagement: 78,
        verified: true
    },
    {
        time: '3 hours ago',
        platform: 'LinkedIn',
        location: 'Singapore',
        sentiment: 'informative',
        content: 'Emergency management conference highlights importance of regional cooperation.',
        engagement: 134,
        verified: true
    }
];

// Emergency management best practices
export const BEST_PRACTICES = {
    'hurricane': [
        'Evacuate coastal areas 72 hours before landfall',
        'Pre-position resources outside the storm path',
        'Establish redundant communication systems',
        'Coordinate with neighboring jurisdictions'
    ],
    'flood': [
        'Monitor upstream river conditions continuously',
        'Activate emergency alert systems early',
        'Ensure hospital evacuation plans are current',
        'Pre-stage rescue equipment at safe locations'
    ],
    'earthquake': [
        'Immediate structural assessments prevent casualties',
        'Hospital surge capacity planning critical',
        'Search and rescue coordination within first 72 hours',
        'Aftershock preparedness saves lives'
    ],
    'wildfire': [
        'Create defensible space around critical infrastructure',
        'Establish multiple evacuation routes',
        'Pre-position firefighting resources',
        'Monitor weather conditions continuously'
    ],
    'tsunami': [
        'Immediate evacuation to higher ground',
        'Monitor seismic activity continuously',
        'Test warning systems regularly',
        'Educate coastal communities on natural warning signs'
    ],
    'volcanic_eruption': [
        'Monitor volcanic activity continuously',
        'Establish evacuation zones based on hazard maps',
        'Protect against ashfall impacts',
        'Coordinate with aviation authorities'
    ],
    'severe_storm': [
        'Monitor weather systems 72 hours in advance',
        'Secure critical infrastructure',
        'Prepare for extended power outages',
        'Coordinate emergency shelter operations'
    ],
    'pandemic': [
        'Implement early detection systems',
        'Establish healthcare surge capacity',
        'Coordinate public health messaging',
        'Ensure supply chain continuity'
    ],
    'epidemic': [
        'Activate contact tracing protocols',
        'Isolate affected populations quickly',
        'Coordinate with health authorities',
        'Implement community containment measures'
    ],
    'landslide': [
        'Monitor rainfall and slope conditions',
        'Evacuate high-risk areas promptly',
        'Establish debris flow barriers',
        'Coordinate search and rescue operations'
    ]
};
