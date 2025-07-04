// Historical disaster database - ordered from most recent to oldest
export const HISTORICAL_DISASTERS = [
    {
        id: 'valencia_floods_2024',
        name: '2024 Valencia Flash Floods',
        type: 'Severe Storm',
        year: 2024,
        location: 'Valencia, Spain',
        casualties: 223,
        damage: '$31 billion',
        lessons: ['Flash flood warning systems inadequate', 'Urban drainage infrastructure overwhelmed', 'Climate change intensifies rainfall', 'Mobile alert systems critical'],
        timeline: '5 days',
        similarity: 94
    },
    {
        id: 'turkey_syria_earthquake_2023',
        name: '2023 Turkey-Syria Earthquake',
        type: 'Earthquake',
        year: 2023,
        location: 'Turkey and Syria',
        casualties: 59259,
        damage: '$104 billion',
        lessons: ['International aid coordination vital', 'Building codes enforcement critical', 'Cross-border cooperation essential'],
        timeline: '21 days',
        similarity: 90
    },
    {
        id: 'texas_winter_storm_2021',
        name: 'Texas Winter Storm Uri',
        type: 'Severe Storm',
        year: 2021,
        location: 'Texas, USA',
        casualties: 246,
        damage: '$195 billion',
        lessons: ['Energy grid winterization needed', 'Water system freeze protection', 'Emergency shelter coordination'],
        timeline: '7 days',
        similarity: 85
    },
    {
        id: 'covid19_pandemic_2020',
        name: 'COVID-19 Global Pandemic',
        type: 'Pandemic',
        year: 2020,
        location: 'Global',
        casualties: 7000000,
        damage: '$16 trillion',
        lessons: ['Digital health systems essential', 'Supply chain vulnerabilities exposed', 'International coordination critical', 'Remote work capabilities vital'],
        timeline: '1095 days',
        similarity: 95
    },
    {
        id: 'australia_bushfires_2020',
        name: 'Australian Bushfire Crisis',
        type: 'Wildfire',
        year: 2020,
        location: 'Australia',
        casualties: 34,
        damage: '$103 billion',
        lessons: ['Climate change factor', 'Wildlife evacuation protocols', 'Smoke health impacts underestimated'],
        timeline: '240 days',
        similarity: 78
    },
    {
        id: 'dorian_hurricane_2019',
        name: 'Hurricane Dorian',
        type: 'Hurricane',
        year: 2019,
        location: 'Bahamas and US East Coast',
        casualties: 84,
        damage: '$5.1 billion',
        lessons: ['Island nation evacuation complexity', 'Storm surge prediction accuracy', 'Long-duration storm impacts'],
        timeline: '21 days',
        similarity: 87
    },
    {
        id: 'guatemala_volcano_2018',
        name: '2018 Volcán de Fuego Eruption',
        type: 'Volcanic Eruption',
        year: 2018,
        location: 'Guatemala',
        casualties: 215,
        damage: '$550 million',
        lessons: ['Pyroclastic flow unpredictability', 'Evacuation zone enforcement critical', 'Volcanic monitoring system gaps', 'Community education vital'],
        timeline: '30 days',
        similarity: 89
    },
    {
        id: 'maria_hurricane_2017',
        name: 'Hurricane Maria',
        type: 'Hurricane',
        year: 2017,
        location: 'Puerto Rico',
        casualties: 2975,
        damage: '$90 billion',
        lessons: ['Island infrastructure vulnerability', 'Extended power outage management', 'Federal response coordination'],
        timeline: '365 days',
        similarity: 93
    },
    {
        id: 'nepal_earthquake_2015',
        name: '2015 Nepal Earthquake',
        type: 'Earthquake',
        year: 2015,
        location: 'Nepal',
        casualties: 8964,
        damage: '$7 billion',
        lessons: ['Mountain rescue operations', 'Historic structure vulnerability', 'International SAR coordination'],
        timeline: '45 days',
        similarity: 86
    },
    {
        id: 'ebola_outbreak_2014',
        name: '2014 West Africa Ebola Outbreak',
        type: 'Epidemic',
        year: 2014,
        location: 'West Africa',
        casualties: 11323,
        damage: '$53 billion',
        lessons: ['Contact tracing protocols', 'Healthcare worker protection', 'Community engagement critical'],
        timeline: '730 days',
        similarity: 89
    },
    {
        id: 'sandy_hurricane_2012',
        name: 'Hurricane Sandy',
        type: 'Hurricane',
        year: 2012,
        location: 'US East Coast',
        casualties: 233,
        damage: '$65 billion',
        lessons: ['Urban coastal vulnerability', 'Mass transit shutdown planning', 'Hospital evacuation procedures'],
        timeline: '14 days',
        similarity: 88
    },
    {
        id: 'brazil_landslides_2011',
        name: '2011 Rio de Janeiro Landslides',
        type: 'Landslide',
        year: 2011,
        location: 'Rio de Janeiro, Brazil',
        casualties: 918,
        damage: '$1.2 billion',
        lessons: ['Urban hillside development risks', 'Early warning systems for slopes', 'Informal settlement vulnerability', 'Rainfall monitoring critical'],
        timeline: '14 days',
        similarity: 91
    },
    {
        id: 'thailand_floods_2011',
        name: '2011 Thailand Floods',
        type: 'Flood',
        year: 2011,
        location: 'Central Thailand',
        casualties: 815,
        damage: '$45 billion',
        lessons: ['Industrial zone protection failed', 'Social media crucial for coordination', 'International aid logistics complex'],
        timeline: '175 days',
        similarity: 94
    },
    {
        id: 'japan_tsunami_2011',
        name: '2011 Tōhoku Earthquake and Tsunami',
        type: 'Tsunami',
        year: 2011,
        location: 'Japan',
        casualties: 19747,
        damage: '$235 billion',
        lessons: ['Early warning systems worked', 'Nuclear preparedness failed', 'Community drills reduced casualties'],
        timeline: '14 days',
        similarity: 87
    },
    {
        id: 'haiti_earthquake_2010',
        name: '2010 Haiti Earthquake',
        type: 'Earthquake',
        year: 2010,
        location: 'Haiti',
        casualties: 316000,
        damage: '$8 billion',
        lessons: ['Building code enforcement critical', 'International aid coordination', 'Urban search and rescue challenges'],
        timeline: '90 days',
        similarity: 92
    },
    {
        id: 'hurricane_katrina_2005',
        name: 'Hurricane Katrina',
        type: 'Hurricane',
        year: 2005,
        location: 'New Orleans, Louisiana',
        casualties: 1833,
        damage: '$125 billion',
        lessons: ['Evacuation planning critical', 'Communication failures deadly', 'Hospital coordination saved lives'],
        timeline: '7 days',
        similarity: 92
    },
    {
        id: 'indian_ocean_tsunami_2004',
        name: '2004 Indian Ocean Tsunami',
        type: 'Tsunami',
        year: 2004,
        location: 'Indonesia, Sri Lanka, India, and Thailand',
        casualties: 230000,
        damage: '$15 billion',
        lessons: ['Tsunami warning systems critical', 'Tourist area vulnerability', 'International aid coordination', 'Coastal community preparedness vital'],
        timeline: '30 days',
        similarity: 96
    }
];
