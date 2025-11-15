#!/usr/bin/env python3
"""
SAMHSA (Substance Abuse and Mental Health Services Administration) Data Integration
Provides access to mental health prevalence data for correlation with semantic search results
"""

import requests
import json
from typing import Dict, List, Optional
from datetime import datetime

# SAMHSA Data Sources
# Primary: data.gov API for NSDUH (National Survey on Drug Use and Health)
# Alternative: Direct CSV/JSON downloads from SAMHSA website

DATA_GOV_BASE_URL = "https://catalog.data.gov/api/3/action"
SAMHSA_NSDUH_DATASET_ID = None  # Will be determined from data.gov search

# State abbreviation mapping
STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# Reverse mapping
ABBREV_TO_STATE = {v: k for k, v in STATE_ABBREV.items()}

def search_samhsa_datasets(query: str = "SAMHSA NSDUH", limit: int = 10) -> List[Dict]:
    """
    Search data.gov for SAMHSA datasets
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        List of dataset metadata
    """
    try:
        url = f"{DATA_GOV_BASE_URL}/package_search"
        params = {
            'q': query,
            'rows': limit
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('success') and 'result' in data:
            return data['result'].get('results', [])
        return []
    except Exception as e:
        print(f"Warning: Could not search data.gov for SAMHSA datasets: {e}")
        return []

def fetch_samhsa_state_data(state: str = None, metric: str = "SMI") -> Dict[str, float]:
    """
    Fetch SAMHSA mental health prevalence data by state
    
    Args:
        state: State name or abbreviation (None for all states)
        metric: Metric type ('SMI' for Serious Mental Illness, 'AMI' for Any Mental Illness, etc.)
        
    Returns:
        Dictionary mapping state abbreviations to prevalence rates (as percentages)
    """
    # For now, return mock data structure
    # In production, this would fetch from data.gov API or SAMHSA's direct data sources
    
    # Mock data structure based on approximate NSDUH 2021 data
    # These are placeholder values - real implementation would fetch from API
    mock_data = {
        'AL': 4.2, 'AK': 4.8, 'AZ': 4.5, 'AR': 4.1, 'CA': 5.1,
        'CO': 4.9, 'CT': 4.6, 'DE': 4.3, 'FL': 4.7, 'GA': 4.0,
        'HI': 4.4, 'ID': 4.2, 'IL': 4.8, 'IN': 4.3, 'IA': 4.1,
        'KS': 4.2, 'KY': 4.4, 'LA': 4.3, 'ME': 4.7, 'MD': 4.6,
        'MA': 5.0, 'MI': 4.9, 'MN': 4.5, 'MS': 3.9, 'MO': 4.4,
        'MT': 4.6, 'NE': 4.0, 'NV': 4.8, 'NH': 4.7, 'NJ': 4.9,
        'NM': 4.4, 'NY': 5.2, 'NC': 4.3, 'ND': 4.1, 'OH': 4.6,
        'OK': 4.2, 'OR': 5.0, 'PA': 4.8, 'RI': 4.7, 'SC': 4.1,
        'SD': 4.0, 'TN': 4.2, 'TX': 4.4, 'UT': 4.3, 'VT': 4.8,
        'VA': 4.5, 'WA': 5.1, 'WV': 4.5, 'WI': 4.6, 'WY': 4.4
    }
    
    if state:
        # Convert state name to abbreviation if needed
        state_abbrev = STATE_ABBREV.get(state, state) if len(state) > 2 else state
        return {state_abbrev: mock_data.get(state_abbrev, 0.0)}
    
    return mock_data

def get_samhsa_metrics_for_states(states: List[str], metric: str = "SMI") -> Dict[str, float]:
    """
    Get SAMHSA metrics for a list of states
    
    Args:
        states: List of state names or abbreviations
        metric: Metric type ('SMI', 'AMI', 'Substance Use Disorder', etc.)
        
    Returns:
        Dictionary mapping state abbreviations to prevalence rates
    """
    all_data = fetch_samhsa_state_data(metric=metric)
    
    result = {}
    for state in states:
        state_abbrev = STATE_ABBREV.get(state, state) if len(state) > 2 else state
        if state_abbrev in all_data:
            result[state_abbrev] = all_data[state_abbrev]
    
    return result

def correlate_search_results_with_samhsa(
    search_results_geo: Dict,
    samhsa_metric: str = "SMI"
) -> Dict:
    """
    Correlate semantic search results with SAMHSA prevalence data
    
    Args:
        search_results_geo: Geographic distribution from search results
        samhsa_metric: SAMHSA metric to compare against
        
    Returns:
        Dictionary with correlation data including:
        - state_samhsa_data: SAMHSA prevalence by state
        - correlation_scores: Correlation between search results and SAMHSA data
        - comparison_data: Side-by-side comparison
    """
    # Get states from search results
    states = list(search_results_geo.get('state_counts', {}).keys())
    
    # Fetch SAMHSA data for those states
    samhsa_data = get_samhsa_metrics_for_states(states, metric=samhsa_metric)
    
    # Calculate correlation
    comparison_data = {}
    for state in states:
        state_abbrev = STATE_ABBREV.get(state, state) if len(state) > 2 else state
        state_count_data = search_results_geo['state_counts'].get(state, {})
        
        # Get search result intensity (category score or count)
        search_intensity = 0
        if 'category_scores' in state_count_data:
            # Use highest category score
            if state_count_data['category_scores']:
                search_intensity = max(state_count_data['category_scores'].values())
        else:
            search_intensity = state_count_data.get('count', 0)
        
        samhsa_prevalence = samhsa_data.get(state_abbrev, 0.0)
        
        comparison_data[state_abbrev] = {
            'state_name': state,
            'search_intensity': search_intensity,
            'samhsa_prevalence': samhsa_prevalence,
            'correlation_ratio': search_intensity / samhsa_prevalence if samhsa_prevalence > 0 else 0
        }
    
    return {
        'state_samhsa_data': samhsa_data,
        'comparison_data': comparison_data,
        'metric_used': samhsa_metric,
        'note': 'SAMHSA data represents annual prevalence rates. Correlation ratios indicate relative intensity.'
    }

def enrich_geographic_distribution_with_samhsa(
    geo_distribution: Dict,
    include_samhsa: bool = True,
    samhsa_metric: str = "SMI"
) -> Dict:
    """
    Enrich geographic distribution data with SAMHSA prevalence metrics
    
    Args:
        geo_distribution: Geographic distribution from search results
        include_samhsa: Whether to include SAMHSA data
        samhsa_metric: SAMHSA metric to include
        
    Returns:
        Enriched geographic distribution with SAMHSA data
    """
    if not include_samhsa:
        return geo_distribution
    
    try:
        correlation_data = correlate_search_results_with_samhsa(geo_distribution, samhsa_metric)
        geo_distribution['samhsa_data'] = correlation_data
        return geo_distribution
    except Exception as e:
        print(f"Warning: Could not enrich with SAMHSA data: {e}")
        return geo_distribution

