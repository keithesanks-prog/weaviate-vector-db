#!/usr/bin/env python3
"""
Data Visualization Module for Social Services Experience Analytics Platform
Provides visual analytics for policy staff to understand multi-dimensional data patterns.
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional
import weaviate
import weaviate.classes.query as wvq

# Try to import dimensionality reduction libraries
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    print("Warning: scikit-learn not available. t-SNE visualization will use UMAP or be disabled.")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not available. UMAP visualization will be disabled.")

CLASS_NAME = "ClientExperience"

def get_all_experience_vectors(client) -> tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Fetch all experience vectors and metadata from Weaviate.
    
    Returns:
        Tuple of (vectors list, metadata list)
    """
    collection = client.collections.get(CLASS_NAME)
    
    result = collection.query.fetch_objects(
        limit=1000,
        include_vector=True,
        return_properties=[
            "text", "tag_abstract", "ed_level_primary", "religious_participation",
            "survey_anxiety", "survey_control", "survey_hope",
            "current_program_enrollment", "enrollment_status_change",
            "service_office_location", "urban_rural_designation",
            "residential_moves_count", "financial_volatility_index",
            "primary_language"
        ]
    )
    
    vectors = []
    metadata_list = []
    
    for obj in result.objects:
        vector = obj.vector.get("default") if obj.vector else None
        if vector:
            vectors.append(np.array(vector))
            metadata_list.append({
                'id': str(obj.uuid),
                'properties': obj.properties
            })
    
    return vectors, metadata_list

def generate_tsne_visualization(client, n_components: int = 2, perplexity: float = 30.0) -> Dict[str, Any]:
    """
    Generate t-SNE visualization data for conceptual vector clustering.
    
    Args:
        client: Weaviate client
        n_components: Number of dimensions (2 for 2D plot)
        perplexity: t-SNE perplexity parameter
        
    Returns:
        Dictionary with visualization data including coordinates and metadata
    """
    vectors, metadata_list = get_all_experience_vectors(client)
    
    if not vectors:
        return {'error': 'No vectors found in database'}
    
    vectors_array = np.array(vectors)
    
    # Use t-SNE if available, otherwise try UMAP, otherwise return error
    if TSNE_AVAILABLE:
        try:
            # Adjust perplexity if we have fewer samples than recommended
            n_samples = len(vectors)
            adjusted_perplexity = min(perplexity, max(5, n_samples - 1))
            # Create TSNE with basic parameters (avoid version-specific parameters)
            reducer = TSNE(
                n_components=n_components, 
                perplexity=adjusted_perplexity, 
                random_state=42, 
                init='pca',
                learning_rate='auto'
            )
            coordinates = reducer.fit_transform(vectors_array)
        except Exception as e:
            return {'error': f't-SNE computation failed: {str(e)}. Try refreshing the page or check server logs.'}
    elif UMAP_AVAILABLE:
        try:
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            coordinates = reducer.fit_transform(vectors_array)
        except Exception as e:
            return {'error': f'UMAP computation failed: {str(e)}'}
    else:
        return {'error': 'Neither t-SNE nor UMAP available. Install scikit-learn (pip install scikit-learn) for t-SNE visualization.'}
    
    # Prepare data for visualization
    visualization_data = {
        'type': 'tsne_clustering',
        'coordinates': coordinates.tolist(),
        'points': []
    }
    
    for i, (coord, metadata) in enumerate(zip(coordinates, metadata_list)):
        props = metadata['properties']
        point = {
            'id': metadata['id'],
            'x': float(coord[0]),
            'y': float(coord[1]),
            'text_snippet': props.get('text', '')[:100] + '...' if len(props.get('text', '')) > 100 else props.get('text', ''),
            'tag_abstract': props.get('tag_abstract', 'N/A'),
            'program': props.get('current_program_enrollment', 'N/A'),
            'urban_rural': props.get('urban_rural_designation', 'N/A'),
            'anxiety': props.get('survey_anxiety'),
            'moves': props.get('residential_moves_count', 0)
        }
        visualization_data['points'].append(point)
    
    return visualization_data

def generate_geospatial_heatmap(client, query_text: str = None, tag_filter: str = None) -> Dict[str, Any]:
    """
    Generate geospatial heatmap data for barrier friction analysis.
    
    Args:
        client: Weaviate client
        query_text: Optional semantic query to filter experiences
        tag_filter: Optional tag_abstract filter
        
    Returns:
        Dictionary with location-based aggregation data
    """
    collection = client.collections.get(CLASS_NAME)
    
    # Build filter if tag is specified
    filter_chain = None
    if tag_filter:
        filter_chain = wvq.Filter.by_property("tag_abstract").equal(tag_filter)
    
    # If query text provided, get matching experiences
    if query_text:
        # This would need query vector generation - simplified for now
        # In full implementation, would use semantic search first
        pass
    
    # Fetch all experiences with location data
    result = collection.query.fetch_objects(
        limit=1000,
        filters=filter_chain if filter_chain else None,
        return_properties=[
            "tag_abstract", "service_office_location", "urban_rural_designation",
            "current_program_enrollment", "text"
        ]
    )
    
    # Aggregate by location
    location_counts = {}
    urban_rural_counts = {'Urban': 0, 'Suburban': 0, 'Rural': 0}
    
    for obj in result.objects:
        props = obj.properties
        location = props.get('service_office_location', 'Unknown')
        urban_rural = props.get('urban_rural_designation', 'Unknown')
        tag = props.get('tag_abstract', 'Uncategorized')
        
        # Count by location
        if location not in location_counts:
            location_counts[location] = {
                'count': 0,
                'tags': {},
                'urban_rural': urban_rural
            }
        location_counts[location]['count'] += 1
        if tag not in location_counts[location]['tags']:
            location_counts[location]['tags'][tag] = 0
        location_counts[location]['tags'][tag] += 1
        
        # Count by urban/rural
        if urban_rural in urban_rural_counts:
            urban_rural_counts[urban_rural] += 1
    
    return {
        'type': 'geospatial_heatmap',
        'locations': location_counts,
        'urban_rural_breakdown': urban_rural_counts,
        'total_experiences': len(result.objects)
    }

def generate_geographic_map(client, tag_filter: str = None) -> Dict[str, Any]:
    """
    Generate geographic map visualization data for US locations with issue category analysis.
    
    Args:
        client: Weaviate client
        tag_filter: Optional tag_abstract filter
        
    Returns:
        Dictionary with geographic data including city, state, and issue category scores
    """
    collection = client.collections.get(CLASS_NAME)
    
    # Build filter if tag is specified (supports partial matching for issue categories)
    filter_chain = None
    if tag_filter:
        # Support both exact tag matches and category-based filtering
        filter_chain = wvq.Filter.by_property("tag_abstract").contains_any([tag_filter])
    
    # Fetch all experiences with geographic data
    result = collection.query.fetch_objects(
        limit=1000,
        filters=filter_chain if filter_chain else None,
        return_properties=[
            "tag_abstract", "city", "state", "zip_code", "text",
            "current_program_enrollment", "survey_anxiety", 
            "survey_control", "survey_hope", "residential_moves_count",
            "financial_volatility_index"
        ]
    )
    
    # Issue category keywords for classification
    issue_categories = {
        'Cognitive Load': ['cognitive load', 'administrative', 'paperwork', 'overwhelm', 'complexity', 'confusion', 'buried'],
        'Dignity Deprivation': ['dignity', 'humiliation', 'shame', 'judgment', 'stigma', 'dehumanization', 'embarrassed'],
        'Systemic Frustration': ['systemic', 'frustration', 'barrier', 'bureaucracy', 'red tape', 'government', 'system'],
        'Housing Insecurity': ['housing', 'eviction', 'homeless', 'shelter', 'housing insecurity', 'move', 'landlord', 'residential'],
        'Time Poverty': ['time poverty', 'no time', 'busy', 'rushed', 'waiting', 'time constraint', 'skip', 'schedule'],
        'Financial Stress': ['debt', 'money', 'financial', 'bills', 'poverty', 'economic', 'can\'t afford', 'volatility'],
        'Social Isolation': ['isolation', 'alone', 'lonely', 'no support', 'isolated', 'social isolation', 'no one to talk'],
        'Anxiety': ['anxiety', 'worry', 'fear', 'panic', 'stress', 'overwhelmed', 'nervous'],
        'Healthcare Barrier': ['healthcare', 'doctor', 'clinic', 'medical', 'appointment', 'health', 'treatment', 'access'],
        'Community Support': ['community', 'support', 'church', 'food bank', 'pantry', 'help', 'assistance', 'network']
    }
    
    # Aggregate by location with issue category scoring
    location_data = {}
    state_counts = {}
    
    for obj in result.objects:
        props = obj.properties
        city = props.get('city', '').strip()
        state = props.get('state', '').strip()
        tag = props.get('tag_abstract', 'Uncategorized')
        text = props.get('text', '').lower()
        combined_text = f"{text} {tag.lower()}"
        
        if not city or not state:
            continue
        
        # Create location key
        location_key = f"{city}, {state}"
        
        if location_key not in location_data:
            location_data[location_key] = {
                'city': city,
                'state': state,
                'count': 0,
                'tags': {},
                'issue_scores': {cat: 0 for cat in issue_categories.keys()},
                'programs': {},
                'avg_anxiety': [],
                'avg_control': [],
                'avg_hope': []
            }
        
        location_data[location_key]['count'] += 1
        
        # Track tags
        if tag not in location_data[location_key]['tags']:
            location_data[location_key]['tags'][tag] = 0
        location_data[location_key]['tags'][tag] += 1
        
        # Score issue categories
        for category, keywords in issue_categories.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            location_data[location_key]['issue_scores'][category] += score
        
        # Track programs
        program = props.get('current_program_enrollment', 'None')
        if program not in location_data[location_key]['programs']:
            location_data[location_key]['programs'][program] = 0
        location_data[location_key]['programs'][program] += 1
        
        # Track survey metrics
        anxiety = props.get('survey_anxiety')
        if anxiety is not None:
            location_data[location_key]['avg_anxiety'].append(anxiety)
        control = props.get('survey_control')
        if control is not None:
            location_data[location_key]['avg_control'].append(control)
        hope = props.get('survey_hope')
        if hope is not None:
            location_data[location_key]['avg_hope'].append(hope)
        
        # State-level aggregation
        if state not in state_counts:
            state_counts[state] = {
                'count': 0,
                'issue_scores': {cat: 0 for cat in issue_categories.keys()},
                'tags': {},
                'avg_anxiety': [],
                'avg_control': [],
                'avg_hope': []
            }
        
        state_counts[state]['count'] += 1
        
        # Aggregate issue scores at state level
        for category, keywords in issue_categories.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            state_counts[state]['issue_scores'][category] += score
        
        # Track tags at state level
        if tag not in state_counts[state]['tags']:
            state_counts[state]['tags'][tag] = 0
        state_counts[state]['tags'][tag] += 1
        
        # Track survey metrics at state level
        if anxiety is not None:
            state_counts[state]['avg_anxiety'].append(anxiety)
        if control is not None:
            state_counts[state]['avg_control'].append(control)
        if hope is not None:
            state_counts[state]['avg_hope'].append(hope)
    
    # Calculate averages
    for location_key in location_data:
        if location_data[location_key]['avg_anxiety']:
            location_data[location_key]['avg_anxiety'] = sum(location_data[location_key]['avg_anxiety']) / len(location_data[location_key]['avg_anxiety'])
        else:
            location_data[location_key]['avg_anxiety'] = None
    
    for state in state_counts:
        if state_counts[state]['avg_anxiety']:
            state_counts[state]['avg_anxiety'] = sum(state_counts[state]['avg_anxiety']) / len(state_counts[state]['avg_anxiety'])
        if state_counts[state]['avg_control']:
            state_counts[state]['avg_control'] = sum(state_counts[state]['avg_control']) / len(state_counts[state]['avg_control'])
        if state_counts[state]['avg_hope']:
            state_counts[state]['avg_hope'] = sum(state_counts[state]['avg_hope']) / len(state_counts[state]['avg_hope'])
    
    return {
        'type': 'geographic_map',
        'locations': location_data,
        'state_counts': state_counts,
        'total_experiences': len(result.objects),
        'locations_with_data': len(location_data),
        'issue_categories': list(issue_categories.keys())
    }

def generate_correlation_plot(client, x_axis: str = 'survey_anxiety', y_axis: str = 'residential_moves_count') -> Dict[str, Any]:
    """
    Generate correlation plot data comparing subjective vs objective measures.
    
    Args:
        client: Weaviate client
        x_axis: Metadata field for X-axis (e.g., 'survey_anxiety', 'survey_control', 'survey_hope')
        y_axis: Metadata field for Y-axis (e.g., 'residential_moves_count', 'financial_volatility_index')
        
    Returns:
        Dictionary with scatter plot data
    """
    collection = client.collections.get(CLASS_NAME)
    
    result = collection.query.fetch_objects(
        limit=1000,
        return_properties=[
            x_axis, y_axis, "tag_abstract", "text", "current_program_enrollment",
            "urban_rural_designation", "religious_participation"
        ]
    )
    
    points = []
    for obj in result.objects:
        props = obj.properties
        x_value = props.get(x_axis)
        y_value = props.get(y_axis)
        
        # Skip if either value is missing
        if x_value is None or y_value is None:
            continue
        
        # Convert to numeric if needed
        try:
            x_val = float(x_value) if x_value is not None else None
            y_val = float(y_value) if y_value is not None else None
        except (ValueError, TypeError):
            continue
        
        if x_val is not None and y_val is not None:
            point = {
                'x': x_val,
                'y': y_val,
                'tag_abstract': props.get('tag_abstract', 'N/A'),
                'text_snippet': props.get('text', '')[:80] + '...' if len(props.get('text', '')) > 80 else props.get('text', ''),
                'program': props.get('current_program_enrollment', 'N/A'),
                'urban_rural': props.get('urban_rural_designation', 'N/A'),
                'religiosity': props.get('religious_participation', 'N/A')
            }
            points.append(point)
    
    return {
        'type': 'correlation_plot',
        'x_axis': x_axis,
        'y_axis': y_axis,
        'points': points,
        'total_points': len(points)
    }

# Example usage
if __name__ == "__main__":
    try:
        client = weaviate.connect_to_local(port=8080, grpc_port=50051)
        
        print("Generating t-SNE visualization...")
        tsne_data = generate_tsne_visualization(client)
        print(f"t-SNE: {len(tsne_data.get('points', []))} points")
        
        print("\nGenerating geospatial heatmap...")
        geo_data = generate_geospatial_heatmap(client, tag_filter="Cognitive Load")
        print(f"Geospatial: {geo_data.get('total_experiences', 0)} experiences across {len(geo_data.get('locations', {}))} locations")
        
        print("\nGenerating correlation plot...")
        corr_data = generate_correlation_plot(client, 'survey_anxiety', 'residential_moves_count')
        print(f"Correlation: {corr_data.get('total_points', 0)} points")
        
        client.close()
    except Exception as e:
        print(f"Error: {e}")

