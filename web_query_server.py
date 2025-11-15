#!/usr/bin/env python3
"""
Flask web server for Social Services Experience Analytics Platform query interface
Provides REST API endpoints for querying Weaviate and NLM integration
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import weaviate
import weaviate.classes.query as wvq
import numpy as np
import os

# Import NLM integration
try:
    from nlm_integration import get_nlm_enrichment_for_experience, enrich_experience_with_nlm
    NLM_AVAILABLE = True
except ImportError:
    NLM_AVAILABLE = False
    print("Warning: NLM integration module not available. Install requests: pip install requests")

# Try to import CLIP for query vector generation
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Queries will use Weaviate's default text vectorizer.")

app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for frontend

CLASS_NAME = "ClientExperience"

# CLIP Model for queries (loaded once)
_query_clip_model = None
_query_clip_processor = None

def get_query_clip_model():
    """Load CLIP model for query vector generation"""
    global _query_clip_model, _query_clip_processor
    if _query_clip_model is None and CLIP_AVAILABLE:
        print("Loading CLIP model for queries...")
        _query_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _query_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _query_clip_model.eval()
        if torch.cuda.is_available():
            _query_clip_model = _query_clip_model.cuda()
        print("CLIP model loaded.")
    return _query_clip_model, _query_clip_processor

def generate_query_vector(query_text: str) -> list:
    """Generate CLIP embedding for query text"""
    if not CLIP_AVAILABLE:
        return None
    
    try:
        model, processor = get_query_clip_model()
        if model is None:
            return None
        
        inputs = processor(text=[query_text], return_tensors="pt", padding=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            model = model.cuda()
        
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            text_emb = outputs[0].cpu().numpy()
        
        text_emb = text_emb / np.linalg.norm(text_emb)
        return text_emb.tolist()
    except Exception as e:
        print(f"Warning: Could not generate query vector: {e}")
        return None

def connect_to_weaviate():
    """Connect to Weaviate instance"""
    try:
        client = weaviate.connect_to_local(port=8080, grpc_port=50051)
        return client
    except Exception as e:
        raise ConnectionError(f"Cannot connect to Weaviate: {e}")

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'query_interface.html')

def apply_vector_reweighting(query_vector, clip_weight, audio_weight, timeseries_weight):
    """
    Apply query-time modality re-weighting to the query vector.
    Since we're querying against fused vectors, we need to adjust the query vector
    to match the new weights. This is a simplified approach - in practice, you'd
    need to regenerate the query vector with different weights.
    """
    # For now, we'll use the query vector as-is since we can't easily re-weight
    # a single query vector. In a full implementation, you'd regenerate embeddings
    # with different weights. This is a placeholder for the feature.
    return query_vector

def apply_negative_filtering(query_vector, exclude_text):
    """
    Apply conceptual exclusion (negative filtering).
    Calculates query_vector - exclude_vector to find items close to A but far from B.
    """
    if not exclude_text or not CLIP_AVAILABLE:
        return query_vector
    
    try:
        exclude_vector = generate_query_vector(exclude_text)
        if exclude_vector:
            # Subtract the exclusion vector from the query vector
            query_vec = np.array(query_vector)
            exclude_vec = np.array(exclude_vector)
            # Normalize both before subtraction
            query_vec = query_vec / np.linalg.norm(query_vec)
            exclude_vec = exclude_vec / np.linalg.norm(exclude_vec)
            # Subtract and renormalize
            result_vec = query_vec - exclude_vec
            result_vec = result_vec / np.linalg.norm(result_vec)
            return result_vec.tolist()
    except Exception as e:
        print(f"Warning: Negative filtering failed: {e}")
    
    return query_vector

@app.route('/api/search', methods=['POST'])
def search():
    """Handle semantic search queries with advanced filters and capabilities"""
    try:
        data = request.json
        query_text = data.get('query', '').strip()
        exclude_text = data.get('excludeQuery', '').strip()  # Negative filtering
        filter_religiosity = data.get('filterReligiosity', 'ALL')
        filter_anxiety = data.get('filterAnxiety', 'ALL')
        filter_anxiety_min = data.get('filterAnxietyMin', None)  # Range query
        filter_anxiety_max = data.get('filterAnxietyMax', None)  # Range query
        filter_program = data.get('filterProgram', 'ALL')
        filter_enrollment_status = data.get('filterEnrollmentStatus', 'ALL')
        filter_urban_rural = data.get('filterUrbanRural', 'ALL')
        filter_time_of_day = data.get('filterTimeOfDay', 'ALL')
        filter_language = data.get('filterLanguage', 'ALL')
        filter_moves_min = data.get('filterMovesMin', None)
        filter_moves_max = data.get('filterMovesMax', None)
        limit = data.get('limit', 5)
        
        # Modality re-weighting (for future implementation)
        clip_weight = data.get('clipWeight', 0.6)
        audio_weight = data.get('audioWeight', 0.15)
        timeseries_weight = data.get('timeseriesWeight', 0.15)
        
        if not query_text:
            return jsonify({'error': 'Query text is required'}), 400
        
        # Connect to Weaviate
        client = connect_to_weaviate()
        collection = client.collections.get(CLASS_NAME)
        
        # Build filter chain
        filter_chain = None
        
        # Existing filters
        if filter_religiosity and filter_religiosity != 'ALL':
            religiosity_filter = wvq.Filter.by_property("religious_participation").equal(filter_religiosity)
            filter_chain = religiosity_filter if filter_chain is None else filter_chain & religiosity_filter
        
        # Range query for anxiety (two-dimensional filtering)
        if filter_anxiety_min is not None or filter_anxiety_max is not None:
            if filter_anxiety_min is not None and filter_anxiety_max is not None:
                anxiety_filter = wvq.Filter.by_property("survey_anxiety").greater_or_equal(filter_anxiety_min) & \
                                wvq.Filter.by_property("survey_anxiety").less_or_equal(filter_anxiety_max)
            elif filter_anxiety_min is not None:
                anxiety_filter = wvq.Filter.by_property("survey_anxiety").greater_or_equal(filter_anxiety_min)
            else:
                anxiety_filter = wvq.Filter.by_property("survey_anxiety").less_or_equal(filter_anxiety_max)
            filter_chain = anxiety_filter if filter_chain is None else filter_chain & anxiety_filter
        elif filter_anxiety and filter_anxiety != 'ALL':
            # Legacy single-value filter
            if filter_anxiety == 'low':
                anxiety_filter = wvq.Filter.by_property("survey_anxiety").less_or_equal(2)
            elif filter_anxiety == 'high':
                anxiety_filter = wvq.Filter.by_property("survey_anxiety").greater_or_equal(4)
            else:
                anxiety_filter = None
            if anxiety_filter:
                filter_chain = anxiety_filter if filter_chain is None else filter_chain & anxiety_filter
        
        # Policy-driven metadata filters
        if filter_program and filter_program != 'ALL':
            program_filter = wvq.Filter.by_property("current_program_enrollment").equal(filter_program)
            filter_chain = program_filter if filter_chain is None else filter_chain & program_filter
        
        if filter_enrollment_status and filter_enrollment_status != 'ALL':
            status_filter = wvq.Filter.by_property("enrollment_status_change").equal(filter_enrollment_status)
            filter_chain = status_filter if filter_chain is None else filter_chain & status_filter
        
        if filter_urban_rural and filter_urban_rural != 'ALL':
            urban_rural_filter = wvq.Filter.by_property("urban_rural_designation").equal(filter_urban_rural)
            filter_chain = urban_rural_filter if filter_chain is None else filter_chain & urban_rural_filter
        
        if filter_time_of_day and filter_time_of_day != 'ALL':
            time_filter = wvq.Filter.by_property("incident_time_of_day").equal(filter_time_of_day)
            filter_chain = time_filter if filter_chain is None else filter_chain & time_filter
        
        if filter_language and filter_language != 'ALL':
            lang_filter = wvq.Filter.by_property("primary_language").equal(filter_language)
            filter_chain = lang_filter if filter_chain is None else filter_chain & lang_filter
        
        # Range query for residential moves
        if filter_moves_min is not None or filter_moves_max is not None:
            if filter_moves_min is not None and filter_moves_max is not None:
                moves_filter = wvq.Filter.by_property("residential_moves_count").greater_or_equal(filter_moves_min) & \
                              wvq.Filter.by_property("residential_moves_count").less_or_equal(filter_moves_max)
            elif filter_moves_min is not None:
                moves_filter = wvq.Filter.by_property("residential_moves_count").greater_or_equal(filter_moves_min)
            else:
                moves_filter = wvq.Filter.by_property("residential_moves_count").less_or_equal(filter_moves_max)
            filter_chain = moves_filter if filter_chain is None else filter_chain & moves_filter
        
        # Generate query vector
        query_vector = generate_query_vector(query_text)
        
        # Apply negative filtering (conceptual exclusion) if specified
        if exclude_text:
            query_vector = apply_negative_filtering(query_vector, exclude_text)
        
        # Apply modality re-weighting (placeholder for future implementation)
        if clip_weight != 0.6 or audio_weight != 0.15 or timeseries_weight != 0.15:
            query_vector = apply_vector_reweighting(query_vector, clip_weight, audio_weight, timeseries_weight)
        
        # Perform search
        if query_vector:
            # Try Weaviate's near_vector first
            try:
                if filter_chain:
                    result = collection.query.near_vector(
                        near_vector=query_vector,
                        limit=limit,
                        filters=filter_chain,
                        return_metadata=wvq.MetadataQuery(distance=True),
                        return_properties=["text", "tag_abstract", "ed_level_primary", 
                                         "religious_participation", "survey_anxiety", 
                                         "survey_control", "survey_hope", "time_series_data"]
                    )
                else:
                    result = collection.query.near_vector(
                        near_vector=query_vector,
                        limit=limit,
                        return_metadata=wvq.MetadataQuery(distance=True),
                        return_properties=["text", "tag_abstract", "ed_level_primary",
                                         "religious_participation", "survey_anxiety",
                                         "survey_control", "survey_hope", "time_series_data"]
                    )
                
                # If Weaviate returns no results, use manual distance calculation
                if len(result.objects) == 0:
                    raise ValueError("No results from Weaviate, using manual calculation")
                    
            except Exception as e:
                # Fallback: Manual distance calculation
                print(f"Using manual distance calculation: {e}")
                import numpy as np
                
                # Fetch all objects with vectors
                all_objects = collection.query.fetch_objects(
                    limit=1000,
                    include_vector=True,
                    filters=filter_chain if filter_chain else None,
                    return_properties=["text", "tag_abstract", "ed_level_primary", 
                                     "religious_participation", "survey_anxiety", 
                                     "survey_control", "survey_hope", "time_series_data",
                                     "current_program_enrollment", "enrollment_status_change",
                                     "document_submission_success_rate", "service_office_location",
                                     "urban_rural_designation", "incident_time_of_day",
                                     "residential_moves_count", "financial_volatility_index",
                                     "primary_language"]
                )
                
                # Calculate cosine distances manually
                query_vec = np.array(query_vector)
                query_norm = np.linalg.norm(query_vec)
                
                scored_objects = []
                for obj in all_objects.objects:
                    stored_vec = obj.vector.get("default") if obj.vector else None
                    if stored_vec:
                        stored_vec = np.array(stored_vec)
                        stored_norm = np.linalg.norm(stored_vec)
                        
                        # Cosine similarity
                        if query_norm > 0 and stored_norm > 0:
                            cosine_sim = np.dot(query_vec, stored_vec) / (query_norm * stored_norm)
                            cosine_dist = 1 - cosine_sim
                            scored_objects.append((cosine_dist, obj))
                
                # Sort by distance and take top results
                scored_objects.sort(key=lambda x: x[0])
                top_objects = [obj for _, obj in scored_objects[:limit]]
                
                # Create a mock result object
                class MockResult:
                    def __init__(self, objects):
                        self.objects = objects
                
                class MockMetadata:
                    def __init__(self, distance):
                        self.distance = distance
                
                # Create result objects with metadata
                result_objects = []
                for obj, (dist, _) in zip(top_objects, scored_objects[:limit]):
                    obj.metadata = MockMetadata(dist)
                    result_objects.append(obj)
                
                result = MockResult(result_objects)
        else:
            # Fall back to near_text
            if filter_chain:
                result = collection.query.near_text(
                    query=query_text,
                    limit=limit,
                    filters=filter_chain,
                    return_metadata=wvq.MetadataQuery(distance=True),
                    return_properties=["text", "tag_abstract", "ed_level_primary",
                                     "religious_participation", "survey_anxiety",
                                     "survey_control", "survey_hope", "time_series_data",
                                     "current_program_enrollment", "enrollment_status_change",
                                     "document_submission_success_rate", "service_office_location",
                                     "urban_rural_designation", "incident_time_of_day",
                                     "residential_moves_count", "financial_volatility_index",
                                     "primary_language"]
                )
            else:
                result = collection.query.near_text(
                    query=query_text,
                    limit=limit,
                    return_metadata=wvq.MetadataQuery(distance=True),
                    return_properties=["text", "tag_abstract", "ed_level_primary",
                                     "religious_participation", "survey_anxiety",
                                     "survey_control", "survey_hope", "time_series_data",
                                     "current_program_enrollment", "enrollment_status_change",
                                     "document_submission_success_rate", "service_office_location",
                                     "urban_rural_designation", "incident_time_of_day",
                                     "residential_moves_count", "financial_volatility_index",
                                     "primary_language"]
                )
        
        # Format results
        results = []
        for obj in result.objects:
            props = obj.properties
            metadata = obj.metadata
            distance = metadata.distance if metadata.distance is not None else 0.0
            
            # Calculate volatility from time-series data if available
            volatility = None
            ts_data = props.get('time_series_data', '')
            if ts_data:
                try:
                    import json
                    data = json.loads(ts_data)
                    if 'value' in data and len(data['value']) > 1:
                        values = np.array(data['value'])
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        if mean_val > 0:
                            cv = std_val / mean_val
                            # Convert to 1-5 scale for display
                            volatility = min(5, max(1, int(cv * 10)))
                except:
                    pass
            
            # Ensure similarity score is between 0 and 1
            similarity = max(0.0, min(1.0, 1 - distance))
            
            metadata_dict = {
                'text_snippet': props.get('text', 'N/A'),
                'subjective_concept': props.get('tag_abstract', 'N/A'),
                'religious_participation': props.get('religious_participation', 'N/A'),
                'survey_anxiety': props.get('survey_anxiety'),
                'survey_control': props.get('survey_control'),
                'survey_hope': props.get('survey_hope'),
                'time_series_volatility': volatility,
                'ed_level_primary': props.get('ed_level_primary', 'N/A'),
                # Policy-driven metadata
                'current_program_enrollment': props.get('current_program_enrollment', 'N/A'),
                'enrollment_status_change': props.get('enrollment_status_change', 'N/A'),
                'document_submission_success_rate': props.get('document_submission_success_rate'),
                'service_office_location': props.get('service_office_location', 'N/A'),
                'urban_rural_designation': props.get('urban_rural_designation', 'N/A'),
                'incident_time_of_day': props.get('incident_time_of_day', 'N/A'),
                'residential_moves_count': props.get('residential_moves_count'),
                'financial_volatility_index': props.get('financial_volatility_index'),
                'primary_language': props.get('primary_language', 'N/A')
            }
            
            # Add NLM enrichment if available and requested
            include_nlm = request.json.get('includeNLM', False) if request.json else False
            if include_nlm and NLM_AVAILABLE:
                try:
                    nlm_enrichment = enrich_experience_with_nlm(
                        text_snippet=metadata_dict['text_snippet'],
                        tag_abstract=metadata_dict['subjective_concept'],
                        survey_anxiety=metadata_dict['survey_anxiety']
                    )
                    metadata_dict['nlm_enrichment'] = nlm_enrichment
                except Exception as e:
                    print(f"Warning: NLM enrichment failed: {e}")
                    metadata_dict['nlm_enrichment'] = None
            
            results.append({
                'id': str(obj.uuid),
                'score': similarity,  # Similarity score (0-1)
                'distance': distance,  # Also include distance for debugging
                'metadata': metadata_dict
            })
        
        client.close()
        
        return jsonify({
            'results': results,
            'count': len(results),
            'query': query_text,
            'filters_applied': {
                'religiosity': filter_religiosity,
                'anxiety': filter_anxiety
            }
        })
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

@app.route('/api/enrich/<experience_id>', methods=['GET'])
def enrich_experience(experience_id):
    """Get NLM enrichment for a specific experience by ID"""
    if not NLM_AVAILABLE:
        return jsonify({'error': 'NLM integration not available'}), 503
    
    try:
        client = connect_to_weaviate()
        collection = client.collections.get(CLASS_NAME)
        
        # Fetch the experience by ID
        result = collection.query.fetch_object_by_id(
            experience_id,
            return_properties=["text", "tag_abstract", "survey_anxiety", 
                             "survey_control", "survey_hope"]
        )
        
        if not result:
            return jsonify({'error': 'Experience not found'}), 404
        
        props = result.properties
        enrichment = enrich_experience_with_nlm(
            text_snippet=props.get('text', ''),
            tag_abstract=props.get('tag_abstract', ''),
            survey_anxiety=props.get('survey_anxiety')
        )
        
        client.close()
        
        return jsonify({
            'experience_id': experience_id,
            'enrichment': enrichment
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Check if Weaviate is connected and has data"""
    try:
        client = connect_to_weaviate()
        collection = client.collections.get(CLASS_NAME)
        
        # Fetch objects to check if data exists and count them
        # Use a high limit to get all objects for counting
        result = collection.query.fetch_objects(limit=1000)
        total_count = len(result.objects)
        has_data = total_count > 0
        
        client.close()
        
        return jsonify({
            'connected': True,
            'total_objects': total_count,
            'has_data': has_data
        })
    except Exception as e:
        return jsonify({
            'connected': False,
            'error': str(e),
            'has_data': False
        }), 500

@app.route('/api/stats', methods=['GET'])
def stats():
    """Get database statistics"""
    try:
        client = connect_to_weaviate()
        collection = client.collections.get(CLASS_NAME)
        
        # Get total count by fetching all objects (up to 1000)
        result = collection.query.fetch_objects(limit=1000)
        total = len(result.objects)
        
        client.close()
        
        return jsonify({
            'total_experiences': total,
            'status': 'connected'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    print("="*80)
    print("Social Services Experience Analytics Platform - Web Query Interface")
    print("="*80)
    print("\nStarting Flask server...")
    print("Open your browser to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

