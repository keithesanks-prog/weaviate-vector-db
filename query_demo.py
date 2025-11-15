"""
Query demonstration script for Social Services Experience Analytics Platform
Demonstrates abstract text search, filtered search, and conceptual distance
"""

import weaviate
import weaviate.classes.query as wvq
import numpy as np
from typing import List, Dict, Any

# Try to import CLIP for query vector generation
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Note: CLIP not available for query vector generation. Using Weaviate's default text vectorizer.")

# CLIP Model for queries (loaded once)
_query_clip_model = None
_query_clip_processor = None

def get_query_clip_model():
    """Load CLIP model for query vector generation"""
    global _query_clip_model, _query_clip_processor
    if _query_clip_model is None and CLIP_AVAILABLE:
        _query_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _query_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _query_clip_model.eval()
        if torch.cuda.is_available():
            _query_clip_model = _query_clip_model.cuda()
    return _query_clip_model, _query_clip_processor

def generate_query_vector(query_text: str) -> list:
    """Generate CLIP embedding for query text"""
    if not CLIP_AVAILABLE:
        return None  # Will fall back to near_text
    
    try:
        model, processor = get_query_clip_model()
        if model is None:
            return None
        
        # Process text only (no image for query)
        inputs = processor(text=[query_text], return_tensors="pt", padding=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            model = model.cuda()
        
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            text_emb = outputs[0].cpu().numpy()
        
        # Normalize
        text_emb = text_emb / np.linalg.norm(text_emb)
        return text_emb.tolist()
    except Exception as e:
        print(f"Warning: Could not generate query vector: {e}")
        return None

# Configuration
CLASS_NAME = "ClientExperience"

def connect_to_weaviate():
    """Connect to Weaviate instance"""
    try:
        client = weaviate.connect_to_local(port=8080, grpc_port=50051)
        return client
    except Exception as e:
        raise ConnectionError(f"Cannot connect to Weaviate: {e}")

def abstract_text_search(client, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Perform abstract text search using CLIP embeddings
    """
    print(f"\n{'='*80}")
    print(f"ABSTRACT TEXT SEARCH")
    print(f"{'='*80}")
    print(f"Query: '{query_text}'")
    print(f"\nTop {limit} most similar experiences:\n")
    
    collection = client.collections.get(CLASS_NAME)
    
    # Try to use CLIP-generated query vector, fall back to near_text
    query_vector = generate_query_vector(query_text)
    
    if query_vector:
        # Use near_vector with CLIP-generated query vector
        result = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_metadata=wvq.MetadataQuery(distance=True),
            return_properties=["text", "tag_abstract", "ed_level_primary", "religious_participation", "image_path"]
        )
    else:
        # Fall back to near_text (uses Weaviate's default text vectorizer)
        result = collection.query.near_text(
            query=query_text,
            limit=limit,
            return_metadata=wvq.MetadataQuery(distance=True, certainty=True),
            return_properties=["text", "tag_abstract", "ed_level_primary", "religious_participation", "image_path"]
        )
    
    for i, obj in enumerate(result.objects, 1):
        props = obj.properties
        metadata = obj.metadata
        distance = metadata.distance if metadata.distance is not None else 0.0
        print(f"{i}. Distance: {distance:.4f}")
        print(f"   Text: {props.get('text', 'N/A')}")
        print(f"   Tag: {props.get('tag_abstract', 'N/A')}")
        print(f"   Education: {props.get('ed_level_primary', 'N/A')}")
        print(f"   Religious Participation: {props.get('religious_participation', 'N/A')}")
        print()
    
    return [{"properties": obj.properties, "metadata": obj.metadata} for obj in result.objects]

def filtered_search(
    client,
    query_text: str,
    filter_property: str,
    filter_value: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform filtered search with sociological context
    """
    print(f"\n{'='*80}")
    print(f"FILTERED SEARCH (Sociological Insight)")
    print(f"{'='*80}")
    print(f"Query: '{query_text}'")
    print(f"Filter: {filter_property} = '{filter_value}'")
    print(f"\nTop {limit} results matching filter:\n")
    
    collection = client.collections.get(CLASS_NAME)
    
    # Generate query vector using CLIP
    query_vector = generate_query_vector(query_text)
    filter_obj = wvq.Filter.by_property(filter_property).equal(filter_value)
    
    if query_vector:
        # Use near_vector with CLIP-generated query vector
        result = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            filters=filter_obj,
            return_metadata=wvq.MetadataQuery(distance=True),
            return_properties=["text", "tag_abstract", "ed_level_primary", "religious_participation", "image_path"]
        )
    else:
        # Fall back to near_text
        result = collection.query.near_text(
            query=query_text,
            limit=limit,
            filters=filter_obj,
            return_metadata=wvq.MetadataQuery(distance=True, certainty=True),
            return_properties=["text", "tag_abstract", "ed_level_primary", "religious_participation", "image_path"]
        )
    
    if not result.objects:
        print("No results found matching the filter criteria.")
        return []
    
    for i, obj in enumerate(result.objects, 1):
        props = obj.properties
        metadata = obj.metadata
        distance = metadata.distance if metadata.distance is not None else 0.0
        print(f"{i}. Distance: {distance:.4f}")
        print(f"   Text: {props.get('text', 'N/A')}")
        print(f"   Tag: {props.get('tag_abstract', 'N/A')}")
        print(f"   Education: {props.get('ed_level_primary', 'N/A')}")
        print(f"   Religious Participation: {props.get('religious_participation', 'N/A')}")
        print()
    
    return [{"properties": obj.properties, "metadata": obj.metadata} for obj in result.objects]

def get_all_experiences_by_tag(client, tag: str) -> List[Dict[str, Any]]:
    """Get all experiences with a specific tag"""
    collection = client.collections.get(CLASS_NAME)
    
    result = collection.query.fetch_objects(
        filters=wvq.Filter.by_property("tag_abstract").equal(tag),
        limit=100,
        include_vector=True,
        return_properties=["text", "tag_abstract"]
    )
    
    return [{"properties": obj.properties, "vector": obj.vector.get("default") if obj.vector else None} for obj in result.objects]

def calculate_conceptual_distance(
    client,
    tag: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Calculate average vector for a tag and find experiences farthest from it
    Demonstrates anti-k-NN search (finding outliers)
    """
    print(f"\n{'='*80}")
    print(f"CONCEPTUAL DISTANCE ANALYSIS (Advanced)")
    print(f"{'='*80}")
    print(f"Tag: '{tag}'")
    print(f"Finding experiences farthest from the average '{tag}' experience...\n")
    
    # Get all experiences with this tag
    tagged_experiences = get_all_experiences_by_tag(client, tag)
    
    if len(tagged_experiences) < 2:
        print(f"Not enough experiences with tag '{tag}' for distance analysis.")
        return []
    
    # Calculate average vector
    vectors = []
    for exp in tagged_experiences:
        vector = exp.get("vector")
        if vector is not None:
            vectors.append(vector)
    
    if not vectors:
        print("Could not extract vectors from experiences.")
        return []
    
    avg_vector = np.mean(vectors, axis=0)
    print(f"Calculated average vector from {len(vectors)} experiences with tag '{tag}'")
    print(f"Vector dimension: {len(avg_vector)}")
    
    # Now find all experiences and calculate distance from average
    collection = client.collections.get(CLASS_NAME)
    
    all_result = collection.query.fetch_objects(
        limit=100,
        include_vector=True,
        return_properties=["text", "tag_abstract", "ed_level_primary", "religious_participation"]
    )
    
    # Calculate distances
    distances = []
    for obj in all_result.objects:
        vector = obj.vector.get("default") if obj.vector else None
        if vector is not None:
            # Calculate cosine distance
            distance = 1 - np.dot(avg_vector, vector) / (np.linalg.norm(avg_vector) * np.linalg.norm(vector))
            distances.append({
                "properties": obj.properties,
                "distance": distance
            })
    
    # Sort by distance (farthest first)
    distances.sort(key=lambda x: x["distance"], reverse=True)
    
    print(f"\nTop {limit} experiences FARTHEST from '{tag}' average:\n")
    
    for i, item in enumerate(distances[:limit], 1):
        props = item["properties"]
        print(f"{i}. Distance from '{tag}' average: {item['distance']:.4f}")
        print(f"   Text: {props.get('text', 'N/A')}")
        print(f"   Tag: {props.get('tag_abstract', 'N/A')}")
        print(f"   Education: {props.get('ed_level_primary', 'N/A')}")
        print(f"   Religious Participation: {props.get('religious_participation', 'N/A')}")
        print()
    
    return [item["properties"] for item in distances[:limit]]

def survey_filtered_search(
    client,
    tag: str,
    survey_anxiety_max: int = None,
    survey_control_min: int = None,
    survey_hope_min: int = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for experiences with a specific tag filtered by survey ratings.
    Demonstrates quantifiable subjectivity filtering.
    """
    print(f"\n{'='*80}")
    print(f"SURVEY RATING FILTER (Quantifiable Subjectivity)")
    print(f"{'='*80}")
    print(f"Tag: '{tag}'")
    filters = []
    if survey_anxiety_max is not None:
        filters.append(f"Anxiety ≤ {survey_anxiety_max}")
    if survey_control_min is not None:
        filters.append(f"Control ≥ {survey_control_min}")
    if survey_hope_min is not None:
        filters.append(f"Hope ≥ {survey_hope_min}")
    print(f"Filters: {', '.join(filters) if filters else 'None'}")
    print(f"\nTop {limit} results:\n")
    
    collection = client.collections.get(CLASS_NAME)
    
    # Build filter chain
    filter_chain = wvq.Filter.by_property("tag_abstract").equal(tag)
    
    if survey_anxiety_max is not None:
        filter_chain = filter_chain & wvq.Filter.by_property("survey_anxiety").less_or_equal(survey_anxiety_max)
    if survey_control_min is not None:
        filter_chain = filter_chain & wvq.Filter.by_property("survey_control").greater_or_equal(survey_control_min)
    if survey_hope_min is not None:
        filter_chain = filter_chain & wvq.Filter.by_property("survey_hope").greater_or_equal(survey_hope_min)
    
    result = collection.query.fetch_objects(
        filters=filter_chain,
        limit=limit,
        return_properties=["text", "tag_abstract", "ed_level_primary", "religious_participation", 
                          "survey_anxiety", "survey_control", "survey_hope"]
    )
    
    if not result.objects:
        print("No results found matching the filter criteria.")
        return []
    
    for i, obj in enumerate(result.objects, 1):
        props = obj.properties
        print(f"{i}. Text: {props.get('text', 'N/A')}")
        print(f"   Tag: {props.get('tag_abstract', 'N/A')}")
        print(f"   Education: {props.get('ed_level_primary', 'N/A')}")
        print(f"   Religious Participation: {props.get('religious_participation', 'N/A')}")
        print(f"   Survey Ratings - Anxiety: {props.get('survey_anxiety', 'N/A')}, "
              f"Control: {props.get('survey_control', 'N/A')}, "
              f"Hope: {props.get('survey_hope', 'N/A')}")
        print()
    
    return [{"properties": obj.properties} for obj in result.objects]

def advanced_multi_modal_query(
    client,
    query_text: str,
    tag: str = None,
    survey_anxiety_max: int = None,
    survey_control_min: int = None,
    survey_hope_min: int = None,
    high_volatility: bool = False,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Advanced query combining semantic search with survey and time-series filters.
    
    Example: "Find narratives about spiritual hope, but only where financial 
    volatility is high and the self-reported sense of control is low."
    
    Args:
        query_text: Semantic search query
        tag: Optional tag filter
        survey_anxiety_max: Maximum anxiety rating (1-5)
        survey_control_min: Minimum control rating (1-5)
        survey_hope_min: Minimum hope rating (1-5)
        high_volatility: If True, filter for high financial volatility (based on time-series)
        limit: Maximum number of results
    """
    print(f"\n{'='*80}")
    print(f"ADVANCED MULTI-MODAL QUERY")
    print(f"{'='*80}")
    print(f"Semantic Query: '{query_text}'")
    
    filters = []
    if tag:
        filters.append(f"Tag: {tag}")
    if survey_anxiety_max is not None:
        filters.append(f"Anxiety ≤ {survey_anxiety_max}")
    if survey_control_min is not None:
        filters.append(f"Control ≥ {survey_control_min}")
    if survey_hope_min is not None:
        filters.append(f"Hope ≥ {survey_hope_min}")
    if high_volatility:
        filters.append("High Financial Volatility")
    
    print(f"Filters: {', '.join(filters) if filters else 'None'}")
    print(f"\nTop {limit} results:\n")
    
    collection = client.collections.get(CLASS_NAME)
    
    # Build filter chain
    filter_chain = None
    
    if tag:
        filter_chain = wvq.Filter.by_property("tag_abstract").equal(tag)
    
    if survey_anxiety_max is not None:
        anxiety_filter = wvq.Filter.by_property("survey_anxiety").less_or_equal(survey_anxiety_max)
        filter_chain = filter_chain & anxiety_filter if filter_chain else anxiety_filter
    
    if survey_control_min is not None:
        control_filter = wvq.Filter.by_property("survey_control").greater_or_equal(survey_control_min)
        filter_chain = filter_chain & control_filter if filter_chain else control_filter
    
    if survey_hope_min is not None:
        hope_filter = wvq.Filter.by_property("survey_hope").greater_or_equal(survey_hope_min)
        filter_chain = filter_chain & hope_filter if filter_chain else hope_filter
    
    # For high volatility, we need to calculate it from time-series data
    # This is a simplified approach - in production, you'd pre-calculate volatility
    # For now, we'll filter by experiences with high standard deviation in time-series
    # Note: This requires fetching all results and filtering, which is not ideal
    # A better approach would be to pre-calculate volatility and store it as a property
    
    # Generate query vector using CLIP
    query_vector = generate_query_vector(query_text)
    
    # Perform semantic search with filters
    if query_vector:
        # Use near_vector with CLIP-generated query vector
        if filter_chain:
            result = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit * 2,  # Get more results to filter for volatility
                filters=filter_chain,
                return_metadata=wvq.MetadataQuery(distance=True),
                return_properties=["text", "tag_abstract", "ed_level_primary", "religious_participation",
                                 "survey_anxiety", "survey_control", "survey_hope", "time_series_data"]
            )
        else:
            result = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit * 2,
                return_metadata=wvq.MetadataQuery(distance=True),
                return_properties=["text", "tag_abstract", "ed_level_primary", "religious_participation",
                                 "survey_anxiety", "survey_control", "survey_hope", "time_series_data"]
            )
    else:
        # Fall back to near_text
        if filter_chain:
            result = collection.query.near_text(
                query=query_text,
                limit=limit * 2,
                filters=filter_chain,
                return_metadata=wvq.MetadataQuery(distance=True),
                return_properties=["text", "tag_abstract", "ed_level_primary", "religious_participation",
                                 "survey_anxiety", "survey_control", "survey_hope", "time_series_data"]
            )
        else:
            result = collection.query.near_text(
                query=query_text,
                limit=limit * 2,
                return_metadata=wvq.MetadataQuery(distance=True),
                return_properties=["text", "tag_abstract", "ed_level_primary", "religious_participation",
                                 "survey_anxiety", "survey_control", "survey_hope", "time_series_data"]
            )
    
    # Filter by volatility if requested
    if high_volatility and result.objects:
        import json
        filtered_objects = []
        for obj in result.objects:
            ts_data = obj.properties.get('time_series_data', '')
            if ts_data:
                try:
                    data = json.loads(ts_data)
                    if 'value' in data and len(data['value']) > 1:
                        values = np.array(data['value'])
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        # High volatility = coefficient of variation > 0.15
                        if mean_val > 0:
                            cv = std_val / mean_val
                            if cv > 0.15:  # High volatility threshold
                                filtered_objects.append(obj)
                except:
                    pass
        result.objects = filtered_objects[:limit]
    else:
        result.objects = result.objects[:limit]
    
    if not result.objects:
        print("No results found matching all filter criteria.")
        return []
    
    for i, obj in enumerate(result.objects, 1):
        props = obj.properties
        metadata = obj.metadata
        distance = metadata.distance if metadata.distance is not None else 0.0
        
        # Calculate volatility if time-series data exists
        volatility_info = ""
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
                        volatility_info = f" (Volatility: {cv:.3f})"
            except:
                pass
        
        print(f"{i}. Distance: {distance:.4f}{volatility_info}")
        print(f"   Text: {props.get('text', 'N/A')}")
        print(f"   Tag: {props.get('tag_abstract', 'N/A')}")
        print(f"   Education: {props.get('ed_level_primary', 'N/A')}")
        print(f"   Religious Participation: {props.get('religious_participation', 'N/A')}")
        print(f"   Survey - Anxiety: {props.get('survey_anxiety', 'N/A')}, "
              f"Control: {props.get('survey_control', 'N/A')}, "
              f"Hope: {props.get('survey_hope', 'N/A')}")
        print()
    
    return [{"properties": obj.properties, "metadata": obj.metadata} for obj in result.objects]

def demonstrate_queries():
    """Demonstrate all query capabilities"""
    print("\n" + "="*80)
    print("SOCIAL SERVICES EXPERIENCE ANALYTICS PLATFORM - QUERY DEMONSTRATIONS")
    print("="*80)
    
    # Connect to Weaviate
    try:
        client = connect_to_weaviate()
        print("✓ Connected to Weaviate successfully")
    except Exception as e:
        print(f"✗ Error connecting to Weaviate: {e}")
        print("\nMake sure Weaviate is running:")
        print("  docker-compose up -d")
        return
    
    try:
        # Demonstration 1: Abstract Text Search
        abstract_text_search(
            client,
            "What does it look like when hope conflicts with reality?",
            limit=5
        )
        
        # Demonstration 2: Filtered Search
        filtered_search(
            client,
            "The quiet strength of community support",
            filter_property="religious_participation",
            filter_value="High (Weekly)",
            limit=5
        )
        
        # Demonstration 3: Conceptual Distance
        calculate_conceptual_distance(
            client,
            tag="Spiritual Resilience",
            limit=5
        )
        
        # Demonstration 4: Survey Rating Filter
        survey_filtered_search(
            client,
            "Spiritual Resilience",
            survey_anxiety_max=2,
            limit=5
        )
        
        # Demonstration 5: Advanced Multi-Modal Query
        # "Find narratives about spiritual hope, but only where financial 
        # volatility is high and the self-reported sense of control is low."
        advanced_multi_modal_query(
            client,
            query_text="spiritual hope and resilience in difficult times",
            tag="Spiritual Resilience",
            survey_control_min=None,  # We want LOW control, so we'll filter manually
            survey_hope_min=3,  # At least some hope
            high_volatility=True,  # High financial volatility
            limit=5
        )
        
        # Another advanced example: Low control, high anxiety
        print("\n" + "="*80)
        print("Additional Advanced Query Example:")
        print("="*80)
        advanced_multi_modal_query(
            client,
            query_text="struggling with financial instability and uncertainty",
            survey_anxiety_max=5,  # Any anxiety level
            survey_control_min=None,  # We want to find low control
            high_volatility=True,
            limit=3
        )
        
        print("\n" + "="*80)
        print("Query demonstrations complete!")
        print("="*80)
    finally:
        client.close()

if __name__ == "__main__":
    demonstrate_queries()

