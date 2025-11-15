import numpy as np
import uuid
import os
import json
import time

# --- WEAVIATE IMPLEMENTATION ---
import weaviate
import weaviate.classes.config as wvcc
import weaviate.classes.query as wvq

CLASS_NAME = "SubjectiveExperience"

def get_weaviate_client():
    """Initializes and returns a Weaviate client instance using v4 API."""
    try:
        # Connect to local Weaviate instance
        client = weaviate.connect_to_local(port=8080, grpc_port=50051)
        print("Weaviate Client connected successfully.")
        return client
    except Exception as e:
        print(f"Error connecting to Weaviate: {e} (Using Mock Index)")
        return None

def create_weaviate_schema(client):
    """Defines the data class for Weaviate using v4 API with manual vectorization."""
    # Check if class already exists and delete it
    try:
        client.collections.delete(CLASS_NAME)
        print(f"Class {CLASS_NAME} already exists. Deleting it...")
        time.sleep(1)  # Wait for deletion to complete
    except:
        pass  # Class doesn't exist, which is fine
    
    # Create schema with manual vectorization (we provide vectors)
    client.collections.create(
        name=CLASS_NAME,
        description="Social services client experiences with manual embeddings",
        vectorizer_config=None,  # Disable automatic vectorization
        properties=[
            wvcc.Property(
                name="text_snippet",
                data_type=wvcc.DataType.TEXT,
                description="Evocative narrative text snippet"
            ),
            wvcc.Property(
                name="subjective_concept",
                data_type=wvcc.DataType.TEXT,
                description="Abstract concept tag"
            ),
            wvcc.Property(
                name="religious_participation",
                data_type=wvcc.DataType.TEXT,
                description="Religious participation level"
            ),
        ]
    )
    print(f"Schema for class '{CLASS_NAME}' created.")

def weaviate_upsert(client, vectors_to_upsert):
    """Inserts data objects into the live Weaviate index with manual vectors."""
    print(f"\n[WEAVIATE] Inserting {len(vectors_to_upsert)} data objects...")
    collection = client.collections.get(CLASS_NAME)
    
    with collection.batch.dynamic() as batch:
        for vector_id, vector, metadata in vectors_to_upsert:
            batch.add_object(
                properties={
                    "text_snippet": metadata['text_snippet'],
                    "subjective_concept": metadata['subjective_concept'],
                    "religious_participation": metadata['religious_participation'],
                },
                vector=vector  # Provide the manually generated vector
            )
    print("[WEAVIATE] Ingestion complete.")
    
def weaviate_query(client, query_vector, top_k=3):
    """Performs a semantic query using Weaviate's near_vector search with fallback."""
    print(f"\n[WEAVIATE] Performing semantic search (k={top_k}) with near_vector...")
    
    collection = client.collections.get(CLASS_NAME)
    
    # Try Weaviate's near_vector first
    try:
        result = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=wvq.MetadataQuery(distance=True),
            return_properties=["text_snippet", "subjective_concept", "religious_participation"]
        )
        
        # If Weaviate returns no results, use manual distance calculation
        if len(result.objects) == 0:
            raise ValueError("No results from Weaviate, using manual calculation")
        
        # Reformat the results to match the mock output structure
        matches = []
        for obj in result.objects:
            props = obj.properties
            metadata_obj = obj.metadata
            distance = metadata_obj.distance if metadata_obj.distance is not None else 1.0
            
            matches.append({
                'id': str(obj.uuid),
                'score': 1 - distance,  # Convert distance to similarity score
                'metadata': {
                    'text_snippet': props.get('text_snippet', 'N/A'),
                    'subjective_concept': props.get('subjective_concept', 'N/A'),
                    'religious_participation': props.get('religious_participation', 'N/A'),
                }
            })
        return {'matches': matches}
        
    except Exception as e:
        # Fallback: Manual distance calculation
        print(f"Using manual distance calculation: {e}")
        
        # Fetch all objects with vectors
        all_objects = collection.query.fetch_objects(
            limit=1000,
            include_vector=True,
            return_properties=["text_snippet", "subjective_concept", "religious_participation"]
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
                    props = obj.properties
                    scored_objects.append((cosine_dist, {
                        'id': str(obj.uuid),
                        'score': cosine_sim,
                        'metadata': {
                            'text_snippet': props.get('text_snippet', 'N/A'),
                            'subjective_concept': props.get('subjective_concept', 'N/A'),
                            'religious_participation': props.get('religious_participation', 'N/A'),
                        }
                    }))
        
        # Sort by distance and take top results
        scored_objects.sort(key=lambda x: x[0])
        matches = [item for _, item in scored_objects[:top_k]]
        
        return {'matches': matches}
# --- END WEAVIATE IMPLEMENTATION ---

# --- MOCK IMPLEMENTATION FOR DEMONSTRATION ---

VECTOR_DIMENSION = 1536
MOCK_INDEX_DATA = {} # In-memory dictionary to simulate Weaviate/Vector DB storage

def generate_mock_embedding(text):
    """
    SIMULATION: Generates a random vector to simulate a real embedding model.
    """
    # Create a reproducible random vector based on the text hash for conceptual consistency
    np.random.seed(hash(text) % 2**32)
    vector = np.random.rand(VECTOR_DIMENSION).astype('float32')
    return vector / np.linalg.norm(vector) # Normalize vector

def mock_upsert(vectors_to_upsert):
    """SIMULATION: Stores vectors in the in-memory dictionary."""
    print(f"\n[MOCK] Upserting {len(vectors_to_upsert)} vectors into the index...")
    for vector_id, vector, metadata in vectors_to_upsert:
        MOCK_INDEX_DATA[vector_id] = {
            'vector': vector,
            'metadata': metadata
        }
    print("[MOCK] Ingestion complete.")

def mock_query(query_vector, top_k=5):
    """
    SIMULATION: Finds the closest vectors to the query vector using cosine similarity.
    """
    print(f"\n[MOCK] Performing semantic search (k={top_k})...")
    scores = {}
    
    # Calculate cosine similarity manually against all stored vectors
    for item_id, item_data in MOCK_INDEX_DATA.items():
        stored_vector = item_data['vector']
        # Cosine Similarity: (A . B) / (||A|| * ||B||)
        # Since both vectors are normalized, the denominator is 1.
        similarity = np.dot(query_vector, stored_vector)
        scores[item_id] = {'score': similarity, 'metadata': item_data['metadata']}
        
    # Sort by score in descending order
    sorted_results = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Format results to mimic the query response structure
    matches = []
    for item_id, data in sorted_results[:top_k]:
        matches.append({
            'id': item_id,
            'score': float(f"{data['score']:.4f}"), # Format to 4 decimal places
            'metadata': data['metadata']
        })

    return {'matches': matches}

# --- DATASET AND MAIN LOGIC ---

def prepare_dataset():
    """
    Creates the mock dataset of subjective experiences.
    Includes the 'Religiosity' dimension discussed previously.
    """
    print("1. Preparing Subjective Experience Dataset...")
    
    # COMPREHENSIVE DATASET: Deep psychological insights + DHHS policy-relevant data
    # Data format: (Text, Subjective Concept, Religious Participation)
    data = [
        # === DHHS POLICY-RELEVANT: Administrative Burden and Cognitive Load ===
        ("I missed my Medicaid renewal deadline because the letter looked exactly like junk mail. Now I have to start over and explain everything again.", "Cognitive Load/Administrative Overwhelm", "None"),
        ("The kitchen table is buried under old bills, school papers, and unopened junk mail. I can't tell which ones are important and which ones can wait.", "Cognitive Load", "None"),
        ("I spent three hours waiting in the government office; the fluorescent lights gave me a headache. Then they told me I needed a different form.", "Dignity Deprivation/Administrative Burden", "None"),
        ("The clinic staff made me explain my financial situation three times to three different people. Each time felt like another small piece of my dignity being stripped away.", "Dignity Deprivation/Systemic Frustration", "None"),
        ("I walked a mile to the community center only to find out the list of acceptable IDs had changed again. I had the right documents last month, but not anymore.", "Systemic Frustration/Dignity Deprivation", "None"),
        
        # === DHHS POLICY-RELEVANT: Transportation and Healthcare Access ===
        ("The bus service doesn't run early enough for me to make my 8 AM doctor's appointment. I had to cancel and reschedule, which means another month of waiting.", "Logistical Friction/Healthcare Barrier", "None"),
        ("I've become a master of invisible labor—the hours spent on hold, the bus rides to offices that close before I arrive, the paperwork that requires paperwork. This work has no name and no value.", "Invisible Labor/Time Theft", "Low"),
        ("The food pantry is only open for two hours on Tuesday morning, which means I have to skip a shift. I can't afford to lose the hours, but we need the food.", "Time Poverty/Instrumental Barrier", "Low"),
        
        # === DHHS POLICY-RELEVANT: Dignity Deprivation and Stigma ===
        ("I avoid using my SNAP card at the supermarket because I hate the judgment I get from other shoppers. I can see them watching, calculating, judging.", "Stigma/Dignity Deprivation", "Low"),
        ("Every interaction with a system is designed to make me prove I'm human enough to deserve help. I perform desperation, I perform gratitude, I perform worthiness. The performance is the violence.", "Institutional Violence/Performativity", "None"),
        ("I've learned that my trauma is currency. To access resources, I must narrate my suffering in ways that make others comfortable. My pain becomes a story for their consumption.", "Trauma Commodification/Systemic Exploitation", "Low"),
        ("The social worker's tone shifts when she sees my file. I can hear the judgment in her voice—not enough, too much, wrong choices. I've become a case study, not a person.", "Dehumanization/Institutional Objectification", "None"),
        ("I've learned to make my voice smaller in public spaces, to apologize for existing. Poverty taught me that my presence is an inconvenience to others.", "Internalized Oppression/Identity Erosion", "None"),
        
        # === DHHS POLICY-RELEVANT: Housing Insecurity as Primary Stressor ===
        ("We might have to move next month; the constant fear of eviction makes it impossible to sleep. Every noise could be the landlord, every letter could be the notice.", "Housing Insecurity Stress", "None"),
        ("The eviction notice taped to the door, knowing the children will see it when they come home from school. This is what shame looks like when it's public.", "Dignity Deprivation/Public Shame", "None"),
        ("Time doesn't move forward for us; it circles. Every month is the same impossible choices, the same calculations, the same fear. We're trapped in a loop of survival.", "Temporal Entrapment/Chronicity", "None"),
        
        # === Deep Psychological: Intergenerational Patterns ===
        ("My daughter asked why we can't have what other families have, and I had to explain that it's not because we're bad people, but the words felt hollow even as I said them.", "Intergenerational Shame/Identity Formation", "Low"),
        ("My grandmother's hands show the same exhaustion I see in mine. We're three generations of women who've learned that rest is a luxury we can't afford. The pattern is the inheritance.", "Intergenerational Exhaustion/Pattern Recognition", "High"),
        ("My son has learned to hide his needs. He sees me calculating, sees me choosing, sees me breaking. He's eight years old and already understands the economics of scarcity.", "Intergenerational Trauma/Childhood Premature Awareness", "None"),
        ("My children are learning that time is not theirs. They wait. They adapt. They understand that their needs are secondary to the system's schedule.", "Intergenerational Time Poverty", "None"),
        
        # === Deep Psychological: Hope, Faith, and Cognitive Dissonance ===
        ("Hope has become dangerous. Every time I let myself imagine a different future, the crash back to reality is more devastating. I've learned to kill my own dreams preemptively.", "Hope Suppression/Cognitive Protection", "None"),
        ("I pray for strength, but I also pray for the strength to accept that this might be all there is. Faith and resignation have become the same prayer.", "Spiritual Dissonance/Resigned Faith", "High"),
        ("The church tells me God has a plan, but I look at my children and wonder what kind of plan requires them to go hungry. The cognitive dissonance is crushing.", "Religious Cognitive Dissonance", "High"),
        ("I just pray every morning that I get the strength to face another day. It's the only peace I have.", "Spiritual Resilience", "High"),
        ("The church community feeds us, but I can't tell them about the shame of needing it. I perform gratitude while hiding the parts of myself that don't fit their narrative of resilience.", "Community Performativity/Isolated Belonging", "High"),
        ("The church runs the food bank every Friday at 4 PM. It's the one reliable source of fresh produce we have.", "Instrumental Support", "High"),
        ("It's a comfort to know the pastor knows my name and my struggle. I feel seen.", "Bonding Social Capital", "High"),
        
        # === Systemic Frustration and Material Deprivation ===
        ("No matter how much I work, the debt always follows me like a shadow. I can't catch up.", "Systemic Frustration", "None"),
        ("Every bill that comes in the mail is a small explosion of fear. Will the water stay on?", "Cognitive Load", "None"),
        ("I had to choose between fixing the car or buying my son's prescription this month.", "Trade-off Stress", "Low"),
        ("The hardest part isn't the hunger, it's feeling like I have no future to give my children.", "Hopelessness/Intergenerational Stress", "Low"),
        ("I catch myself calculating the cost of every human interaction—can I afford to be friendly, to accept help, to show vulnerability? The math of dignity is exhausting.", "Cognitive Load/Dignity Calculation", "None"),
        
        # === Social Isolation and Emotional Exhaustion ===
        ("I feel completely alone in this. I have no one to talk to about money, shame keeps me silent.", "Social Isolation", "None"),
        ("I've built walls so high that even when help is offered, I can't receive it. Poverty has taught me that accepting help is dangerous, that it comes with invisible strings.", "Isolation as Protection/Trust Erosion", "None"),
        ("My neighbors and I share the same struggle, but we don't talk about it. There's an unspoken agreement: we'll help each other in emergencies, but we'll never acknowledge the systemic nature of our shared condition.", "Collective Silence/Systemic Denial", "Low"),
        ("I find myself snapping at my daughter over small things; I'm just emotionally exhausted.", "Emotional Exhaustion", "Low"),
        ("The threadbare couch is all we have, but it's where we watch movies to escape reality.", "Material Deprivation/Escape", "Low"),
        
        # === Hidden Costs and Paradox of Assistance ===
        ("I've calculated the hidden costs: the bus fare to get to the free clinic, the childcare to attend the job training, the dignity tax of asking for help. Nothing is free, even when it's free.", "Hidden Costs/Paradox of Assistance", "Low"),
        ("An old TV flickers, casting blue light on the mismatched furniture and piles of toys.", "Systemic Frustration", "Low"),
        ("Grandpa is asleep in the armchair; he moved in last month when his benefits changed.", "Social Capital Strain", "Low"),
        ("The constant hum of the old refrigerator and the neighbor's shouting are the soundtrack to our days.", "Sensory Stress/Time Poverty", "None"),
    ]
    
    vectors_to_upsert = []
    for text, concept, religiosity in data:
        # Generate ID and Vector (Vector is needed for the MOCK implementation)
        vector_id = str(uuid.uuid4())
        vector = generate_mock_embedding(text)
        
        metadata = {
            'text_snippet': text,
            'subjective_concept': concept,
            'religious_participation': religiosity
        }
        
        vectors_to_upsert.append((vector_id, vector, metadata))
        
    return vectors_to_upsert

def main():
    """Main function to execute the indexing and search steps."""
    
    # Step 1 & 2: Prepare and Upsert Data
    vectors_to_upsert = prepare_dataset()
    
    # Try to use Weaviate, fallback to mock if unavailable
    client = get_weaviate_client()
    if client:
        try:
            create_weaviate_schema(client)
            weaviate_upsert(client, vectors_to_upsert)
            use_weaviate = True
        except Exception as e:
            print(f"Error setting up Weaviate: {e}")
            print("Falling back to mock implementation...")
            mock_upsert(vectors_to_upsert)
            use_weaviate = False
            client.close()
    else:
        # Fallback to mock upsert if client connection fails
        mock_upsert(vectors_to_upsert)
        use_weaviate = False
    
    # Step 3: Define and Embed the Search Query
    
    # Semantic Query that tests the abstract concept connection
    query_text = "The pervasive feeling of being overwhelmed by administrative and physical clutter." 
    
    # Generate query vector (same for both Weaviate and mock)
    query_vector = generate_mock_embedding(query_text)
    
    if use_weaviate and client:
        query_response = weaviate_query(client, query_vector, top_k=3)
        client.close()
    else:
        query_response = mock_query(query_vector, top_k=3)

    print("\n--- Semantic Search Results (Top 3) ---")
    
    if query_response.get('matches'):
        for i, match in enumerate(query_response['matches']):
            metadata = match['metadata']
            print(f"[{i+1}] Score: {match['score']}")
            print(f"    Concept: {metadata['subjective_concept']}")
            print(f"    Snippet: \"{metadata['text_snippet']}\"")
            print(f"    Religiosity: {metadata['religious_participation']}\n")
    else:
        print("No matches found.")

if __name__ == "__main__":
    main()

