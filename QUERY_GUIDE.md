# Query Guide for Social Services Experience Analytics Platform

## Quick Start

### Option 1: Run the Demo Script (Easiest)
```bash
source venv/bin/activate
python -B query_demo.py
```

This will run all 5 query demonstrations automatically.

### Option 2: Interactive Python Session
```bash
source venv/bin/activate
python -B
```

Then in Python:
```python
from query_demo import *
import weaviate

# Connect
client = connect_to_weaviate()

# Run a simple query
results = abstract_text_search(client, "What does hope look like in difficult times?", limit=5)

# Close connection
client.close()
```

## Query Types

### 1. Abstract Text Search
Semantic search using natural language queries.

**Python:**
```python
from query_demo import connect_to_weaviate, abstract_text_search

client = connect_to_weaviate()
results = abstract_text_search(
    client,
    "What does it look like when hope conflicts with reality?",
    limit=5
)
client.close()
```

**Direct API (curl):**
```bash
# First, generate a query vector (requires CLIP model)
# Or use Weaviate's text2vec if available
curl -X POST http://localhost:8080/v1/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "{
      Get {
        ClientExperience(
          nearText: {
            concepts: [\"What does hope look like in difficult times?\"]
          }
          limit: 5
        ) {
          text
          tag_abstract
          ed_level_primary
          religious_participation
        }
      }
    }"
  }'
```

### 2. Filtered Search
Combine semantic search with metadata filters.

**Python:**
```python
from query_demo import filtered_search

results = filtered_search(
    client,
    "The quiet strength of community support",
    filter_property="religious_participation",
    filter_value="High (Weekly)",
    limit=5
)
```

**Example Filters:**
- `religious_participation = "High (Weekly)"`
- `ed_level_primary = "College"`
- `tag_abstract = "Spiritual Resilience"`

### 3. Survey Rating Filter
Filter by quantifiable emotional states.

**Python:**
```python
from query_demo import survey_filtered_search

# Find "Spiritual Resilience" experiences with low anxiety
results = survey_filtered_search(
    client,
    tag="Spiritual Resilience",
    survey_anxiety_max=2,  # Anxiety ≤ 2
    limit=5
)

# Find experiences with high control and hope
results = survey_filtered_search(
    client,
    tag="Spiritual Resilience",
    survey_control_min=4,  # Control ≥ 4
    survey_hope_min=4,     # Hope ≥ 4
    limit=5
)
```

### 4. Advanced Multi-Modal Query
Combine semantic search with multiple filters including time-series volatility.

**Python:**
```python
from query_demo import advanced_multi_modal_query

# Find spiritual hope with high financial volatility and low control
results = advanced_multi_modal_query(
    client,
    query_text="spiritual hope and resilience in difficult times",
    tag="Spiritual Resilience",
    survey_hope_min=3,
    high_volatility=True,  # Filter for high financial volatility
    limit=5
)
```

### 5. Conceptual Distance Analysis
Find outliers from a conceptual average.

**Python:**
```python
from query_demo import calculate_conceptual_distance

# Find experiences farthest from "Spiritual Resilience" average
results = calculate_conceptual_distance(
    client,
    tag="Spiritual Resilience",
    limit=5
)
```

## Direct Weaviate API Queries

### Using GraphQL

**Get all objects:**
```bash
curl -X POST http://localhost:8080/v1/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "{
      Get {
        ClientExperience(limit: 10) {
          text
          tag_abstract
          survey_anxiety
          survey_control
          survey_hope
        }
      }
    }"
  }'
```

**Filter by property:**
```bash
curl -X POST http://localhost:8080/v1/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "{
      Get {
        ClientExperience(
          where: {
            path: [\"tag_abstract\"]
            operator: Equal
            valueText: \"Spiritual Resilience\"
          }
          limit: 5
        ) {
          text
          tag_abstract
          survey_anxiety
        }
      }
    }"
  }'
```

**Filter by survey rating:**
```bash
curl -X POST http://localhost:8080/v1/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "{
      Get {
        ClientExperience(
          where: {
            operator: And
            operands: [
              {
                path: [\"tag_abstract\"]
                operator: Equal
                valueText: \"Spiritual Resilience\"
              }
              {
                path: [\"survey_anxiety\"]
                operator: LessOrEqual
                valueInt: 2
              }
            ]
          }
          limit: 5
        ) {
          text
          tag_abstract
          survey_anxiety
          survey_control
          survey_hope
        }
      }
    }"
  }'
```

### Using REST API

**Get schema:**
```bash
curl http://localhost:8080/v1/schema
```

**Get objects:**
```bash
curl "http://localhost:8080/v1/objects?class=ClientExperience&limit=5"
```

**Get specific object:**
```bash
curl "http://localhost:8080/v1/objects/{object-id}"
```

## Query Examples by Use Case

### Find High-Stress Experiences
```python
results = survey_filtered_search(
    client,
    tag=None,  # Any tag
    survey_anxiety_max=5,
    survey_control_min=None,
    survey_control_max=2,  # Low control
    limit=10
)
```

### Find Resilient Experiences with Low Anxiety
```python
results = survey_filtered_search(
    client,
    tag="Spiritual Resilience",
    survey_anxiety_max=2,
    survey_hope_min=3,
    limit=10
)
```

### Find Experiences Similar to a Concept
```python
results = abstract_text_search(
    client,
    "The weight of financial uncertainty and its impact on daily life",
    limit=10
)
```

### Find Experiences with Specific Education Level
```python
results = filtered_search(
    client,
    "Educational barriers and challenges",
    filter_property="ed_level_primary",
    filter_value="No Diploma",
    limit=10
)
```

## Custom Query Script

Create your own query script:

```python
#!/usr/bin/env python3
"""Custom query script"""

from query_demo import connect_to_weaviate, abstract_text_search
import weaviate.classes.query as wvq

def my_custom_query():
    client = connect_to_weaviate()
    collection = client.collections.get("ClientExperience")
    
    # Your custom query here
    result = collection.query.near_vector(
        near_vector=your_query_vector,  # Generate with CLIP
        limit=10,
        filters=wvq.Filter.by_property("tag_abstract").equal("Time Poverty"),
        return_properties=["text", "tag_abstract", "survey_anxiety"]
    )
    
    for obj in result.objects:
        print(obj.properties)
    
    client.close()

if __name__ == "__main__":
    my_custom_query()
```

## Tips

1. **Use the demo script first** to see all query types in action
2. **Start with simple queries** and add filters gradually
3. **Check the distance scores** - lower = more similar
4. **Combine filters** for precise results
5. **Use survey ratings** to quantify emotional states
6. **Time-series volatility** can reveal financial instability patterns

## Troubleshooting

**No results found?**
- Check if data was ingested: `curl http://localhost:8080/v1/objects?class=ClientExperience`
- Verify filter values match exactly (case-sensitive)
- Try broader queries without filters first

**CLIP not available?**
- Queries will fall back to Weaviate's default text vectorizer
- Install transformers/torch for full CLIP functionality

**Connection errors?**
- Ensure Weaviate is running: `docker-compose ps`
- Check Weaviate is ready: `curl http://localhost:8080/v1/.well-known/ready`

