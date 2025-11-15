# Manual Vector Fusion Implementation

## Overview

The Social Services Experience Analytics Platform uses **manual vector fusion** to combine multiple modalities into a single, comprehensive experience vector. This provides maximum control over how different data types contribute to the final embedding.

## Architecture Changes

### Before (Automatic Vectorization)
- Weaviate's `multi2vec-clip` automatically generated vectors
- Only text and image were fused
- Audio and time-series vectors were stored separately

### After (Manual Vector Fusion)
- **Vectorizer disabled**: `vectorizer_config=None`
- **Manual CLIP generation**: Using Hugging Face transformers
- **Weighted fusion**: All vectors combined with custom weights
- **Single fused vector**: One vector per experience encoding all modalities

## Fusion Formula

The final experience vector ($V_{final}$) is calculated as:

$$V_{final} = 0.6 \times V_{clip} + 0.15 \times V_{audio} + 0.15 \times V_{timeseries}$$

Where:
- **$V_{clip}$** (60%): CLIP embedding of text + image (70% text, 30% image)
- **$V_{audio}$** (15%): Audio feature embedding (librosa features)
- **$V_{timeseries}$** (15%): Time-series pattern embedding (statistical + tsfresh features)

All vectors are normalized before fusion, and the final vector is re-normalized.

## Implementation Details

### 1. Schema Configuration

```python
client.collections.create(
    name="ClientExperience",
    vectorizer_config=None,  # Disable automatic vectorization
    properties=[...]
)
```

### 2. CLIP Embedding Generation

```python
def create_clip_embedding(text: str, image_b64: str) -> list:
    """
    Generate CLIP embedding from text and image.
    - Uses OpenAI CLIP ViT-B/32 model
    - Fuses text (70%) and image (30%)
    - Returns 512-dimensional normalized vector
    """
```

### 3. Vector Fusion

```python
def fuse_vectors(v_clip: list, v_audio: list, v_timeseries: list) -> list:
    """
    Fuse vectors with weights:
    - CLIP: 60%
    - Audio: 15%
    - Time-Series: 15%
    """
```

### 4. Data Upload

```python
batch.add_object(
    properties=data_object,
    vector=fused_vector  # Manually provided fused vector
)
```

## Query Vector Generation

For queries, we generate CLIP embeddings for the query text to match the stored vectors:

```python
def generate_query_vector(query_text: str) -> list:
    """
    Generate CLIP embedding for query text.
    Uses the same CLIP model as ingestion for consistency.
    """
```

Queries use `near_vector` with the CLIP-generated query vector, ensuring semantic consistency.

## Benefits

1. **Full Control**: Complete control over fusion weights
2. **Multi-Modal**: All modalities (text, image, audio, time-series) in one vector
3. **Consistency**: Same CLIP model for ingestion and queries
4. **Flexibility**: Easy to adjust weights based on domain needs
5. **Extensibility**: Simple to add new modalities

## Weight Tuning

The current weights (60% CLIP, 15% audio, 15% time-series) are a starting point. You can adjust them based on:

- **Domain requirements**: Which modalities are most important?
- **Data quality**: If audio/time-series data is sparse, reduce their weights
- **Query patterns**: What types of queries are most common?

Example adjustments:
```python
weights = {
    'clip': 0.7,      # Increase if text/image are most important
    'audio': 0.1,     # Decrease if audio data is sparse
    'timeseries': 0.2  # Increase if financial patterns are critical
}
```

## Dependencies

- `transformers>=4.35.0`: For CLIP model
- `torch>=2.1.0`: PyTorch backend
- `librosa>=0.10.0`: Audio processing
- `tsfresh>=0.20.0`: Time-series features

## Performance Considerations

1. **Model Loading**: CLIP model is loaded once and reused
2. **GPU Support**: Automatically uses GPU if available
3. **Batch Processing**: Vectors generated in batch during ingestion
4. **Caching**: Query vectors could be cached for repeated queries

## Future Enhancements

1. **Dynamic Weights**: Adjust weights per experience based on data quality
2. **Learned Fusion**: Train a fusion network instead of weighted average
3. **Multi-Head Attention**: Use attention mechanisms for fusion
4. **Vector Compression**: Reduce dimensionality while preserving information

## Migration Notes

If upgrading from automatic vectorization:

1. **Re-ingest data**: All data must be re-ingested with new fused vectors
2. **Update queries**: Queries now use `near_vector` instead of `near_text` (with CLIP)
3. **Install dependencies**: Ensure transformers and torch are installed
4. **Model download**: First run will download CLIP model (~500MB)

## Example Usage

```python
# Ingestion automatically fuses vectors
python ingest_data.py

# Queries use CLIP-generated query vectors
python query_demo.py
```

The advanced multi-modal query demonstrates combining:
- Semantic search (CLIP query vector)
- Survey filters (anxiety, control, hope)
- Time-series filters (volatility)

