# Extended Artifacts for Social Services Experience Analytics Platform

This document describes the additional artifact types that have been integrated into the Social Services Experience Analytics Platform to capture multi-dimensional experiences.

## 1. Audio Artifacts (The Sensory Dimension)

### Purpose
Audio artifacts capture the sensory environment and emotional texture that images or text might miss.

### Implementation
- **Artifact Type**: Short (5-10 second) ambient sound clips, field recordings, or voice memos
- **Storage**: Audio files stored in `data/audio/` directory
- **Embedding**: Uses librosa to extract audio features:
  - MFCC (Mel-frequency cepstral coefficients)
  - Chroma features
  - Spectral features (centroid, rolloff)
  - Zero-crossing rate
  - Tempo
  - Tonnetz features
- **Vector Dimension**: 512-dimensional normalized vector
- **Field Name**: `audio_vector` (stored as NUMBER_ARRAY in Weaviate)

### Example Use Cases
- Environment: The sound of a loud, bustling street (indicative of crowded living)
- Emotion: The tone of a person's voice (stress, fatigue, or resilience) in a short testimonial
- Context: The persistent drip of a leak, or the hum of a factory nearby

### Usage in CSV
```csv
audio_path
data/audio/street_noise.wav
```

## 2. Time-Series Artifacts (The Duration Dimension)

### Purpose
Time-series artifacts capture patterns or changes over time, reflecting the chronic, cyclical nature of poverty.

### Implementation
- **Artifact Type**: Simplified, anonymized time-series data (JSON format)
- **Data Format**: JSON string with `time` and `value` arrays
- **Embedding**: Uses statistical features and tsfresh for pattern extraction:
  - Basic statistics (mean, std, min, max, median, percentiles)
  - Trend features (mean/std of differences)
  - Volatility (coefficient of variation)
  - Advanced features from tsfresh (if available)
- **Vector Dimension**: 512-dimensional normalized vector
- **Field Name**: `time_series_vector` (stored as NUMBER_ARRAY in Weaviate)

### Example Use Cases
- **Financial Fluctuation**: Monthly cash flow showing spikes and dips over a year
- **Health Patterns**: Illness frequency or sleep quality over a month
- **Instability Visualization**: The "feast or famine" cycle of income

### Usage in CSV
```csv
time_series_data
{"time": [1,2,3,4,5,6,7,8,9,10,11,12], "value": [200,180,220,190,210,195,225,200,215,190,205,200]}
```

## 3. Survey/Likert Scale Artifacts (The Quantifiable Subjectivity)

### Purpose
Quantifiable subjective input to ground abstract narratives with measurable emotional states.

### Implementation
- **Artifact Type**: Numeric ratings on 1-5 scales
- **Survey Dimensions**:
  - `survey_anxiety`: Current level of anxiety (1=Low, 5=High)
  - `survey_control`: Sense of control over future (1=Low, 5=High)
  - `survey_hope`: Level of hope (1=Low, 5=High)
- **Storage**: Stored as INT fields in Weaviate
- **Usage**: Used for filtering and weighting queries

### Example Use Cases
- Find all experiences tagged "Spiritual Resilience" with anxiety ≤ 2
- Filter experiences by high control (≥ 4) and high hope (≥ 4)
- Weight vector searches based on emotional state

### Usage in CSV
```csv
survey_anxiety,survey_control,survey_hope
2,4,4
```

## Integration with Vector Search

### Current Architecture
1. **CLIP Embeddings**: Text + Image → Fused vector via multi2vec-clip
2. **Audio Vectors**: Separate 512-dim vector (stored but not auto-fused)
3. **Time-Series Vectors**: Separate 512-dim vector (stored but not auto-fused)
4. **Survey Ratings**: Used for filtering and metadata

### Future Enhancements
For true multi-modal fusion, you could:
1. **Manual Fusion**: Combine vectors with weighted averaging:
   ```python
   V_exp = 0.5*V_clip + 0.25*V_audio + 0.25*V_timeseries
   ```
2. **Custom Vectorizer**: Implement a custom Weaviate vectorizer that fuses all modalities
3. **Separate Queries**: Query each modality separately and combine results

## Query Examples

### Survey Rating Filter
```python
from query_demo import survey_filtered_search

# Find "Spiritual Resilience" experiences with low anxiety
results = survey_filtered_search(
    client,
    tag="Spiritual Resilience",
    survey_anxiety_max=2,  # Anxiety ≤ 2
    limit=5
)
```

### Combined Filters
```python
# Find experiences with high control and hope
results = survey_filtered_search(
    client,
    tag="Spiritual Resilience",
    survey_control_min=4,  # Control ≥ 4
    survey_hope_min=4,     # Hope ≥ 4
    limit=5
)
```

## Dependencies

### Audio Processing
- `librosa>=0.10.0`: Audio feature extraction
- `soundfile>=0.12.0`: Audio file I/O

### Time-Series Processing
- `tsfresh>=0.20.0`: Time-series feature extraction
- `scikit-learn>=1.3.0`: Statistical utilities

## Notes

1. **Audio Files**: Currently, audio files are optional. If not provided, a zero vector is used.
2. **Time-Series Data**: Must be valid JSON format with `time` and `value` arrays.
3. **Survey Ratings**: Optional fields. If not provided, they are not stored (not set to 0).
4. **Vector Dimensions**: All vectors are normalized to unit length for consistent distance calculations.

## Data Schema

The extended schema includes:
- `audio_path` (TEXT): Path to audio file
- `audio_vector` (NUMBER_ARRAY[512]): Pre-computed audio embedding
- `time_series_data` (TEXT): JSON string of time-series data
- `time_series_vector` (NUMBER_ARRAY[512]): Pre-computed time-series embedding
- `survey_anxiety` (INT): Anxiety rating 1-5
- `survey_control` (INT): Control rating 1-5
- `survey_hope` (INT): Hope rating 1-5

## Ethical Considerations

When working with real data:
- **Audio**: Ensure proper consent for recording and storage
- **Time-Series**: Anonymize financial/health data appropriately
- **Survey Data**: Protect sensitive emotional/psychological information
- **Privacy**: All artifacts should be de-identified and stored securely

