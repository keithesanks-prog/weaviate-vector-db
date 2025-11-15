# NLM (National Library of Medicine) Integration

This document describes the integration of the National Library of Medicine (NLM) services with the Social Services Experience Analytics Platform.

## Overview

The NLM integration enriches client experience data with authoritative health information from:
- **MedlinePlus**: Consumer health information
- **PubMed**: Biomedical research articles
- **Future**: UMLS (Unified Medical Language System) for terminology standardization

## Features

### 1. Automatic Health Keyword Extraction
The system automatically extracts health-related keywords from client experience text snippets, including:
- Mental health terms (anxiety, depression, stress)
- Physical health conditions
- Healthcare-related phrases

### 2. MedlinePlus Integration
Provides links to relevant consumer health information from MedlinePlus, including:
- Health topic pages
- Search results for specific conditions
- Authoritative health information for clients

### 3. PubMed Integration
Queries PubMed for relevant research articles related to:
- Social determinants of health
- Mental health and poverty
- Health disparities research
- Evidence-based interventions

### 4. Context-Aware Enrichment
The system uses multiple signals to determine relevant health information:
- Text snippet content
- Abstract tags (e.g., "Cognitive Load", "Emotional Exhaustion")
- Survey data (e.g., high anxiety levels trigger mental health resources)

## Usage

### In the Web Interface

1. **Enable NLM Enrichment**: Check the "Include NLM Health Information" checkbox before searching
2. **View Enrichment**: When enabled, search results will include:
   - Health topics identified in the experience
   - Links to MedlinePlus articles
   - Links to relevant PubMed research articles

### Programmatic Usage

```python
from nlm_integration import enrich_experience_with_nlm

# Enrich a single experience
enrichment = enrich_experience_with_nlm(
    text_snippet="I feel completely alone and anxious about my situation",
    tag_abstract="Social Isolation",
    survey_anxiety=5
)

# Access enrichment data
print(enrichment['health_keywords'])
print(enrichment['medlineplus_articles'])
print(enrichment['pubmed_articles'])
```

## API Endpoints

### `/api/enrich/<experience_id>`
Get NLM enrichment for a specific experience by ID.

**Example Request:**
```bash
curl http://localhost:5000/api/enrich/abc123
```

**Example Response:**
```json
{
  "experience_id": "abc123",
  "enrichment": {
    "health_keywords": ["anxiety", "stress", "mental health"],
    "medlineplus_articles": [
      {
        "title": "MedlinePlus: Anxiety",
        "url": "https://medlineplus.gov/anxiety.html",
        "source": "MedlinePlus",
        "type": "Consumer Health Information"
      }
    ],
    "pubmed_articles": [
      {
        "pmid": "12345678",
        "title": "Mental Health and Social Determinants...",
        "authors": "Smith, J., Doe, A.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
        "source": "PubMed",
        "type": "Research Article"
      }
    ],
    "suggested_topics": ["anxiety", "stress management", "mental health"]
  }
}
```

## Configuration

### Dependencies
The NLM integration requires:
- `requests>=2.31.0` (for API calls)

Install with:
```bash
pip install requests
```

### API Rate Limits
- **PubMed**: No API key required, but rate limits apply (recommended: <3 requests/second)
- **MedlinePlus**: No API key required, public access

## Future Enhancements

1. **UMLS Integration**: Standardize medical terminology across experiences
2. **ClinicalTrials.gov**: Link experiences to relevant clinical trials
3. **Caching**: Cache NLM results to reduce API calls
4. **Advanced NLP**: Use medical NLP models to better extract health concepts
5. **Multi-language Support**: Support for Spanish and other languages via NLM resources

## References

- [MedlinePlus Web Service](https://medlineplus.gov/about/developers/webservices/)
- [PubMed API (E-utilities)](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
- [NLM Homepage](https://www.nlm.nih.gov/)
- [UMLS Terminology Services](https://www.nlm.nih.gov/research/umls/index.html)

## Notes

- The integration is designed to be non-intrusive and optional
- All NLM data is fetched on-demand (not stored in Weaviate)
- Links open in new tabs to preserve the user's search context
- The system gracefully handles API failures and continues without enrichment

