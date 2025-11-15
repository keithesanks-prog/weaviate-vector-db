# Social Services Experience Analytics Platform

A vector database implementation using Weaviate with CLIP module to analyze and query multi-dimensional client experiences in social services programs through multi-modal embeddings (text, images, audio, time-series). Designed for policy analysis, barrier identification, and program improvement for agencies like the Department of Health and Human Services (DHHS).

## Overview

This platform enables social services agencies to capture and analyze client experiences by combining:
- **Text narratives**: Evocative quotes and descriptions
- **Images**: Visual representations of experiences
- **Metadata**: Sociological context (education level, religious participation, abstract tags)

The database uses Weaviate's multi2vec-clip module to create fused Experience Vectors that capture the semantic meaning across both text and images.

## Features

1. **Abstract Text Search**: Query using abstract concepts and find semantically similar experiences
2. **Filtered Search**: Combine semantic search with metadata filters for sociological insights
3. **Conceptual Distance Analysis**: Find experiences that are outliers from a conceptual average (anti-k-NN)

## Project Structure

```
vector_DB/
├── docker-compose.yml      # Weaviate + CLIP module setup
├── requirements.txt        # Python dependencies
├── mock_data.csv          # 30 diverse poverty experience entries
├── ingest_data.py         # Data ingestion script (Weaviate)
├── query_demo.py          # Query demonstration script (Weaviate)
├── mock_demo.py           # Standalone mock demo (no Weaviate)
├── web_query_server.py    # Flask web server for query interface
├── query_interface.html   # Web-based query interface
├── setup.sh               # Virtual environment setup script
├── run.sh                 # Run script (with venv activation)
├── quick_start.sh         # Complete setup and run script
├── start_web_interface.sh # Start web query interface
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- pip

## Setup Instructions

### Step 1: Create Virtual Environment

Modern Linux distributions (Ubuntu 23.04+, Debian 12+) use externally managed Python environments. You must use a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** If `python3 -m venv` fails, you may need to install the venv module:
```bash
sudo apt install python3-venv python3-full
```

**Alternative:** Use the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

### Step 2: Start Weaviate with CLIP Module

```bash
docker-compose up -d
```

This will start:
- Weaviate server on `http://localhost:8080`
- Multi2vec-clip inference API on `http://localhost:8081`

Wait a few moments for the services to fully start. You can check if Weaviate is ready:

```bash
curl http://localhost:8080/v1/.well-known/ready
```

### Step 3: Ingest Mock Data

**Important:** Make sure the virtual environment is activated:
```bash
source venv/bin/activate
python ingest_data.py
```

This script will:
1. Connect to Weaviate
2. Create the `ClientExperience` schema with CLIP module configuration
3. Process the CSV data
4. Generate placeholder images (if needed) from text snippets
5. Upload all data to Weaviate with vector embeddings

The script creates placeholder images automatically if they don't exist. Each image is generated from the text snippet and saved to `data/images/`.

### Step 4: Run Query Demonstrations

**Important:** Make sure the virtual environment is activated:
```bash
source venv/bin/activate
python query_demo.py
```

### Alternative: Use Helper Scripts

**Quick Start (sets up everything):**
```bash
chmod +x quick_start.sh
./quick_start.sh
```

**Run with virtual environment:**
```bash
chmod +x run.sh
./run.sh all      # Run both ingestion and queries
./run.sh ingest   # Run ingestion only
./run.sh query    # Run queries only
```

This will demonstrate three query types:
1. **Abstract Text Search**: "What does it look like when hope conflicts with reality?"
2. **Filtered Search**: Search for "The quiet strength of community support" filtered by `religious_participation = "High (Weekly)"`
3. **Conceptual Distance**: Find experiences farthest from the average "Spiritual Resilience" experience

## Standalone Mock Demo (No Weaviate Required)

For demonstrations or testing without setting up Weaviate, you can use the standalone mock demo:

```bash
# Activate virtual environment
source venv/bin/activate

# Run the mock demo
python -B mock_demo.py
```

This script:
- Uses mock embeddings (deterministic hash-based vectors)
- Demonstrates semantic search concepts without infrastructure
- Includes 18 diverse poverty experience entries
- Shows how to adapt code for real Weaviate integration

The mock demo is useful for:
- Understanding the concept before setting up Weaviate
- Quick demonstrations without Docker/Weaviate dependencies
- Testing query logic and data structures
- Educational purposes

**Note**: The mock demo uses simplified embeddings and won't have the same semantic accuracy as real CLIP embeddings, but it demonstrates the core vector search concepts.

## Data Schema

Each experience includes:

- **text**: Evocative narrative text snippet
- **image**: Base64-encoded image (automatically generated if not present)
- **tag_abstract**: Abstract concept tag (e.g., "Spiritual Resilience", "Systemic Frustration", "Time Poverty")
- **ed_level_primary**: Education level ("No Diploma", "High School", "College")
- **religious_participation**: Religious participation level ("High (Weekly)", "Low (Monthly)", "None")
- **image_path**: Original image file path

## Query Examples

### Abstract Text Search

```python
from query_demo import connect_to_weaviate, abstract_text_search

client = connect_to_weaviate()
results = abstract_text_search(
    client,
    "What does it look like when hope conflicts with reality?",
    limit=5
)
```

### Filtered Search

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

### Conceptual Distance Analysis

```python
from query_demo import calculate_conceptual_distance

results = calculate_conceptual_distance(
    client,
    tag="Spiritual Resilience",
    limit=5
)
```

## Understanding the Results

- **Distance**: Lower distance = more similar to query
- **Certainty**: Higher certainty = more confident match
- **Conceptual Distance**: Measures how far an experience is from the average vector of a tag category

## Customization

### Adding Your Own Data

1. Edit `mock_data.csv` to add new entries
2. Add corresponding images to `data/images/` (or let the script generate placeholders)
3. Run `python ingest_data.py` again (it will delete and recreate the schema)

### Modifying Queries

Edit `query_demo.py` to create custom queries:

```python
# Custom abstract search
abstract_text_search(client, "Your query text here", limit=10)

# Custom filtered search
filtered_search(
    client,
    "Query text",
    filter_property="ed_level_primary",
    filter_value="College",
    limit=10
)
```

### Changing CLIP Configuration

Edit the schema in `ingest_data.py` to adjust:
- Text/image weight ratio
- Vector dimensions
- Module configuration

## Troubleshooting

### Weaviate Connection Error

- Make sure Docker containers are running: `docker-compose ps`
- Check logs: `docker-compose logs weaviate`
- Verify Weaviate is ready: `curl http://localhost:8080/v1/.well-known/ready`

### Image Generation Issues

- The script will create placeholder images automatically
- If font errors occur, the script falls back to default fonts
- Ensure Pillow is installed: `pip install Pillow`

### Query Returns No Results

- Verify data was ingested: Check Weaviate console at `http://localhost:8080/v1/schema`
- Try broader queries
- Check filter values match exact strings in the data

## Technical Details

- **Vector Database**: Weaviate 1.24.11
- **Embedding Model**: CLIP (ViT-B-32 multilingual v1)
- **Vector Dimension**: 512 (CLIP default)
- **Text/Image Weight**: 70% text, 30% image (configurable)

## Ethical Considerations

This project uses mock data to demonstrate technical capabilities. When working with real poverty experience data:

- Obtain proper consent
- Protect privacy and anonymity
- Follow ethical research guidelines
- Consider the social implications of the analysis

## License

This project is for educational and demonstration purposes.

## References

- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Multi2vec-clip Module](https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/multi2vec-clip)
- [CLIP Model](https://openai.com/blog/clip/)

