# Social Services Experience Analytics Platform - Architecture Diagram

## System Architecture Overview

```mermaid
graph TB
    subgraph "Data Sources"
        CSV[CSV Data File<br/>42 Client Experiences]
        IMG[Image Files<br/>Placeholder Generation]
        AUD[Audio Artifacts<br/>Optional]
        TS[Time-Series Data<br/>JSON Format]
        SURV[Survey Ratings<br/>Anxiety, Control, Hope]
    end

    subgraph "Data Ingestion Pipeline"
        INGEST[ingest_data.py<br/>Data Processing]
        CLIP[CLIP Model<br/>Text + Image Embedding]
        AUDIO[Audio Feature Extraction<br/>librosa]
        TSFEAT[Time-Series Features<br/>tsfresh/statistics]
        FUSE[Vector Fusion<br/>60% CLIP, 15% Audio, 15% TS]
    end

    subgraph "Vector Database"
        WEAV[Weaviate 1.27.0<br/>Vector Database]
        SCHEMA[ClientExperience Schema<br/>Multi-modal Properties]
        VECTORS[512D Fused Vectors<br/>+ Metadata]
    end

    subgraph "Query & Analysis Layer"
        FLASK[Flask Web Server<br/>web_query_server.py]
        QUERY[Semantic Search<br/>CLIP Query Vectors]
        FILTER[Advanced Filters<br/>Policy Metadata]
        NLM[NLM Integration<br/>MedlinePlus + PubMed]
        VIZ[Visualizations<br/>t-SNE, Geospatial, Correlation]
    end

    subgraph "Frontend Interface"
        UI[Web Interface<br/>query_interface.html]
        SEARCH[Semantic Search<br/>Natural Language]
        FILTERS[Filter Controls<br/>Policy-Driven Metadata]
        RESULTS[Result Display<br/>Enhanced Presentation]
        CHARTS[Data Visualizations<br/>Plotly.js]
    end

    subgraph "External Services"
        NLM_API[National Library of Medicine<br/>MedlinePlus & PubMed APIs]
    end

    %% Data Flow
    CSV --> INGEST
    IMG --> INGEST
    AUD --> AUDIO
    TS --> TSFEAT
    SURV --> INGEST

    INGEST --> CLIP
    AUDIO --> FUSE
    TSFEAT --> FUSE
    CLIP --> FUSE

    FUSE --> WEAV
    INGEST --> SCHEMA
    SCHEMA --> WEAV
    WEAV --> VECTORS

    VECTORS --> FLASK
    FLASK --> QUERY
    FLASK --> FILTER
    FLASK --> NLM
    FLASK --> VIZ

    QUERY --> UI
    FILTER --> UI
    NLM --> UI
    VIZ --> UI

    UI --> SEARCH
    UI --> FILTERS
    UI --> RESULTS
    UI --> CHARTS

    NLM --> NLM_API

    style CSV fill:#3b82f6,stroke:#2563eb,color:#fff
    style WEAV fill:#10b981,stroke:#059669,color:#fff
    style FLASK fill:#f59e0b,stroke:#d97706,color:#fff
    style UI fill:#8b5cf6,stroke:#7c3aed,color:#fff
    style FUSE fill:#ef4444,stroke:#dc2626,color:#fff
```

## Detailed Data Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant CSV as CSV Data
    participant Ingest as ingest_data.py
    participant CLIP as CLIP Model
    participant Audio as Audio Processor
    participant TS as Time-Series Processor
    participant Fusion as Vector Fusion
    participant Weaviate as Weaviate DB
    participant Flask as Flask Server
    participant Frontend as Web Interface
    participant NLM as NLM Services

    User->>CSV: Prepare Data
    CSV->>Ingest: Load CSV Records
    
    loop For Each Experience
        Ingest->>CLIP: Generate Text+Image Embedding
        CLIP-->>Ingest: 512D CLIP Vector
        
        Ingest->>Audio: Extract Audio Features
        Audio-->>Ingest: 512D Audio Vector
        
        Ingest->>TS: Extract Time-Series Features
        TS-->>Ingest: 512D TS Vector
        
        Ingest->>Fusion: Fuse Vectors (60/15/15)
        Fusion-->>Ingest: 512D Fused Vector
        
        Ingest->>Weaviate: Store Vector + Metadata
    end
    
    User->>Frontend: Enter Query
    Frontend->>Flask: POST /api/search
    Flask->>CLIP: Generate Query Vector
    CLIP-->>Flask: Query Vector
    Flask->>Weaviate: Semantic Search
    Weaviate-->>Flask: Matching Experiences
    
    opt NLM Enabled
        Flask->>NLM: Enrich with Health Info
        NLM-->>Flask: MedlinePlus + PubMed Links
    end
    
    Flask-->>Frontend: Results + Enrichment
    Frontend->>User: Display Results
    
    User->>Frontend: Request Visualization
    Frontend->>Flask: GET /api/visualizations/tsne
    Flask->>Weaviate: Fetch All Vectors
    Weaviate-->>Flask: All Experience Vectors
    Flask->>Flask: t-SNE Dimensionality Reduction
    Flask-->>Frontend: 2D Coordinates
    Frontend->>User: Interactive Plotly Chart
```

## Component Interaction Diagram

```mermaid
graph LR
    subgraph "Client Experience Data"
        TEXT[Text Snippets<br/>Client Narratives]
        IMAGE[Images<br/>Visual Context]
        AUDIO[Audio<br/>Sensory Data]
        TS_DATA[Time-Series<br/>Stability Patterns]
        META[Metadata<br/>Policy Metrics]
    end

    subgraph "Embedding Generation"
        CLIP_EMB[CLIP Embedding<br/>512D]
        AUDIO_EMB[Audio Embedding<br/>512D]
        TS_EMB[Time-Series Embedding<br/>512D]
    end

    subgraph "Vector Fusion"
        FUSED[Fused Vector<br/>60% CLIP<br/>15% Audio<br/>15% TS<br/>512D]
    end

    subgraph "Storage"
        WEAVIATE[(Weaviate<br/>Vector Database)]
    end

    subgraph "Query Processing"
        QUERY_VEC[Query Vector<br/>CLIP Generated]
        SEARCH[Semantic Search<br/>Cosine Similarity]
        FILTERS[Metadata Filters<br/>Policy-Driven]
    end

    subgraph "Results & Analysis"
        RESULTS[Search Results]
        VIZ[Visualizations]
        NLM_ENRICH[NLM Enrichment]
    end

    TEXT --> CLIP_EMB
    IMAGE --> CLIP_EMB
    AUDIO --> AUDIO_EMB
    TS_DATA --> TS_EMB
    META --> WEAVIATE

    CLIP_EMB --> FUSED
    AUDIO_EMB --> FUSED
    TS_EMB --> FUSED

    FUSED --> WEAVIATE

    QUERY_VEC --> SEARCH
    FILTERS --> SEARCH
    SEARCH --> WEAVIATE
    WEAVIATE --> RESULTS
    RESULTS --> VIZ
    RESULTS --> NLM_ENRICH

    style FUSED fill:#ef4444,stroke:#dc2626,color:#fff
    style WEAVIATE fill:#10b981,stroke:#059669,color:#fff
    style RESULTS fill:#3b82f6,stroke:#2563eb,color:#fff
```

## Policy-Driven Metadata Flow

```mermaid
graph TD
    subgraph "Policy Metadata Categories"
        PROG[Programmatic Status<br/>- Program Enrollment<br/>- Enrollment Status<br/>- Document Success Rate]
        GEO[Geographic Context<br/>- Service Office Location<br/>- Urban/Rural<br/>- Time of Day]
        STAB[Stability Measures<br/>- Residential Moves<br/>- Financial Volatility<br/>- Primary Language]
    end

    subgraph "Query Capabilities"
        SEM[Semantic Search<br/>Natural Language]
        NEG[Negative Filtering<br/>Conceptual Exclusion]
        RANGE[Range Queries<br/>Min/Max Filters]
        REWEIGHT[Query-Time Re-weighting<br/>Modality Adjustment]
    end

    subgraph "Analysis Outputs"
        CLUSTER[Vector Clustering<br/>t-SNE/UMAP]
        HEATMAP[Geospatial Heatmap<br/>Location Analysis]
        CORR[Correlation Plots<br/>Subjective vs Objective]
    end

    PROG --> SEM
    GEO --> SEM
    STAB --> SEM

    PROG --> NEG
    GEO --> RANGE
    STAB --> RANGE

    SEM --> CLUSTER
    NEG --> CLUSTER
    RANGE --> HEATMAP
    REWEIGHT --> CORR

    CLUSTER --> POLICY[Policy Insights<br/>& Recommendations]
    HEATMAP --> POLICY
    CORR --> POLICY

    style POLICY fill:#8b5cf6,stroke:#7c3aed,color:#fff
    style SEM fill:#3b82f6,stroke:#2563eb,color:#fff
```

## Technology Stack Diagram

```mermaid
graph TB
    subgraph "Infrastructure"
        DOCKER[Docker Compose<br/>Container Orchestration]
        WEAVIATE_CONTAINER[Weaviate Container<br/>Port 8080/50051]
    end

    subgraph "Python Backend"
        VENV[Python Virtual Environment<br/>PEP 668 Compliant]
        DEPS[Python Dependencies<br/>- weaviate-client<br/>- transformers<br/>- torch<br/>- pandas<br/>- librosa<br/>- tsfresh<br/>- flask<br/>- scikit-learn<br/>- umap-learn]
    end

    subgraph "AI/ML Models"
        CLIP_MODEL[CLIP Model<br/>openai/clip-vit-base-patch32<br/>Text + Image Understanding]
        TSNE[scikit-learn t-SNE<br/>Dimensionality Reduction]
        UMAP[UMAP<br/>Alternative Dimensionality Reduction]
    end

    subgraph "Frontend"
        HTML[HTML5 + Tailwind CSS<br/>Dark Theme UI]
        JS[JavaScript<br/>Fetch API, Plotly.js]
        PLOTLY[Plotly.js<br/>Interactive Visualizations]
    end

    subgraph "External APIs"
        NLM_API[NLM APIs<br/>MedlinePlus + PubMed]
    end

    DOCKER --> WEAVIATE_CONTAINER
    VENV --> DEPS
    DEPS --> CLIP_MODEL
    DEPS --> TSNE
    DEPS --> UMAP
    HTML --> JS
    JS --> PLOTLY
    JS --> NLM_API

    style CLIP_MODEL fill:#ef4444,stroke:#dc2626,color:#fff
    style WEAVIATE_CONTAINER fill:#10b981,stroke:#059669,color:#fff
    style PLOTLY fill:#8b5cf6,stroke:#7c3aed,color:#fff
```

## Query Processing Flow

```mermaid
flowchart TD
    START[User Enters Query] --> CHECK{Query Type?}
    
    CHECK -->|Semantic Search| SEM[Generate CLIP Query Vector]
    CHECK -->|With Filters| FILTER[Build Filter Chain]
    CHECK -->|With Exclusion| NEG[Apply Negative Filtering]
    
    SEM --> VEC[Query Vector Ready]
    FILTER --> VEC
    NEG --> VEC
    
    VEC --> SEARCH[Weaviate near_vector Search]
    SEARCH --> RESULTS{Results Found?}
    
    RESULTS -->|Yes| ENRICH{NLM Enabled?}
    RESULTS -->|No| FALLBACK[Manual Distance Calculation]
    FALLBACK --> ENRICH
    
    ENRICH -->|Yes| NLM[NLM Enrichment]
    ENRICH -->|No| FORMAT[Format Results]
    NLM --> FORMAT
    
    FORMAT --> DISPLAY[Display in UI]
    DISPLAY --> VIZ{Visualization Requested?}
    
    VIZ -->|t-SNE| TSNE[Generate Clustering]
    VIZ -->|Geospatial| GEO[Generate Heatmap]
    VIZ -->|Correlation| CORR[Generate Scatter Plot]
    
    TSNE --> RENDER[Render with Plotly]
    GEO --> RENDER
    CORR --> RENDER
    
    RENDER --> END[User Views Results]

    style SEM fill:#3b82f6,stroke:#2563eb,color:#fff
    style SEARCH fill:#10b981,stroke:#059669,color:#fff
    style NLM fill:#f59e0b,stroke:#d97706,color:#fff
    style RENDER fill:#8b5cf6,stroke:#7c3aed,color:#fff
```

