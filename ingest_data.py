"""
Ingestion script for Social Services Experience Analytics Platform
Uploads mock data to Weaviate vector database with CLIP module
"""

import weaviate
import weaviate.classes.config as wvc
import pandas as pd
import base64
import os
import json
import csv
from PIL import Image, ImageDraw, ImageFont
import io
from pathlib import Path
import numpy as np
try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: librosa/soundfile not available. Audio embeddings will be skipped.")

try:
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    TIMESERIES_AVAILABLE = True
except ImportError:
    TIMESERIES_AVAILABLE = False
    print("Warning: tsfresh not available. Time-series embeddings will use simple features.")

try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: transformers/torch not available. CLIP embeddings will not be generated.")

# Configuration
WEAVIATE_URL = "http://localhost:8080"
CLASS_NAME = "ClientExperience"
DATA_CSV = "mock_data.csv"
IMAGE_DIR = "data/images"
AUDIO_DIR = "data/audio"
TIMESERIES_DIR = "data/timeseries"

# CLIP Model (loaded once, reused for all embeddings)
_clip_model = None
_clip_processor = None

def get_clip_model():
    """Load CLIP model once and reuse it"""
    global _clip_model, _clip_processor
    if _clip_model is None and CLIP_AVAILABLE:
        print("Loading CLIP model...")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()  # Set to evaluation mode
        if torch.cuda.is_available():
            _clip_model = _clip_model.cuda()
        print("CLIP model loaded successfully.")
    return _clip_model, _clip_processor

def create_clip_embedding(text: str, image_b64: str) -> list:
    """
    Create CLIP embedding from text and image.
    Returns a 512-dimensional vector (fused text + image).
    Uses weights: 70% text, 30% image
    """
    if not CLIP_AVAILABLE:
        # Return a zero vector if CLIP is not available
        return [0.0] * 512
    
    try:
        model, processor = get_clip_model()
        if model is None:
            return [0.0] * 512
        
        # Decode base64 image
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        
        # Process inputs
        inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            model = model.cuda()
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            text_emb = outputs.text_embeds[0].cpu().numpy()
            image_emb = outputs.image_embeds[0].cpu().numpy()
        
        # Normalize embeddings
        text_emb = text_emb / np.linalg.norm(text_emb)
        image_emb = image_emb / np.linalg.norm(image_emb)
        
        # Fuse with weights: 70% text, 30% image
        fused = 0.7 * text_emb + 0.3 * image_emb
        fused = fused / np.linalg.norm(fused)  # Renormalize
        
        return fused.tolist()
    except Exception as e:
        print(f"Warning: Could not generate CLIP embedding: {e}")
        return [0.0] * 512

def fuse_vectors(v_clip: list, v_audio: list, v_timeseries: list) -> list:
    """
    Fuse multiple vectors with weights:
    - CLIP (text + image): 60%
    - Audio: 15%
    - Time-Series: 15%
    
    Returns a normalized 512-dimensional vector.
    """
    # Convert to numpy arrays
    v_clip = np.array(v_clip)
    v_audio = np.array(v_audio)
    v_timeseries = np.array(v_timeseries)
    
    # Ensure all vectors are normalized
    if np.linalg.norm(v_clip) > 0:
        v_clip = v_clip / np.linalg.norm(v_clip)
    if np.linalg.norm(v_audio) > 0:
        v_audio = v_audio / np.linalg.norm(v_audio)
    if np.linalg.norm(v_timeseries) > 0:
        v_timeseries = v_timeseries / np.linalg.norm(v_timeseries)
    
    # Weighted fusion
    weights = {
        'clip': 0.6,
        'audio': 0.15,
        'timeseries': 0.15
    }
    
    v_final = weights['clip'] * v_clip + weights['audio'] * v_audio + weights['timeseries'] * v_timeseries
    
    # Normalize the final vector
    norm = np.linalg.norm(v_final)
    if norm > 0:
        v_final = v_final / norm
    else:
        v_final = np.zeros(512)
    
    return v_final.tolist()

def create_placeholder_image(text_snippet: str, image_path: str) -> str:
    """
    Create a placeholder image if it doesn't exist.
    Returns the base64-encoded image string.
    """
    # Create directory if it doesn't exist
    dir_path = os.path.dirname(image_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    # If image already exists, load it
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    
    # Create a simple placeholder image
    img = Image.new('RGB', (512, 512), color=(73, 109, 137))
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to default if not available
    font = ImageFont.load_default()
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf"
    ]
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 24)
                break
        except:
            continue
    
    # Wrap text
    words = text_snippet.split()
    lines = []
    line = ""
    for word in words:
        test_line = line + word + " "
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] <= 450:
            line = test_line
        else:
            if line:
                lines.append(line)
            line = word + " "
    if line:
        lines.append(line)
    
    # Draw text (limit to 10 lines)
    y = 50
    for line in lines[:10]:
        draw.text((30, y), line, fill=(255, 255, 255), font=font)
        y += 40
    
    # Save the image
    img.save(image_path)
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str

def create_audio_embedding(audio_path: str) -> list:
    """
    Create audio embedding using librosa features (simplified OpenL3/VGGish alternative).
    Returns a 512-dimensional vector similar to CLIP dimensions.
    """
    if not AUDIO_AVAILABLE:
        # Return a zero vector if audio processing is not available
        return [0.0] * 512
    
    if not os.path.exists(audio_path):
        # Return a zero vector if file doesn't exist
        return [0.0] * 512
    
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, duration=10.0, sr=22050)
        
        # Extract multiple audio features
        features = []
        
        # MFCC features (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1).tolist())
        features.extend(np.std(mfccs, axis=1).tolist())
        
        # Chroma features (12)
        chroma = librosa.feature.chroma(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1).tolist())
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.append(np.mean(spectral_centroids))
        features.append(np.std(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.append(np.mean(spectral_rolloff))
        features.append(np.std(spectral_rolloff))
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        features.append(np.mean(zero_crossing_rate))
        features.append(np.std(zero_crossing_rate))
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)
        
        # Tonnetz (6 features)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features.extend(np.mean(tonnetz, axis=1).tolist())
        
        # Pad or truncate to 512 dimensions
        if len(features) < 512:
            features.extend([0.0] * (512 - len(features)))
        else:
            features = features[:512]
        
        # Normalize
        features = np.array(features)
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.tolist()
    except Exception as e:
        print(f"Warning: Could not process audio {audio_path}: {e}")
        return [0.0] * 512

def create_timeseries_embedding(timeseries_data: str) -> list:
    """
    Create time-series embedding from JSON string or CSV data.
    Returns a 512-dimensional vector.
    """
    if not timeseries_data or timeseries_data.strip() == "":
        return [0.0] * 512
    
    try:
        # Parse JSON string
        if isinstance(timeseries_data, str):
            data = json.loads(timeseries_data)
        else:
            data = timeseries_data
        
        # Convert to DataFrame format
        if isinstance(data, dict):
            # Assume format: {"time": [1,2,3,...], "value": [10,20,15,...]}
            if "time" in data and "value" in data:
                df = pd.DataFrame({"time": data["time"], "value": data["value"]})
            else:
                # Try to use first two keys
                keys = list(data.keys())
                if len(keys) >= 2:
                    df = pd.DataFrame({"time": data[keys[0]], "value": data[keys[1]]})
                else:
                    return [0.0] * 512
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return [0.0] * 512
        
        # Extract time-series features
        features = []
        
        # Basic statistical features
        if "value" in df.columns:
            values = df["value"].values
            features.extend([
                np.mean(values),
                np.std(values) if len(values) > 1 else 0.0,
                np.min(values),
                np.max(values),
                np.median(values),
                np.percentile(values, 25) if len(values) > 0 else 0.0,
                np.percentile(values, 75) if len(values) > 0 else 0.0,
            ])
            
            # Trend features
            if len(values) > 1:
                diff = np.diff(values)
                features.append(np.mean(diff))
                features.append(np.std(diff))
                
                # Volatility (coefficient of variation)
                if np.mean(values) != 0:
                    features.append(np.std(values) / np.mean(values))
                else:
                    features.append(0.0)
            else:
                features.extend([0.0, 0.0, 0.0])
        
        # Use tsfresh if available for more sophisticated features
        if TIMESERIES_AVAILABLE and len(df) > 10:
            try:
                # Prepare data for tsfresh
                df_tsfresh = df.copy()
                df_tsfresh["id"] = 0  # Single time series
                df_tsfresh = df_tsfresh[["id", "time", "value"]]
                
                extracted = extract_features(df_tsfresh, column_id="id", column_sort="time", 
                                           column_value="value", impute_function=impute)
                if not extracted.empty:
                    # Take top features and pad/truncate
                    ts_features = extracted.iloc[0].values.tolist()
                    features.extend(ts_features[:200])  # Limit to avoid too many features
            except:
                pass  # Fall back to basic features
        
        # Pad or truncate to 512 dimensions
        if len(features) < 512:
            features.extend([0.0] * (512 - len(features)))
        else:
            features = features[:512]
        
        # Normalize
        features = np.array(features)
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.tolist()
    except Exception as e:
        print(f"Warning: Could not process time-series data: {e}")
        return [0.0] * 512

def setup_weaviate_schema(client) -> None:
    """Create the Weaviate schema for ClientExperience class"""
    
    # Check if class already exists and delete it
    try:
        client.collections.delete(CLASS_NAME)
        print(f"Class {CLASS_NAME} already exists. Deleting it...")
    except:
        pass  # Class doesn't exist, which is fine
    
    # Define the schema using v4 API with manual vectorization
    # CRITICAL: vectorizer="none" means we provide vectors manually
    client.collections.create(
        name=CLASS_NAME,
        description="Social services client experiences with multi-modal embeddings (manual vector fusion)",
        vectorizer_config=None,  # Disable automatic vectorization
        properties=[
            wvc.Property(
                name="text",
                data_type=wvc.DataType.TEXT,
                description="Evocative narrative text snippet"
            ),
            wvc.Property(
                name="image",
                data_type=wvc.DataType.BLOB,
                description="Base64-encoded image"
            ),
            wvc.Property(
                name="tag_abstract",
                data_type=wvc.DataType.TEXT,
                description="Abstract subjective concept tag"
            ),
            wvc.Property(
                name="ed_level_primary",
                data_type=wvc.DataType.TEXT,
                description="Household education level"
            ),
            wvc.Property(
                name="religious_participation",
                data_type=wvc.DataType.TEXT,
                description="Level of religious participation"
            ),
            wvc.Property(
                name="image_path",
                data_type=wvc.DataType.TEXT,
                description="Original image file path"
            ),
            wvc.Property(
                name="audio_path",
                data_type=wvc.DataType.TEXT,
                description="Path to audio artifact file"
            ),
            wvc.Property(
                name="audio_vector",
                data_type=wvc.DataType.NUMBER_ARRAY,
                description="Pre-computed audio embedding vector (512 dimensions)"
            ),
            wvc.Property(
                name="time_series_data",
                data_type=wvc.DataType.TEXT,
                description="JSON string of time-series data"
            ),
            wvc.Property(
                name="time_series_vector",
                data_type=wvc.DataType.NUMBER_ARRAY,
                description="Pre-computed time-series embedding vector (512 dimensions)"
            ),
            wvc.Property(
                name="survey_anxiety",
                data_type=wvc.DataType.INT,
                description="Self-reported anxiety level (1-5 scale)"
            ),
            wvc.Property(
                name="survey_control",
                data_type=wvc.DataType.INT,
                description="Self-reported sense of control (1-5 scale)"
            ),
            wvc.Property(
                name="survey_hope",
                data_type=wvc.DataType.INT,
                description="Self-reported level of hope (1-5 scale)"
            ),
        ]
    )
    print(f"Schema created for class: {CLASS_NAME}")

def ingest_data(client, csv_path: str) -> int:
    """Ingest data from CSV into Weaviate"""
    
    # Read CSV using csv.DictReader for better handling of JSON in time_series_data
    # The JSON contains commas which can confuse pandas' CSV parser
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean up the row - remove None keys that might appear
            cleaned_row = {k: v for k, v in row.items() if k is not None}
            rows.append(cleaned_row)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    total_records = len(df)
    print(f"Loaded {total_records} records from {csv_path}")
    
    # Verify we have the expected columns
    expected_cols = ['image_path', 'text_snippet', 'tag_abstract', 'ed_level_primary', 
                     'religious_participation', 'audio_path', 'time_series_data',
                     'survey_anxiety', 'survey_control', 'survey_hope']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
    
    # Get collection
    collection = client.collections.get(CLASS_NAME)
    
    uploaded_count = 0
    with collection.batch.dynamic() as batch:
        row_num = 0
        for idx, row in df.iterrows():
            row_num += 1
            try:
                # Get column values
                image_path = str(row.get('image_path', '')).strip()
                text_snippet = str(row.get('text_snippet', '')).strip()
                
                # Basic validation
                if not image_path or image_path.lower() == 'nan':
                    print(f"Warning: Row {row_num} has empty image_path, skipping...")
                    continue
                
                print(f"Processing {row_num}/{total_records}: {image_path}")
                
                # Get base64 image
                image_b64 = create_placeholder_image(text_snippet, image_path)
                
                # Generate CLIP embedding (text + image fused)
                clip_vector = create_clip_embedding(text_snippet, image_b64)
                
                # Process audio if available
                audio_path = row.get('audio_path', '')
                audio_vector = [0.0] * 512
                if audio_path and pd.notna(audio_path) and audio_path.strip():
                    audio_vector = create_audio_embedding(audio_path)
                
                # Process time-series if available
                time_series_data = row.get('time_series_data', '')
                time_series_vector = [0.0] * 512
                if time_series_data and pd.notna(time_series_data) and time_series_data.strip():
                    time_series_vector = create_timeseries_embedding(time_series_data)
                
                # Fuse all vectors: 60% CLIP, 15% audio, 15% time-series
                fused_vector = fuse_vectors(clip_vector, audio_vector, time_series_vector)
                
                # Get survey ratings (handle NaN values)
                survey_anxiety = int(row.get('survey_anxiety', 0)) if pd.notna(row.get('survey_anxiety', 0)) else None
                survey_control = int(row.get('survey_control', 0)) if pd.notna(row.get('survey_control', 0)) else None
                survey_hope = int(row.get('survey_hope', 0)) if pd.notna(row.get('survey_hope', 0)) else None
                
                # Prepare data object
                data_object = {
                    "text": text_snippet,
                    "image": image_b64,
                    "tag_abstract": row['tag_abstract'],
                    "ed_level_primary": row['ed_level_primary'],
                    "religious_participation": row['religious_participation'],
                    "image_path": image_path,
                    "audio_path": audio_path if audio_path else "",
                    "audio_vector": audio_vector,
                    "time_series_data": time_series_data if time_series_data else "",
                    "time_series_vector": time_series_vector,
                }
                
                # Add survey ratings only if they exist
                if survey_anxiety is not None:
                    data_object["survey_anxiety"] = survey_anxiety
                if survey_control is not None:
                    data_object["survey_control"] = survey_control
                if survey_hope is not None:
                    data_object["survey_hope"] = survey_hope
                
                # Add to batch with fused vector
                batch.add_object(
                    properties=data_object,
                    vector=fused_vector  # CRITICAL: Provide the manually fused vector
                )
                uploaded_count += 1
                
            except Exception as e:
                print(f"Error processing row {row_num}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\nIngestion complete! Uploaded {uploaded_count}/{total_records} records to Weaviate.")
    return uploaded_count

def main():
    """Main ingestion function"""
    print("Connecting to Weaviate...")
    try:
        client = weaviate.connect_to_local(port=8080, grpc_port=50051)
        print("Connected to Weaviate successfully!")
    except Exception as e:
        print(f"ERROR: Cannot connect to Weaviate: {e}")
        print("Make sure it's running on http://localhost:8080")
        print("Start Weaviate with: docker-compose up -d")
        return
    
    try:
        # Setup schema
        print("\nSetting up schema...")
        setup_weaviate_schema(client)
        
        # Ingest data
        print("\nIngesting data...")
        uploaded = ingest_data(client, DATA_CSV)
        
        print(f"\n✅ Data ingestion complete! Uploaded {uploaded} records.")
        print(f"You can now query the database using: python query_demo.py")
    finally:
        client.close()

if __name__ == "__main__":
    main()

