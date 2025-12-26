import numpy as np
import os
from typing import List

# Load model globally once to avoid redundant reloads
try:
    from sentence_transformers import SentenceTransformer
    print("⏳ Loading AI Model (this may take a moment)...")
    # 'all-MiniLM-L6-v2' is chosen for being lightweight and fast
    _model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded successfully.")
except ImportError:
    _model = None
    print("WARNING: 'sentence-transformers' not installed. Please run 'pip install sentence-transformers'.")

def get_embedding(text: str) -> List[float]:
    """
    Converts text into a semantic vector (384-dimensional).
    Returns a zero-vector if the model is not loaded or text is empty.
    """
    if _model is None:
        return [0.0] * 384
    
    # Pre-processing: Remove newlines and trim whitespace
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return [0.0] * 384
        
    # Generate embedding and convert numpy array to list
    embedding = _model.encode(clean_text, convert_to_numpy=True)
    return embedding.tolist()

def get_stub_embedding(text: str) -> List[float]:
    """
    Maintained for backward compatibility. 
    Calls the actual embedding model in Iteration 2 context.
    """
    return get_embedding(text)

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Calculates the cosine similarity between two vectors using NumPy.
    Used for reranking hits in the RAG pipeline.
    """
    if not v1 or not v2: 
        return 0.0
    
    # Convert lists to NumPy arrays for faster computation
    array_1 = np.array(v1)
    array_2 = np.array(v2)
    
    # Calculate Frobenius norms
    norm_1 = np.linalg.norm(array_1)
    norm_2 = np.linalg.norm(array_2)
    
    # Avoid division by zero
    if norm_1 == 0 or norm_2 == 0: 
        return 0.0
    
    # Return the dot product divided by the product of norms
    return float(np.dot(array_1, array_2) / (norm_1 * norm_2))