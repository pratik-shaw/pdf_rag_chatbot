"""
Embedding Generation Utilities
Handles text embedding generation using SentenceTransformers for semantic search.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    A class to generate embeddings using SentenceTransformers.
    Uses all-MiniLM-L6-v2 model for good performance and speed balance.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name (str): Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the SentenceTransformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise Exception(f"Could not load embedding model: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        if not texts:
            return np.array([])
        
        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            
            if not valid_texts:
                logger.warning("No valid texts to embed")
                return np.array([])
            
            logger.info(f"Generating embeddings for {len(valid_texts)} texts")
            
            # Generate embeddings
            embeddings = self.model.encode(
                valid_texts,
                convert_to_numpy=True,
                show_progress_bar=len(valid_texts) > 10  # Show progress for large batches
            )
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise Exception(f"Failed to generate embeddings: {e}")
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        if not text or not text.strip():
            return np.array([])
        
        try:
            embedding = self.model.encode([text.strip()], convert_to_numpy=True)
            return embedding[0]  # Return single embedding vector
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            raise Exception(f"Failed to generate embedding: {e}")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.
        
        Returns:
            int: Embedding dimension
        """
        try:
            # Generate a test embedding to get dimension
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            return test_embedding.shape[1]
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}")
            # Default dimension for all-MiniLM-L6-v2
            return 384
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding
            
        Returns:
            float: Cosine similarity score
        """
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def batch_process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of chunks and add embeddings to them.
        
        Args:
            chunks (List[Dict]): List of chunk dictionaries
            
        Returns:
            List[Dict]: Chunks with embeddings added
        """
        if not chunks:
            return []
        
        try:
            # Extract texts from chunks
            texts = [chunk.get('text', '') for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Add embeddings to chunks
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    chunk_copy = chunk.copy()
                    chunk_copy['embedding'] = embeddings[i]
                    chunk_copy['embedding_model'] = self.model_name
                    processed_chunks.append(chunk_copy)
                else:
                    logger.warning(f"No embedding generated for chunk {i}")
                    processed_chunks.append(chunk)
            
            logger.info(f"Processed {len(processed_chunks)} chunks with embeddings")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error in batch processing chunks: {e}")
            raise Exception(f"Failed to process chunks: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict: Model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.get_embedding_dimension(),
            'max_sequence_length': getattr(self.model, 'max_seq_length', 'Unknown'),
            'model_loaded': self.model is not None
        }