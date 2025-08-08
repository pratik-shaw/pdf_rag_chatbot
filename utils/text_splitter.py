"""
Text Splitting Utilities
Handles intelligent text chunking with overlap for better context preservation.
"""

import re
from typing import List, Dict, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)

class TextSplitter:
    """
    A class to split text into chunks with configurable size and overlap.
    Uses sentence boundaries for more coherent chunks.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size (int): Maximum size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Regex pattern to split on sentence boundaries
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata preservation.
        
        Args:
            text (str): Text to be split
            metadata (Dict): Metadata to be preserved with each chunk
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text or not text.strip():
            return []
        
        # Clean the text
        text = self._clean_text(text)
        
        # Split into sentences first for better boundaries
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_chunk_sentences = []
        
        for sentence in sentences:
            # Check if adding this sentence exceeds chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
                current_chunk_sentences.append(sentence)
            else:
                # Save current chunk if it's not empty
                if current_chunk:
                    chunk_metadata = self._create_chunk_metadata(
                        metadata, len(chunks), len(current_chunk)
                    )
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': chunk_metadata,
                        'sentence_count': len(current_chunk_sentences)
                    })
                
                # Start new chunk with overlap
                overlap_text = self._create_overlap(current_chunk_sentences)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_chunk_sentences = self._get_overlap_sentences(current_chunk_sentences) + [sentence]
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_metadata = self._create_chunk_metadata(
                metadata, len(chunks), len(current_chunk)
            )
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': chunk_metadata,
                'sentence_count': len(current_chunk_sentences)
            })
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex pattern."""
        sentences = self.sentence_pattern.split(text)
        # Filter out empty sentences and strip whitespace
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_overlap(self, sentences: List[str]) -> str:
        """Create overlap text from the end of previous chunk."""
        if not sentences:
            return ""
        
        overlap_text = ""
        # Start from the end and work backwards until we reach overlap limit
        for i in range(len(sentences) - 1, -1, -1):
            potential_overlap = sentences[i] + " " + overlap_text if overlap_text else sentences[i]
            if len(potential_overlap) <= self.chunk_overlap:
                overlap_text = potential_overlap
            else:
                break
        
        return overlap_text.strip()
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences that form the overlap."""
        if not sentences:
            return []
        
        overlap_sentences = []
        overlap_length = 0
        
        # Start from the end and work backwards
        for i in range(len(sentences) - 1, -1, -1):
            sentence = sentences[i]
            if overlap_length + len(sentence) <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)  # Insert at beginning to maintain order
                overlap_length += len(sentence)
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk_metadata(self, original_metadata: Dict[str, Any], 
                              chunk_index: int, chunk_size: int) -> Dict[str, Any]:
        """Create metadata for a chunk."""
        chunk_metadata = {
            'chunk_index': chunk_index,
            'chunk_size': chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
        
        # Merge with original metadata if provided
        if original_metadata:
            chunk_metadata.update(original_metadata)
        
        return chunk_metadata
    
    def get_optimal_chunk_size(self, text: str, max_chunks: int = 100) -> int:
        """
        Calculate optimal chunk size based on text length and desired number of chunks.
        
        Args:
            text (str): Input text
            max_chunks (int): Maximum desired number of chunks
            
        Returns:
            int: Recommended chunk size
        """
        text_length = len(text)
        
        if max_chunks <= 0:
            return self.chunk_size
        
        # Calculate optimal chunk size
        optimal_size = text_length // max_chunks
        
        # Ensure it's within reasonable bounds
        optimal_size = max(500, min(optimal_size, 2000))
        
        return optimal_size