"""
Core RAG System
Handles the complete RAG pipeline from document processing to answer generation.
"""

import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from openai import OpenAI
from dotenv import load_dotenv

from utils import PDFProcessor, TextSplitter, EmbeddingGenerator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Complete RAG system for PDF-based question answering.
    Handles document processing, embedding, storage, retrieval, and generation.
    """
    
    def __init__(self, 
                 vector_store_path: str = "./vector_store",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 top_k: int = 5):
        """
        Initialize the RAG system.
        
        Args:
            vector_store_path (str): Path to store vector database
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
            top_k (int): Number of top results to retrieve
        """
        self.vector_store_path = vector_store_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_generator = EmbeddingGenerator()
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Vector store components
        self.index = None
        self.chunks_metadata = []
        
        # Ensure vector store directory exists
        os.makedirs(vector_store_path, exist_ok=True)
        
        logger.info("RAG System initialized successfully")
    
    def process_pdfs(self, pdf_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple PDF files and create vector store.
        
        Args:
            pdf_paths (List[str]): List of PDF file paths
            
        Returns:
            Dict: Processing results and statistics
        """
        try:
            logger.info(f"Processing {len(pdf_paths)} PDF files")
            
            # Step 1: Extract text from PDFs
            pdf_data = self.pdf_processor.extract_text_from_multiple_pdfs(pdf_paths)
            
            if not pdf_data:
                raise Exception("No valid PDFs could be processed")
            
            # Step 2: Split text into chunks
            all_chunks = []
            for pdf_content in pdf_data:
                chunks = self.text_splitter.split_text(
                    text=pdf_content['text'],
                    metadata=pdf_content['metadata']
                )
                all_chunks.extend(chunks)
            
            logger.info(f"Created {len(all_chunks)} text chunks")
            
            # Step 3: Generate embeddings
            chunks_with_embeddings = self.embedding_generator.batch_process_chunks(all_chunks)
            
            # Step 4: Create and save vector store
            self._create_vector_store(chunks_with_embeddings)
            
            # Prepare results
            results = {
                'success': True,
                'num_pdfs': len(pdf_data),
                'num_chunks': len(chunks_with_embeddings),
                'embedding_dimension': self.embedding_generator.get_embedding_dimension(),
                'pdf_files': [pdf['metadata']['filename'] for pdf in pdf_data]
            }
            
            logger.info("PDF processing completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error processing PDFs: {e}")
            return {
                'success': False,
                'error': str(e),
                'num_pdfs': 0,
                'num_chunks': 0
            }
    
    def _create_vector_store(self, chunks_with_embeddings: List[Dict[str, Any]]):
        """Create FAISS vector store from chunks with embeddings."""
        try:
            # Extract embeddings and metadata
            embeddings = []
            metadata = []
            
            for chunk in chunks_with_embeddings:
                if 'embedding' in chunk:
                    embeddings.append(chunk['embedding'])
                    metadata.append({
                        'text': chunk['text'],
                        'metadata': chunk['metadata'],
                        'sentence_count': chunk.get('sentence_count', 0)
                    })
            
            if not embeddings:
                raise Exception("No embeddings found in processed chunks")
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Create FAISS index
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Add embeddings to index
            self.index.add(embeddings_array)
            
            # Store metadata
            self.chunks_metadata = metadata
            
            # Save to disk
            self._save_vector_store()
            
            logger.info(f"Vector store created with {len(embeddings)} embeddings")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def _save_vector_store(self):
        """Save vector store and metadata to disk."""
        try:
            # Save FAISS index
            index_path = os.path.join(self.vector_store_path, "faiss_index.index")
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            metadata_path = os.path.join(self.vector_store_path, "chunks_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.chunks_metadata, f)
            
            logger.info("Vector store saved to disk")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def load_vector_store(self) -> bool:
        """Load vector store from disk."""
        try:
            index_path = os.path.join(self.vector_store_path, "faiss_index.index")
            metadata_path = os.path.join(self.vector_store_path, "chunks_metadata.pkl")
            
            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                logger.info("Vector store not found on disk")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.chunks_metadata = pickle.load(f)
            
            logger.info(f"Vector store loaded with {len(self.chunks_metadata)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def retrieve_relevant_chunks(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks for a query.
        
        Args:
            query (str): User query
            
        Returns:
            List[Dict]: Retrieved chunks with similarity scores
        """
        try:
            if self.index is None:
                if not self.load_vector_store():
                    raise Exception("No vector store available")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(query)
            
            if query_embedding.size == 0:
                return []
            
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search for similar chunks
            scores, indices = self.index.search(query_embedding, self.top_k)
            
            # Prepare results
            retrieved_chunks = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks_metadata):
                    chunk_data = self.chunks_metadata[idx].copy()
                    chunk_data['similarity_score'] = float(score)
                    chunk_data['rank'] = i + 1
                    retrieved_chunks.append(chunk_data)
            
            logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks")
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer using OpenAI API based on retrieved chunks.
        
        Args:
            query (str): User query
            retrieved_chunks (List[Dict]): Retrieved relevant chunks
            
        Returns:
            Dict: Generated answer with metadata
        """
        try:
            if not retrieved_chunks:
                return {
                    'answer': "I couldn't find relevant information to answer your question.",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Prepare context from retrieved chunks
            context_parts = []
            sources = []
            
            for chunk in retrieved_chunks:
                context_parts.append(f"Source: {chunk['metadata']['filename']}\n{chunk['text']}")
                sources.append({
                    'filename': chunk['metadata']['filename'],
                    'similarity_score': chunk['similarity_score'],
                    'chunk_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                })
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Create prompt
            prompt = f"""Based on the following context from PDF documents, please answer the user's question. 
If the answer is not directly available in the context, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=os.getenv('CHAT_MODEL', 'gpt-3.5-turbo'),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided PDF content. Be accurate and cite specific information when possible."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            # Calculate confidence based on similarity scores
            avg_similarity = np.mean([chunk['similarity_score'] for chunk in retrieved_chunks])
            confidence = min(avg_similarity * 100, 100)  # Convert to percentage
            
            result = {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'num_sources': len(sources)
            }
            
            logger.info("Answer generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': [],
                'confidence': 0.0
            }
    
    def ask_question(self, query: str) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve relevant chunks and generate answer.
        
        Args:
            query (str): User question
            
        Returns:
            Dict: Complete response with answer and metadata
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self.retrieve_relevant_chunks(query)
            
            # Step 2: Generate answer
            response = self.generate_answer(query, retrieved_chunks)
            
            # Add query to response
            response['query'] = query
            response['timestamp'] = str(np.datetime64('now'))
            
            return response
            
        except Exception as e:
            logger.error(f"Error in ask_question: {e}")
            return {
                'query': query,
                'answer': f"Error processing question: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'error': str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics."""
        try:
            status = {
                'vector_store_loaded': self.index is not None,
                'num_chunks': len(self.chunks_metadata) if self.chunks_metadata else 0,
                'embedding_model': self.embedding_generator.model_name,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'top_k': self.top_k
            }
            
            if self.index is not None:
                status['index_size'] = self.index.ntotal
                status['embedding_dimension'] = self.index.d
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}