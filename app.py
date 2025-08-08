"""
Streamlit App for PDF RAG Chatbot
Main application interface for Level 1 - PDF RAG with Semantic Search
"""

import streamlit as st
import os
import tempfile
from typing import List
import logging

from rag_system import RAGSystem

# Configure page
st.set_page_config(
    page_title="PDF RAG Chatbot - Level 1",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    if 'pdfs_processed' not in st.session_state:
        st.session_state.pdfs_processed = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}

def save_uploaded_files(uploaded_files) -> List[str]:
    """Save uploaded files to temporary directory and return file paths."""
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            saved_paths.append(tmp_file.name)
    
    return saved_paths

def display_sidebar():
    """Display sidebar with system information and controls."""
    with st.sidebar:
        st.title("📚 PDF RAG Chatbot")
        st.markdown("**Level 1: Basic Semantic Search**")
        
        # System status
        st.subheader("System Status")
        status = st.session_state.rag_system.get_system_status()
        
        if 'error' in status:
            st.error(f"System Error: {status['error']}")
        else:
            if st.session_state.pdfs_processed and status['vector_store_loaded']:
                st.success("✅ System Ready")
                st.metric("Documents in Database", status.get('num_chunks', 0))
                st.metric("Embedding Dimension", status.get('embedding_dimension', 'N/A'))
            else:
                st.warning("⚠️ Please upload PDF documents first")
        
        st.markdown("---")
        
        # Configuration
        st.subheader("Configuration")
        st.info(f"**Chunk Size:** {status.get('chunk_size', 'N/A')}")
        st.info(f"**Chunk Overlap:** {status.get('chunk_overlap', 'N/A')}")
        st.info(f"**Top-K Results:** {status.get('top_k', 'N/A')}")
        st.info(f"**Embedding Model:** {status.get('embedding_model', 'N/A')}")
        
        st.markdown("---")
        
        # Clear data option (only show if data exists)
        if st.session_state.pdfs_processed:
            if st.button("🗑️ Clear All Data", type="secondary"):
                clear_all_data()

def clear_all_data():
    """Clear all processed data and reset the system."""
    # Reset session state
    st.session_state.pdfs_processed = False
    st.session_state.chat_history = []
    st.session_state.processing_status = {}
    
    # Clear vector store files
    try:
        vector_store_path = st.session_state.rag_system.vector_store_path
        for file in ['faiss_index.index', 'chunks_metadata.pkl']:
            file_path = os.path.join(vector_store_path, file)
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Reinitialize RAG system
        st.session_state.rag_system = RAGSystem()
        st.success("All data cleared successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Error clearing data: {e}")

def display_pdf_upload():
    """Display PDF upload interface."""
    st.header("📄 Upload PDF Documents")
    
    # Check if there are existing processed documents
    existing_docs = check_existing_documents()
    if existing_docs:
        st.info("📋 Found previously processed documents. You can load them or upload new ones.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 Load Existing Documents", type="secondary"):
                load_existing_documents()
                return
        with col2:
            if st.button("🗑️ Clear & Upload New", type="secondary"):
                clear_all_data()
                st.rerun()
                return
        
        st.markdown("---")
        st.markdown("**Or upload new PDF documents:**")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to create your knowledge base"
    )
    
    if uploaded_files:
        st.subheader("Selected Files:")
        total_size = 0
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size:,} bytes)")
            total_size += file.size
        
        st.write(f"**Total size:** {total_size:,} bytes")
        
        # Only enable processing if files are selected
        if st.button("🚀 Process PDFs", type="primary"):
            process_uploaded_pdfs(uploaded_files)
    else:
        st.info("👆 Please select PDF files above to continue")

def check_existing_documents() -> bool:
    """Check if there are existing processed documents."""
    vector_store_path = st.session_state.rag_system.vector_store_path
    index_path = os.path.join(vector_store_path, "faiss_index.index")
    metadata_path = os.path.join(vector_store_path, "chunks_metadata.pkl")
    
    return os.path.exists(index_path) and os.path.exists(metadata_path)

def load_existing_documents():
    """Load existing processed documents."""
    with st.spinner("Loading existing documents..."):
        if st.session_state.rag_system.load_vector_store():
            st.session_state.pdfs_processed = True
            status = st.session_state.rag_system.get_system_status()
            st.success(f"✅ Loaded {status.get('num_chunks', 0)} document chunks!")
            st.rerun()
        else:
            st.error("❌ Failed to load existing documents")

def process_uploaded_pdfs(uploaded_files):
    """Process the uploaded PDF files."""
    with st.spinner("Processing PDFs... This may take a few minutes."):
        try:
            # Save uploaded files
            file_paths = save_uploaded_files(uploaded_files)
            
            # Process PDFs
            results = st.session_state.rag_system.process_pdfs(file_paths)
            
            # Clean up temporary files
            for path in file_paths:
                if os.path.exists(path):
                    os.unlink(path)
            
            # Store results
            st.session_state.processing_status = results
            
            if results['success']:
                st.session_state.pdfs_processed = True
                st.success("✅ PDFs processed successfully!")
                st.rerun()
            else:
                st.error(f"❌ Error processing PDFs: {results.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            logger.error(f"PDF processing error: {e}")

def display_processing_results():
    """Display results of PDF processing."""
    if st.session_state.processing_status and st.session_state.processing_status.get('success'):
        results = st.session_state.processing_status
        
        st.success("✅ PDF Processing Complete!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("PDFs Processed", results['num_pdfs'])
        with col2:
            st.metric("Text Chunks Created", results['num_chunks'])
        with col3:
            st.metric("Embedding Dimension", results['embedding_dimension'])
        
        if results.get('pdf_files'):
            st.subheader("Processed Files:")
            for filename in results['pdf_files']:
                st.write(f"• {filename}")

def display_chat_interface():
    """Display the main chat interface."""
    st.header("💬 Ask Questions About Your Documents")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History:")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5 conversations
            with st.expander(f"Q: {chat['query'][:100]}...", expanded=(i == 0)):
                st.markdown(f"**Question:** {chat['query']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                
                if chat.get('sources'):
                    st.markdown("**Sources:**")
                    for j, source in enumerate(chat['sources'][:3]):  # Show top 3 sources
                        st.markdown(f"{j+1}. **{source['filename']}** (Similarity: {source['similarity_score']:.2f})")
                        st.markdown(f"   _{source['chunk_preview'][:150]}..._")
                
                confidence = chat.get('confidence', 0)
                if confidence > 70:
                    st.success(f"Confidence: {confidence:.1f}%")
                elif confidence > 40:
                    st.warning(f"Confidence: {confidence:.1f}%")
                else:
                    st.error(f"Confidence: {confidence:.1f}%")
    
    # Query input
    st.markdown("---")
    query = st.text_area(
        "Ask a question about your uploaded documents:",
        placeholder="e.g., What is the main topic of the document? Can you summarize the key points?",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🔍 Ask Question", type="primary", disabled=not query.strip())
    
    if ask_button and query.strip():
        process_question(query)

def process_question(query):
    """Process a user question and display the response."""
    with st.spinner("Searching documents and generating answer..."):
        try:
            # Get answer from RAG system
            response = st.session_state.rag_system.ask_question(query)
            
            # Add to chat history
            st.session_state.chat_history.append(response)
            
            # Display current response
            st.markdown("---")
            st.subheader("Latest Response:")
            
            st.markdown(f"**Question:** {response['query']}")
            st.markdown(f"**Answer:** {response['answer']}")
            
            # Display sources
            if response.get('sources'):
                st.subheader("Sources:")
                for i, source in enumerate(response['sources']):
                    with st.expander(f"Source {i+1}: {source['filename']} (Similarity: {source['similarity_score']:.2f})"):
                        st.markdown(source['chunk_preview'])
            
            # Display confidence
            confidence = response.get('confidence', 0)
            if confidence > 70:
                st.success(f"High Confidence: {confidence:.1f}%")
            elif confidence > 40:
                st.warning(f"Medium Confidence: {confidence:.1f}%")
            else:
                st.error(f"Low Confidence: {confidence:.1f}%")
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            logger.error(f"Chat error: {e}")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    st.title("📚 PDF RAG Chatbot - Level 1")
    st.markdown("Upload PDF documents and ask questions about their content using semantic search.")
    
    # Check if OpenAI API key is configured
    if not os.getenv('OPENAI_API_KEY'):
        st.error("⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
        st.stop()
    
    # Main application flow
    if not st.session_state.pdfs_processed:
        # Show PDF upload interface
        display_pdf_upload()
    else:
        # Show processing results and chat interface
        display_processing_results()
        st.markdown("---")
        display_chat_interface()
        
        # Option to upload new documents
        st.markdown("---")
        if st.button("📄 Upload New Documents", type="secondary"):
            st.session_state.pdfs_processed = False
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <small>PDF RAG Chatbot Level 1 - Basic Semantic Search | Built with Streamlit, OpenAI, and FAISS</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()