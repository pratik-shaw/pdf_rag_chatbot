"""
PDF Processing Utilities
Handles extraction of text from PDF files with error handling and metadata preservation.
"""

import PyPDF2
import os
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    A class to handle PDF text extraction with metadata preservation.
    Simple approach using PyPDF2 for reliability.
    """
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from a single PDF file with metadata.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict containing extracted text, metadata, and page information
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"File must have .pdf extension: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                metadata = {
                    'filename': os.path.basename(pdf_path),
                    'filepath': pdf_path,
                    'num_pages': len(pdf_reader.pages),
                    'title': getattr(pdf_reader.metadata, 'title', '') if pdf_reader.metadata else '',
                    'author': getattr(pdf_reader.metadata, 'author', '') if pdf_reader.metadata else ''
                }
                
                # Extract text from all pages
                pages_text = []
                full_text = ""
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        pages_text.append({
                            'page_number': page_num,
                            'text': page_text,
                            'char_count': len(page_text)
                        })
                        full_text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
                        pages_text.append({
                            'page_number': page_num,
                            'text': "",
                            'char_count': 0,
                            'error': str(e)
                        })
                
                result = {
                    'text': full_text.strip(),
                    'metadata': metadata,
                    'pages': pages_text,
                    'total_chars': len(full_text)
                }
                
                logger.info(f"Successfully extracted {len(full_text)} characters from {pdf_path}")
                return result
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise Exception(f"Failed to process PDF: {e}")
    
    def extract_text_from_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Extract text from multiple PDF files.
        
        Args:
            pdf_paths (List[str]): List of paths to PDF files
            
        Returns:
            List of dictionaries containing extracted text and metadata
        """
        results = []
        
        for pdf_path in pdf_paths:
            try:
                result = self.extract_text_from_pdf(pdf_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                # Continue processing other PDFs even if one fails
                continue
        
        logger.info(f"Successfully processed {len(results)} out of {len(pdf_paths)} PDFs")
        return results
    
    def validate_pdf_file(self, pdf_path: str) -> bool:
        """
        Validate if the file is a readable PDF.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            bool: True if valid PDF, False otherwise
        """
        try:
            if not os.path.exists(pdf_path):
                return False
            
            if not pdf_path.lower().endswith('.pdf'):
                return False
            
            # Try to open and read the PDF
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                # Try to access first page to verify it's readable
                if len(pdf_reader.pages) > 0:
                    pdf_reader.pages[0].extract_text()
                return True
        except Exception:
            return False