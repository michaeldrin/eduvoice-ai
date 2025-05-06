import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the extraction functions from main.py
sys.path.append('.')
try:
    from main import extract_text_from_pdf, extract_text_from_docx
    logger.info("Successfully imported extraction functions")
except ImportError as e:
    logger.error(f"Error importing extraction functions: {e}")
    sys.exit(1)

def test_pdf_extraction(pdf_path):
    """Test the PDF extraction with a specific file"""
    logger.info(f"Testing PDF extraction on: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return
    
    # Extract text and check for errors
    extracted_text, error = extract_text_from_pdf(pdf_path)
    
    if error:
        logger.error(f"Extraction error: {error}")
        return
    
    if not extracted_text:
        logger.warning("No text extracted")
        return
    
    logger.info(f"Successfully extracted {len(extracted_text)} characters of text")
    
    # Check for image and table extraction markers
    image_sections = [line for line in extracted_text.split('\n') if '[Extracted from image on page' in line]
    table_sections = [line for line in extracted_text.split('\n') if '[Extracted from table on page' in line]
    
    logger.info(f"Found {len(image_sections)} image sections and {len(table_sections)} table sections")
    
    # Print a sample of the extracted text
    logger.info("Sample of extracted text:")
    sample_lines = extracted_text.split('\n')[:20]  # First 20 lines
    for line in sample_lines:
        print(line)
    
    # Print all image and table extraction sections
    if image_sections:
        logger.info("Image extraction sections:")
        for i, section in enumerate(image_sections):
            section_start = extracted_text.find(section)
            section_content = extracted_text[section_start:section_start+500]  # Show 500 chars after the section marker
            print(f"--- Image Section {i+1} ---")
            print(section_content)
            print()
    
    if table_sections:
        logger.info("Table extraction sections:")
        for i, section in enumerate(table_sections):
            section_start = extracted_text.find(section)
            section_content = extracted_text[section_start:section_start+500]  # Show 500 chars after the section marker
            print(f"--- Table Section {i+1} ---")
            print(section_content)
            print()

def test_docx_extraction(docx_path):
    """Test the DOCX extraction with a specific file"""
    logger.info(f"Testing DOCX extraction on: {docx_path}")
    
    if not os.path.exists(docx_path):
        logger.error(f"DOCX file not found: {docx_path}")
        return
    
    # Extract text and check for errors
    extracted_text, error = extract_text_from_docx(docx_path)
    
    if error:
        logger.error(f"Extraction error: {error}")
        return
    
    if not extracted_text:
        logger.warning("No text extracted")
        return
    
    logger.info(f"Successfully extracted {len(extracted_text)} characters of text")
    
    # Check for image and table extraction markers
    image_sections = [line for line in extracted_text.split('\n') if '[Extracted from image' in line]
    table_sections = [line for line in extracted_text.split('\n') if '[Extracted from table' in line]
    
    logger.info(f"Found {len(image_sections)} image sections and {len(table_sections)} table sections")
    
    # Print a sample of the extracted text
    logger.info("Sample of extracted text:")
    sample_lines = extracted_text.split('\n')[:20]  # First 20 lines
    for line in sample_lines:
        print(line)
    
    # Print all image and table extraction sections
    if image_sections:
        logger.info("Image extraction sections:")
        for i, section in enumerate(image_sections):
            section_start = extracted_text.find(section)
            section_content = extracted_text[section_start:section_start+500]  # Show 500 chars after the section marker
            print(f"--- Image Section {i+1} ---")
            print(section_content)
            print()
    
    if table_sections:
        logger.info("Table extraction sections:")
        for i, section in enumerate(table_sections):
            section_start = extracted_text.find(section)
            section_content = extracted_text[section_start:section_start+500]  # Show 500 chars after the section marker
            print(f"--- Table Section {i+1} ---")
            print(section_content)
            print()

if __name__ == "__main__":
    # Look for PDF files in the uploads directory
    uploads_dir = "uploads"
    
    # Try with a small PDF first - choose the smallest file in the uploads directory
    pdf_files = [(f, os.path.getsize(os.path.join(uploads_dir, f))) 
                for f in os.listdir(uploads_dir) if f.endswith('.pdf')]
    
    if pdf_files:
        # Sort by file size and choose the smallest
        pdf_files.sort(key=lambda x: x[1])
        small_pdf = os.path.join(uploads_dir, pdf_files[0][0])
        logger.info(f"Testing with smallest PDF file: {small_pdf} ({pdf_files[0][1]/1024:.2f} KB)")
        test_pdf_extraction(small_pdf)
    else:
        logger.warning("No PDF files found in uploads directory")
    
    # Look for DOCX files to test
    docx_files = [f for f in os.listdir(uploads_dir) if f.endswith('.docx')]
    
    if docx_files:
        # Choose a DOCX file to test
        test_docx = os.path.join(uploads_dir, docx_files[0])
        test_docx_extraction(test_docx)
    else:
        logger.warning("No DOCX files found in uploads directory")