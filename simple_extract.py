import os
import sys
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add necessary imports
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import pdfplumber

class FitzPlaceholder:
    def open(self, *args, **kwargs):
        pass

# Basic extraction function - similar to what we added to main.py but simplified
def extract_pdf_text(file_path):
    """Extract text and images from PDF"""
    try:
        text = ""
        image_texts = []
        table_texts = []
        total_pages = 0
        
        # Open and extract text from the PDF
        pdf_document = fitz.open(file_path)
        
        if pdf_document.page_count == 0:
            pdf_document.close()
            return "The PDF file doesn't contain any pages.", {"error": "Empty document"}
            
        # Store page count before we start processing
        total_pages = pdf_document.page_count
            
        # Extract text from each page
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            page_text = page.get_text()
            text += page_text
            
            # Extract images from this page
            image_list = page.get_images(full=True)
            logger.info(f"Found {len(image_list)} images on page {page_num+1}")
            
            # Process each image on the page
            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]  # Get reference number for the image
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Create a PIL Image object from the bytes
                    img = Image.open(io.BytesIO(image_bytes))
                    width, height = img.size
                    
                    # Skip small images (likely icons or decorative elements)
                    if width < 100 or height < 100:
                        continue
                    
                    # Try OCR on the image
                    ocr_text = pytesseract.image_to_string(img)
                    
                    # Only add if meaningful text was found (more than 10 chars)
                    if ocr_text and len(ocr_text.strip()) > 10:
                        image_texts.append({
                            'page': page_num + 1,
                            'text': ocr_text.strip()
                        })
                        logger.info(f"Extracted OCR text from image on page {page_num+1}")
                    
                except Exception as img_err:
                    logger.warning(f"Error processing image on page {page_num+1}: {str(img_err)}")
        
        # Close PDF document after extraction
        pdf_document.close()
        
        # Try basic table extraction with pdfplumber
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    
                    if tables:
                        logger.info(f"Found {len(tables)} tables on page {page_num+1}")
                        for table_idx, table in enumerate(tables):
                            # Convert table to text format
                            if table and len(table) > 0:
                                table_text = ""
                                
                                # Process each row
                                for row in table:
                                    # Filter None values and convert to strings
                                    processed_row = [str(cell).strip() if cell is not None else "" for cell in row]
                                    # Add row text
                                    table_text += " | ".join(processed_row) + "\n"
                                
                                if table_text.strip():
                                    table_texts.append({
                                        'page': page_num + 1,
                                        'text': table_text.strip()
                                    })
                                    logger.info(f"Extracted table {table_idx+1} from page {page_num+1}")
        except Exception as table_err:
            logger.warning(f"Error extracting tables: {str(table_err)}")
        
        # Combine all extracted content
        combined_text = text
        
        # Add image texts if any were found
        for img_data in image_texts:
            combined_text += f"\n\n[Extracted from image on page {img_data['page']}]\n{img_data['text']}"
        
        # Add table texts if any were found
        for table_data in table_texts:
            combined_text += f"\n\n[Extracted from table on page {table_data['page']}]\n{table_data['text']}"
        
        # Summary of extraction results
        summary = {
            "total_pages": total_pages,
            "text_chars": len(text),
            "images_processed": len(image_texts),
            "tables_processed": len(table_texts),
            "total_chars": len(combined_text)
        }
        
        return combined_text, summary
        
    except Exception as e:
        logger.exception(f"Error extracting text from PDF: {e}")
        return None, {"error": str(e)}

if __name__ == "__main__":
    # Process a small PDF file from the uploads directory
    uploads_dir = "uploads"
    
    # Find PDF files and sort by size
    pdf_files = [(f, os.path.getsize(os.path.join(uploads_dir, f))) 
                for f in os.listdir(uploads_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        logger.error("No PDF files found in uploads directory")
        sys.exit(1)
    
    # Sort by file size and test with the smallest one
    pdf_files.sort(key=lambda x: x[1])
    test_pdf = os.path.join(uploads_dir, pdf_files[0][0])
    
    logger.info(f"Testing extraction on: {test_pdf} ({pdf_files[0][1]/1024:.2f} KB)")
    
    # Extract text and print results
    extracted_text, summary = extract_pdf_text(test_pdf)
    
    if extracted_text:
        # Log summary info
        logger.info(f"Extraction summary: {json.dumps(summary, indent=2)}")
        
        # Print a sample of the extracted text
        logger.info("Sample of extracted text (first 20 lines):")
        for line in extracted_text.split('\n')[:20]:
            print(line)
        
        # Look for image and table sections
        img_sections = [i for i, line in enumerate(extracted_text.split('\n')) 
                      if '[Extracted from image on page' in line]
        table_sections = [i for i, line in enumerate(extracted_text.split('\n')) 
                        if '[Extracted from table on page' in line]
        
        # Print image extraction sections if any
        if img_sections:
            logger.info(f"Found {len(img_sections)} image extraction sections")
            lines = extracted_text.split('\n')
            for idx in img_sections[:2]:  # Show first 2 only
                print("\n--- Image Extraction ---")
                # Print a few lines around this section
                for i in range(max(0, idx-1), min(len(lines), idx+10)):
                    print(lines[i])
        
        # Print table extraction sections if any
        if table_sections:
            logger.info(f"Found {len(table_sections)} table extraction sections")
            lines = extracted_text.split('\n')
            for idx in table_sections[:2]:  # Show first 2 only
                print("\n--- Table Extraction ---")
                # Print a few lines around this section
                for i in range(max(0, idx-1), min(len(lines), idx+10)):
                    print(lines[i])
    else:
        logger.error(f"Extraction failed: {summary.get('error', 'Unknown error')}")