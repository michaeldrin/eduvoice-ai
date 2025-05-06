import os
import logging
import uuid
from datetime import datetime
import json

# Define placeholder classes for libraries that might not be installed
class FitzPlaceholder:
    def open(self, *args, **kwargs):
        raise ImportError("PyMuPDF (fitz) module is not installed")

class DocxPlaceholder:
    def Document(self, *args, **kwargs):
        raise ImportError("python-docx module is not installed")

class PILPlaceholder:
    def open(self, *args, **kwargs):
        raise ImportError("PIL/Pillow module is not installed")

class PytesseractPlaceholder:
    def image_to_string(self, *args, **kwargs):
        raise ImportError("pytesseract module is not installed")

# Import required libraries with fallbacks
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = FitzPlaceholder()

try:
    import docx
except ImportError:
    docx = DocxPlaceholder()

try:
    from PIL import Image
except ImportError:
    Image = PILPlaceholder()

try:
    import pytesseract
except ImportError:
    pytesseract = PytesseractPlaceholder()

from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from flask import send_from_directory, flash
from werkzeug.utils import secure_filename
from openai import OpenAI
from models import db, Document, ChatMessage

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_key")

# Custom Jinja2 filter for converting newlines to <br>
@app.template_filter('nl2br')
def nl2br(value):
    if value:
        return value.replace('\n', '<br>')
    return value

# Add global template variables
@app.context_processor
def inject_global_variables():
    return {
        'current_year': datetime.now().year
    }

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ATTACHMENT_FOLDER = os.path.join(UPLOAD_FOLDER, 'attachments')
# Define file extension groups
PDF_EXTENSIONS = {'pdf'}
DOCX_EXTENSIONS = {'docx'}
TEXT_EXTENSIONS = {'txt'}
IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
ALLOWED_EXTENSIONS = PDF_EXTENSIONS.union(DOCX_EXTENSIONS).union(TEXT_EXTENSIONS).union(IMAGE_EXTENSIONS)

# Define supported OCR languages
OCR_LANGUAGES = {
    'eng': 'English',
    'deu': 'German',
    'fra': 'French',
    'spa': 'Spanish',
    'ara': 'Arabic',
    'fas': 'Persian',
    'urd': 'Urdu',
    'rus': 'Russian',
    'ukr': 'Ukrainian',
    'chi_sim': 'Chinese (Simplified)',
    'chi_tra': 'Chinese (Traditional)',
    'jpn': 'Japanese',
    'kor': 'Korean'
}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ATTACHMENT_FOLDER'] = ATTACHMENT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ATTACHMENT_FOLDER, exist_ok=True)
os.makedirs('static/summaries', exist_ok=True)

# Initialize database
with app.app_context():
    # Create tables (they should already exist from SQL initialization)
    db.create_all()
    logger.info("Database tables initialized successfully")

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to check if a file is an image
def is_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS

# Helper function to extract text from image files using OCR
def extract_text_from_image(file_path, language='eng'):
    """
    Extract text from image file using pytesseract OCR
    
    Args:
        file_path: Path to the image file
        language: OCR language. Defaults to 'eng' (English).
                  Can be 'eng' (English), 'deu' (German), 'fas' (Persian),
                  'fra' (French), 'spa' (Spanish), etc. or combined with '+' 
                  like 'eng+deu' for multiple languages.
    
    Returns:
        Tuple of (extracted_text, error_message)
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Image file not found: {file_path}")
            return None, "Image file not found. Please upload the file again."
        
        # Check if pytesseract is properly installed
        if isinstance(pytesseract, PytesseractPlaceholder):
            logger.error("pytesseract module is not installed")
            return None, "Image OCR is currently unavailable. The server is missing required libraries."
        
        # Open the image with PIL
        image = Image.open(file_path)
        
        # Try to detect script/language if not specified
        # First try with default English
        text = pytesseract.image_to_string(image, lang='eng')
        
        # If English doesn't give good results, try other common languages
        if not text or len(text.strip()) < 10:  # Very little text was found
            logger.info(f"Trying multi-language OCR for image: {file_path}")
            
            # Try common languages one by one
            for lang_code in ['eng+deu+fra+spa', 'ara+fas+urd', 'rus+ukr+bel', 'chi_sim+chi_tra+jpn+kor']:
                try:
                    lang_text = pytesseract.image_to_string(image, lang=lang_code)
                    if lang_text and len(lang_text.strip()) > len(text.strip()):
                        text = lang_text
                        logger.info(f"Better OCR results with language(s): {lang_code}")
                except Exception as lang_err:
                    logger.warning(f"Error trying language {lang_code}: {lang_err}")
                    continue
        
        # Preprocess the image if text extraction was poor
        if not text or len(text.strip()) < 10:
            logger.info(f"Trying image preprocessing for better OCR: {file_path}")
            try:
                # Convert to grayscale
                gray_image = image.convert('L')
                
                # Apply thresholding to improve contrast
                threshold_image = gray_image.point(lambda x: 0 if x < 128 else 255, '1')
                
                # Try OCR again on the processed image
                processed_text = pytesseract.image_to_string(threshold_image)
                
                if processed_text and len(processed_text.strip()) > len(text.strip()):
                    text = processed_text
                    logger.info("Image preprocessing improved OCR results")
            except Exception as process_err:
                logger.warning(f"Error during image preprocessing: {process_err}")
        
        # Check final result
        if not text or len(text.strip()) == 0:
            logger.warning(f"No text extracted from image: {file_path}")
            return None, "No readable text found in the image file. The image may not contain clear text."
            
        # Add info about multi-language support
        info_text = "This text was extracted using OCR (Optical Character Recognition) technology with multi-language support.\n\n"
        return info_text + text.strip(), None
        
    except Exception as e:
        logger.exception(f"Error extracting text from image: {e}")
        return None, f"Error processing image file: {str(e)}"

# Helper function to extract text from plain text files
def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Text file not found: {file_path}")
            return None, "Text file not found. Please upload the file again."
        
        # Open and read the text file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        # Check if we got any text
        if not text or len(text.strip()) == 0:
            logger.warning(f"No text extracted from text file: {file_path}")
            return None, "The text file appears to be empty."
            
        return text, None
        
    except Exception as e:
        logger.exception(f"Error extracting text from text file: {e}")
        return None, f"Error processing text file: {str(e)}"

# Helper function to extract text from PDF files
def extract_text_from_pdf(file_path):
    """Extract text from PDF file using PyMuPDF"""
    try:
        text = ""
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"PDF file not found: {file_path}")
            return None, "PDF file not found. Please upload the file again."
        
        # Check if PyMuPDF is properly installed
        if isinstance(fitz, FitzPlaceholder):
            logger.error("PyMuPDF (fitz) module is not installed")
            return None, "PDF processing is currently unavailable. The server is missing required libraries."
         
        # Open and extract text from the PDF
        with fitz.open(file_path) as pdf_document:
            if len(pdf_document) == 0:
                logger.warning(f"PDF has no pages: {file_path}")
                return None, "The PDF file doesn't contain any pages."
                
            # Extract text from each page
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text()
                text += page_text
            
        # Check if we got any text
        if not text or len(text.strip()) == 0:
            logger.warning(f"No text extracted from PDF: {file_path}")
            return None, "No readable text found in the PDF file. It may contain only images or be password protected."
            
        return text, None
        
    except Exception as e:
        logger.exception(f"Error extracting text from PDF: {e}")
        return None, f"Error processing PDF file: {str(e)}"

# Helper function to extract text from DOCX files
def extract_text_from_docx(file_path):
    """Extract text from DOCX file using python-docx"""
    try:
        text = ""
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"DOCX file not found: {file_path}")
            return None, "DOCX file not found. Please upload the file again."
        
        # Check if python-docx is properly installed
        if isinstance(docx, DocxPlaceholder):
            logger.error("python-docx module is not installed")
            return None, "DOCX processing is currently unavailable. The server is missing required libraries."
            
        # Open and extract text from the DOCX
        doc = docx.Document(file_path)
        
        # Extract text from paragraphs
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
            
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text += cell.text + "\n"
        
        # Check if we got any text
        if not text or len(text.strip()) == 0:
            logger.warning(f"No text extracted from DOCX: {file_path}")
            return None, "No readable text found in the DOCX file."
            
        return text, None
        
    except Exception as e:
        logger.exception(f"Error extracting text from DOCX: {e}")
        return None, f"Error processing DOCX file: {str(e)}"

# Helper function to generate summary using OpenAI
def generate_summary(text):
    """Generate a summary of the text using OpenAI API"""
    try:
        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found in environment variables")
            return None, "OpenAI API key not configured. Please contact the administrator."
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Create prompt for summarization
        prompt = f"""Please summarize the following document in a detailed yet concise manner. 
        Focus on the main points, key arguments, and significant findings.
        Maintain the original context and meaning while condensing the content.
        
        DOCUMENT:
        {text[:10000]}  # Limiting input to first 10000 chars to avoid token limits
        
        SUMMARY:"""
        
        # Generate summary
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content.strip()
        return summary, None
        
    except Exception as e:
        logger.exception(f"Error generating summary: {e}")
        return None, f"Error generating summary: {str(e)}"

# Generate personalized interaction tips
def generate_interaction_tips(document_summary, filetype):
    """
    Generate personalized document interaction tips based on the document summary
    
    Args:
        document_summary: The summary of the document
        filetype: The type of document (PDF, DOCX, etc.)
        
    Returns:
        List of interaction tips, or None and error message if generation fails
    """
    try:
        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found in environment variables")
            return None, "OpenAI API key not configured. Please contact the administrator."
            
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Create prompt for generating tips
        prompt = f"""Based on the following document summary, generate 5 personalized tips for interacting with and learning from this document.
        These tips should help the user ask better questions and get more value from the AI chat feature.
        Each tip should be specific to the document content, not generic advice.
        Format the response as a list of tips, each starting with a number and a brief title, followed by a short explanation.
        
        Document type: {filetype}
        Document summary: {document_summary[:1000]}  # Limiting to first 1000 chars
        
        INTERACTION TIPS:"""
        
        # Generate tips
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.7
        )
        
        tips = response.choices[0].message.content.strip()
        return tips, None
        
    except Exception as e:
        logger.exception(f"Error generating interaction tips: {e}")
        return None, f"Error generating tips: {str(e)}"

# Simple text similarity search function
def simple_text_search(query, document_text, chunk_size=1000, overlap=200, top_k=3):
    """
    Perform a simple text similarity search to find relevant document chunks
    
    Args:
        query: The search query (user question)
        document_text: The document text to search within
        chunk_size: Size of each document chunk
        overlap: Overlap between chunks
        top_k: Number of most relevant chunks to return
    
    Returns:
        List of most relevant text chunks
    """
    try:
        # Normalize query (lowercase, remove extra spaces)
        query = " ".join(query.lower().split())
        
        # Break document into overlapping chunks
        chunks = []
        doc_length = len(document_text)
        
        for i in range(0, doc_length, chunk_size - overlap):
            chunk = document_text[i:i + chunk_size]
            if len(chunk) < 100:  # Skip very small chunks
                continue
            chunks.append(chunk)
        
        # If we have no chunks, return the whole document
        if not chunks:
            return [document_text[:5000]]  # Return first 5000 chars
            
        # Calculate simple relevance score for each chunk
        chunk_scores = []
        query_terms = set(query.split())
        
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            
            # Count how many query terms appear in the chunk
            term_matches = sum(1 for term in query_terms if term in chunk_lower)
            
            # Count exact phrase matches (more weight)
            phrase_matches = chunk_lower.count(query) * 3
            
            # Calculate final score
            score = term_matches + phrase_matches
            
            chunk_scores.append((i, score, chunk))
        
        # Sort by score (highest first) and take top_k
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [chunk for _, _, chunk in chunk_scores[:top_k]]
        
        # If no chunks matched, return first chunk as fallback
        if not top_chunks or all(score == 0 for _, score, _ in chunk_scores[:top_k]):
            return [chunks[0]]
            
        return top_chunks
        
    except Exception as e:
        logger.exception(f"Error in similarity search: {e}")
        # Fallback to first 5000 chars if search fails
        return [document_text[:5000]]

# Generate response for chat using OpenAI
def generate_chat_response(document_id, user_message):
    """Generate a response to a user question about a document using OpenAI API"""
    try:
        # Retrieve document
        document = Document.query.get(document_id)
        if not document or not document.text_content:
            return None, "Document not found or has no content"
        
        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found in environment variables")
            return None, "OpenAI API key not configured. Please contact the administrator."
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Perform text similarity search to find relevant chunks
        relevant_chunks = simple_text_search(
            user_message, 
            document.text_content,
            chunk_size=1500,
            overlap=300, 
            top_k=3
        )
        
        # Combine relevant chunks for context (limit total to ~5000 chars)
        combined_chunks = " ".join(relevant_chunks)
        if len(combined_chunks) > 5000:
            combined_chunks = combined_chunks[:5000]
        
        # Create system message with enhanced instructions and context
        system_message = f"""You are an AI assistant answering questions about a specific document.
        The following are the most relevant sections of the document related to the user's question.
        
        DOCUMENT CONTENT:
        {combined_chunks}
        
        Instructions:
        1. Only use the document content provided above to answer questions.
        2. If the answer cannot be found in these document sections, say so politely.
        3. Do not make up information or use external knowledge.
        4. If the document content is in a language other than English, answer in that same language.
        5. Format your response neatly with paragraphs where appropriate.
        """
        
        # Generate response
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content.strip()
        return answer, None
        
    except Exception as e:
        logger.exception(f"Error generating chat response: {e}")
        return None, f"Error generating response: {str(e)}"

# Routes
@app.route('/')
def index():
    """Home page with file upload form"""
    # Generate a session ID if none exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        
    # Initialize session storage for uploads and chat history if not exists
    if 'uploads' not in session:
        session['uploads'] = []
        
    if 'chat_history' not in session:
        session['chat_history'] = {}
        
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
        
    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
        
    # Check file type
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a PDF, DOCX, TXT, JPG, JPEG, or PNG file')
        return redirect(url_for('index'))
    
    try:
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text based on file type
        filetype = filename.rsplit('.', 1)[1].lower()
        if filetype == 'pdf':
            extracted_text, error = extract_text_from_pdf(file_path)
        elif filetype == 'docx':
            extracted_text, error = extract_text_from_docx(file_path)
        elif filetype == 'txt':
            extracted_text, error = extract_text_from_txt(file_path)
        elif filetype in ['jpg', 'jpeg', 'png']:
            # Inform user about OCR processing through the logger
            logger.info(f"Processing image with OCR: {filename}")
            
            # Extract text using OCR
            extracted_text, error = extract_text_from_image(file_path)
            
            # Add OCR indicator to inform the user
            if extracted_text and not error:
                ocr_info = (
                    "[TEXT EXTRACTED USING OCR TECHNOLOGY]\n\n"
                    "This text was extracted from your image using Optical Character Recognition (OCR) "
                    "with multi-language support. The quality of extraction depends on image clarity "
                    "and text formatting. You can ask questions about this content just like with any "
                    "text document.\n\n"
                    "---------------------------------------------------\n\n"
                )
                extracted_text = ocr_info + extracted_text
                logger.info(f"Successfully extracted text from image: {filename}")
        else:
            # This shouldn't happen due to allowed_file check, but just in case
            extracted_text, error = None, "Unsupported file type"
        
        # Handle extraction errors
        if error:
            flash(f"Error: {error}")
            return redirect(url_for('index'))
        
        # Generate summary
        summary, summary_error = generate_summary(extracted_text)
        
        # Handle summary generation errors
        if summary_error:
            flash(f"Error generating summary: {summary_error}")
            return redirect(url_for('index'))
            
        # Generate interaction tips
        tips, tips_error = generate_interaction_tips(summary, filetype)
        
        # If tips generation failed, log the error but continue (non-critical)
        if tips_error:
            logger.warning(f"Error generating interaction tips: {tips_error}")
            tips = None
        
        # Initialize document variable
        document = None
        
        # Store in database
        try:
            document = Document(
                session_id=session['session_id'],
                filename=filename,
                filetype=filetype,
                summary=summary,
                text_content=extracted_text,
                interaction_tips=tips,
                upload_time=datetime.now()
            )
            db.session.add(document)
            db.session.commit()
            logger.info(f"Document saved to database: {filename}")
            
            # Store document ID in session
            session['last_document_id'] = document.id
            
            # Store document metadata in session for history tracking
            session_document = {
                'id': document.id,
                'filename': document.filename,
                'filetype': document.filetype,
                'summary': summary[:200] if summary else "",  # Store the first 200 chars
                'upload_time': document.upload_time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add to the beginning of the list (most recent first)
            session['uploads'].insert(0, session_document)
            
            # Initialize empty chat history for this document
            session['chat_history'][str(document.id)] = []
            
            # Make sure to persist the session
            session.modified = True
            
        except Exception as db_error:
            logger.error(f"Database error: {db_error}")
            db.session.rollback()
            flash("Error saving document to database")
            return redirect(url_for('index'))
        
        # Redirect to document view
        return redirect(url_for('view_document', document_id=document.id))
        
    except Exception as e:
        logger.exception(f"Error processing file upload: {e}")
        flash(f"Error: {str(e)}")
        return redirect(url_for('index'))

@app.route('/document/<int:document_id>')
def view_document(document_id):
    """View uploaded document with summary and chat"""
    # Retrieve document
    document = Document.query.get_or_404(document_id)
    
    # Check if user owns this document (session-based)
    if document.session_id != session.get('session_id'):
        flash("You don't have permission to view this document")
        return redirect(url_for('index'))
    
    # Retrieve chat history from database
    chat_messages = ChatMessage.query.filter_by(
        document_id=document_id,
        session_id=session['session_id']
    ).order_by(ChatMessage.created_at).all()
    
    # Initialize chat history in session if not exists
    if 'chat_history' not in session:
        session['chat_history'] = {}
    
    # If we don't have chat history for this document in session, initialize it
    doc_id_str = str(document_id)
    if doc_id_str not in session['chat_history']:
        session['chat_history'][doc_id_str] = []
        session.modified = True
    
    # Convert database messages to dict format for template rendering
    db_messages = [msg.to_dict() for msg in chat_messages]
    
    return render_template(
        'document.html',
        document=document,
        chat_messages=db_messages,
        session_chat_history=session['chat_history'].get(doc_id_str, [])
    )

@app.route('/api/chat/<int:document_id>', methods=['POST'])
def chat_with_document(document_id):
    """API endpoint for chat functionality"""
    try:
        # Check if user is in session
        if 'session_id' not in session:
            return jsonify({'error': 'Session expired. Please refresh the page.'}), 401
        
        # Check if it's a JSON request or a file upload (multipart/form-data)
        if request.is_json:
            # Handle standard text message
            data = request.get_json()
            if not data or 'message' not in data:
                return jsonify({'error': 'No message provided'}), 400
            
            user_message = data['message'].strip()
            if not user_message:
                return jsonify({'error': 'Empty message'}), 400
                
            has_attachment = False
            attachment_filename = None
            attachment_original_filename = None
            attachment_type = None
            attachment_text = None
        else:
            # This might be a file upload with form data
            user_message = request.form.get('message', '').strip()
            if not user_message and 'file' not in request.files:
                return jsonify({'error': 'No message or file provided'}), 400
                
            # Initialize attachment variables
            has_attachment = False
            attachment_filename = None
            attachment_original_filename = None
            attachment_type = None
            attachment_text = None
            
            # Check if there's a file attached
            if 'file' in request.files:
                file = request.files['file']
                if file and file.filename and allowed_file(file.filename):
                    # Process the attachment
                    has_attachment = True
                    original_filename = secure_filename(file.filename)
                    filename_parts = os.path.splitext(original_filename)
                    extension = filename_parts[1].lower()
                    
                    # Generate a unique filename
                    unique_filename = f"{filename_parts[0]}_{uuid.uuid4().hex}{extension}"
                    file_path = os.path.join(app.config['ATTACHMENT_FOLDER'], unique_filename)
                    
                    # Save the file
                    file.save(file_path)
                    
                    # Set attachment information
                    attachment_filename = unique_filename
                    attachment_original_filename = original_filename
                    
                    # Determine attachment type and extract text if possible
                    if extension in ['.pdf']:
                        attachment_type = 'pdf'
                        extracted_text, error = extract_text_from_pdf(file_path)
                        if extracted_text:
                            attachment_text = extracted_text
                    elif extension in ['.docx']:
                        attachment_type = 'docx'
                        extracted_text, error = extract_text_from_docx(file_path)
                        if extracted_text:
                            attachment_text = extracted_text
                    elif extension in ['.txt']:
                        attachment_type = 'txt'
                        extracted_text, error = extract_text_from_txt(file_path)
                        if extracted_text:
                            attachment_text = extracted_text
                    elif extension in ['.jpg', '.jpeg', '.png']:
                        attachment_type = 'image'
                        # Log OCR processing for attachment
                        logger.info(f"Processing attachment image with OCR: {original_filename}")
                        extracted_text, error = extract_text_from_image(file_path)
                        if extracted_text:
                            # Add OCR header for image attachments
                            ocr_note = "[Text extracted from the attached image using OCR technology]\n\n"
                            attachment_text = ocr_note + extracted_text
                            logger.info(f"Successfully extracted text from image attachment: {original_filename}")
        
        # Retrieve document
        document = Document.query.get(document_id)
        if not document:
            return jsonify({'error': 'Document not found'}), 404
            
        # Check if document has content
        if not document.text_content:
            return jsonify({
                'error': 'This document has no extractable text content',
                'details': 'The file may be an image-based PDF or contain unsupported formatting.'
            }), 422
        
        # Check if user owns this document
        if document.session_id != session['session_id']:
            return jsonify({'error': 'Unauthorized access'}), 403
        
        # Check for OpenAI API key
        if not os.environ.get("OPENAI_API_KEY"):
            return jsonify({
                'error': 'OpenAI API key not configured',
                'details': 'Please contact the administrator to set up the API key.'
            }), 503
        
        # Begin a database transaction to handle potential errors
        db.session.begin_nested()
        
        # Save user message with attachment info if present
        user_chat = ChatMessage(
            document_id=document_id,
            session_id=session['session_id'],
            message_type='user',
            content=user_message,
            has_attachment=has_attachment,
            attachment_filename=attachment_filename,
            attachment_original_filename=attachment_original_filename,
            attachment_type=attachment_type,
            attachment_text=attachment_text
        )
        db.session.add(user_chat)
        db.session.commit()
        
        # Prepare context for AI response
        # If there's attachment text, include it in the context
        additional_context = ""
        if attachment_text:
            additional_context = f"\n\nThe user has also attached a file with the following content:\n{attachment_text[:2000]}"
        
        # Generate AI response
        ai_response, error = generate_chat_response(document_id, user_message + additional_context)
        
        # Handle errors during response generation
        if error:
            # Create an error response message in chat
            error_message = f"I'm sorry, I couldn't process your request: {error}"
            
            ai_chat = ChatMessage(
                document_id=document_id,
                session_id=session['session_id'],
                message_type='assistant',
                content=error_message
            )
            db.session.add(ai_chat)
            db.session.commit()
            
            return jsonify({
                'user_message': user_chat.to_dict(),
                'ai_response': ai_chat.to_dict(),
                'warning': error
            }), 200  # Still return 200 to display the error in the chat
        
        # Save AI response
        ai_chat = ChatMessage(
            document_id=document_id,
            session_id=session['session_id'],
            message_type='assistant',
            content=ai_response
        )
        db.session.add(ai_chat)
        db.session.commit()
        
        # Store messages in session for history tracking
        doc_id_str = str(document_id)
        
        # Add user message to session history
        session['chat_history'][doc_id_str].append({
            'role': 'user',
            'content': user_message,
            'has_attachment': has_attachment,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Add AI response to session history
        session['chat_history'][doc_id_str].append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Make sure to persist the session
        session.modified = True
        
        # Return both messages
        return jsonify({
            'user_message': user_chat.to_dict(),
            'ai_response': ai_chat.to_dict()
        })
        
    except Exception as e:
        # Log the error
        logger.exception(f"Error in chat API: {e}")
        
        # Rollback any transaction in progress
        db.session.rollback()
        
        # Return error response
        return jsonify({
            'error': 'An unexpected error occurred',
            'details': str(e)
        }), 500

@app.route('/download/summary/<int:document_id>')
def download_summary(document_id):
    """Download document summary as text file"""
    # Retrieve document
    document = Document.query.get_or_404(document_id)
    
    # Check if user owns this document
    if document.session_id != session.get('session_id'):
        flash("You don't have permission to download this summary")
        return redirect(url_for('index'))
    
    # Create summary file
    summary_filename = f"summary_{document_id}.txt"
    summary_path = os.path.join('static/summaries', summary_filename)
    
    # Write summary to file
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Summary of: {document.filename}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(document.summary)
    
    # Send file to user
    return send_from_directory(
        'static/summaries',
        summary_filename,
        as_attachment=True,
        download_name=f"Summary-{document.filename}.txt"
    )

@app.route('/dashboard')
def dashboard():
    """User dashboard showing uploaded documents"""
    # Check if user is in session
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        
    # Initialize session storage if not exists
    if 'uploads' not in session:
        session['uploads'] = []
        
    if 'chat_history' not in session:
        session['chat_history'] = {}
    
    # Retrieve user's documents from database
    db_documents = Document.query.filter_by(session_id=session['session_id']).order_by(Document.upload_time.desc()).all()
    
    # Combine database documents with session documents
    # Session documents have priority to show the most up-to-date information
    session_doc_ids = [doc['id'] for doc in session['uploads']]
    
    # Filter out documents that are already in session
    additional_docs = []
    for doc in db_documents:
        if doc.id not in session_doc_ids:
            # Add to session for future reference
            session_document = {
                'id': doc.id,
                'filename': doc.filename,
                'filetype': doc.filetype,
                'summary': doc.summary[:200] if doc.summary else "",
                'upload_time': doc.upload_time.strftime("%Y-%m-%d %H:%M:%S")
            }
            session['uploads'].append(session_document)
            session.modified = True
    
    # Pass the session documents to the template
    return render_template('dashboard.html', 
                          session_documents=session['uploads'],
                          show_history_controls=True)

# Route to serve chat attachment files
@app.route('/attachments/<path:filename>')
def serve_attachment(filename):
    """Serve attachment files"""
    # Check if user is in session
    if 'session_id' not in session:
        flash("Please login to access attachments")
        return redirect(url_for('index'))
    
    # Check if the attachment exists in database and belongs to current session
    chat_message = ChatMessage.query.filter_by(
        attachment_filename=filename,
        session_id=session['session_id']
    ).first()
    
    if not chat_message:
        flash("Attachment not found or you don't have permission to access it")
        return redirect(url_for('index'))
    
    # Serve the file
    return send_from_directory(app.config['ATTACHMENT_FOLDER'], filename)

# Route to clear session history
@app.route('/clear-history')
def clear_history():
    """Clear session history of uploads and chat"""
    if 'uploads' in session:
        session.pop('uploads')
    
    if 'chat_history' in session:
        session.pop('chat_history')
    
    # Make sure to persist changes
    session.modified = True
    
    flash("Your browsing history has been cleared.", "success")
    return redirect(url_for('dashboard'))

# Static file routes
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(error):
    return render_template('error.html', error="Internal server error"), 500

# Run the application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)