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

# Import required libraries with fallbacks
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = FitzPlaceholder()

try:
    import docx
except ImportError:
    docx = DocxPlaceholder()

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
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/summaries', exist_ok=True)

# Initialize database
with app.app_context():
    # Create tables (they should already exist from SQL initialization)
    db.create_all()
    logger.info("Database tables initialized successfully")

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        
        # Create system message for context
        system_message = f"""You are an AI assistant answering questions about a specific document.
        Only use the document content to answer questions.
        If the answer cannot be found in the document, say so politely.
        Do not make up information or use external knowledge.
        
        DOCUMENT CONTENT:
        {document.text_content[:10000]}  # Limiting to first 10000 chars
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
        flash('Invalid file type. Please upload a PDF or DOCX file')
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
                upload_time=datetime.now()
            )
            db.session.add(document)
            db.session.commit()
            logger.info(f"Document saved to database: {filename}")
            
            # Store document ID in session
            session['last_document_id'] = document.id
            
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
    
    # Retrieve chat history
    chat_messages = ChatMessage.query.filter_by(
        document_id=document_id,
        session_id=session['session_id']
    ).order_by(ChatMessage.created_at).all()
    
    return render_template(
        'document.html',
        document=document,
        chat_messages=[msg.to_dict() for msg in chat_messages]
    )

@app.route('/api/chat/<int:document_id>', methods=['POST'])
def chat_with_document(document_id):
    """API endpoint for chat functionality"""
    # Check if user is in session
    if 'session_id' not in session:
        return jsonify({'error': 'Session expired'}), 401
    
    # Get the message from request
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    user_message = data['message']
    
    # Retrieve document
    document = Document.query.get_or_404(document_id)
    
    # Check if user owns this document
    if document.session_id != session['session_id']:
        return jsonify({'error': 'Unauthorized access'}), 403
    
    # Save user message
    user_chat = ChatMessage(
        document_id=document_id,
        session_id=session['session_id'],
        message_type='user',
        content=user_message
    )
    db.session.add(user_chat)
    db.session.commit()
    
    # Generate AI response
    ai_response, error = generate_chat_response(document_id, user_message)
    
    # Handle errors
    if error:
        return jsonify({'error': error}), 500
    
    # Save AI response
    ai_chat = ChatMessage(
        document_id=document_id,
        session_id=session['session_id'],
        message_type='assistant',
        content=ai_response
    )
    db.session.add(ai_chat)
    db.session.commit()
    
    # Return both messages
    return jsonify({
        'user_message': user_chat.to_dict(),
        'ai_response': ai_chat.to_dict()
    })

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
    
    # Retrieve user's documents
    documents = Document.query.filter_by(session_id=session['session_id']).order_by(Document.upload_time.desc()).all()
    
    return render_template('dashboard.html', documents=documents)

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