import logging
import os
import datetime
import uuid
import fitz  # PyMuPDF
import docx
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash, jsonify
from openai import OpenAI
from gtts import gTTS
from models import db, Document

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app instance
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}
app.config['PREVIEW_TEXT_MAX_LENGTH'] = 10000  # Limit preview text for large documents
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your-secret-key")

# Configure the database with PostgreSQL
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the database with the app
db.init_app(app)

# Create all database tables
with app.app_context():
    db.create_all()

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Helper function to extract text from PDF files
def extract_text_from_pdf(file_path):
    """Extract text from PDF file using PyMuPDF"""
    try:
        text = ""
        # Open the PDF file
        pdf_document = fitz.open(file_path)
        # Iterate through each page and extract text
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text += page.get_text()
        pdf_document.close()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise

# Helper function to extract text from DOCX files
def extract_text_from_docx(file_path):
    """Extract text from DOCX file using python-docx"""
    try:
        text = ""
        # Open the DOCX file
        doc = docx.Document(file_path)
        # Iterate through each paragraph and extract text
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        raise

# Helper function to generate summary from text using OpenAI
def generate_summary(text):
    """
    Generate a summary of the provided text using OpenAI API
    """
    try:
        # Check if the text is empty
        if not text or len(text.strip()) == 0:
            logger.error("Empty text provided for summarization")
            return None, "Cannot summarize empty text."
            
        # If the text is too long, truncate it to avoid token limits
        # Most models have a token limit of around 4000-8000 tokens, which is roughly 3000-6000 words
        # For safety, we'll limit to approximately 12000 characters (about 2000-3000 words)
        max_text_length = 12000
        truncated_for_api = False
        if len(text) > max_text_length:
            text = text[:max_text_length]
            truncated_for_api = True
            logger.info(f"Text truncated for API call (length: {len(text)})")
        
        # Set up the OpenAI client with API key from environment variable
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OpenAI API key not found in environment variables")
            return None, "OpenAI API key not configured. Please contact the administrator."
            
        client = OpenAI(api_key=openai_api_key)
        
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        prompt = f"Summarize the following text in simple English:\n\n{text}"
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text clearly and concisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        
        # Extract the summary from the response
        summary = response.choices[0].message.content
        
        # Return the summary
        if truncated_for_api:
            summary += "\n\n(Note: The original text was truncated due to length constraints before summarization.)"
            
        return summary, None
        
    except Exception as e:
        logger.error(f"Error generating summary with OpenAI: {e}")
        return None, f"Error generating summary: {str(e)}"

# Helper function to convert text to speech using gTTS
def text_to_speech(text, filename=None):
    """
    Convert text to speech using Google Text-to-Speech (gTTS)
    and save it as an MP3 file in the static/audio directory
    
    Args:
        text (str): The text to convert to speech
        filename (str, optional): A specific filename to use. 
                                If None, a UUID will be generated.
                                
    Returns:
        tuple: (audio_filename, error_message)
    """
    try:
        # Check if text is empty
        if not text or len(text.strip()) == 0:
            logger.error("Empty text provided for text-to-speech conversion")
            return None, "Cannot convert empty text to speech."
        
        # Create the audio directory if it doesn't exist
        audio_dir = os.path.join('static', 'audio')
        os.makedirs(audio_dir, exist_ok=True)
        
        # Generate a unique filename if one is not provided
        if not filename:
            filename = f"speech_{uuid.uuid4().hex}.mp3"
        elif not filename.endswith('.mp3'):
            filename = f"{filename}.mp3"
            
        # Full path to the audio file
        audio_path = os.path.join(audio_dir, filename)
        
        # Limit text length for TTS if needed (gTTS has limits)
        max_tts_length = 5000  # Characters
        if len(text) > max_tts_length:
            text = text[:max_tts_length] + "... Text has been truncated for audio conversion."
            logger.info(f"Text truncated for TTS (length: {len(text)})")
        
        # Create gTTS object
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save the audio file
        tts.save(audio_path)
        logger.info(f"Audio file created: {filename}")
        
        # Return the filename (without the full path)
        return filename, None
        
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {e}")
        return None, f"Error converting text to speech: {str(e)}"

# Define routes
@app.route("/")
def home_page():
    """
    Homepage route that returns a welcome message using Jinja2 template
    """
    logger.debug("Accessing homepage route")
    return render_template(
        "index.html", 
        title="Flask with Jinja2",
        request=request
    )

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """
    Route to handle file uploads
    """
    if request.method == 'POST':
        logger.debug("Processing file upload")
        
        # Check if the post request has the file part
        if 'file' not in request.files:
            logger.error("No file part in the request")
            return render_template(
                "upload.html", 
                title="File Upload",
                error="No file selected"
            )
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            logger.error("No file selected")
            return render_template(
                "upload.html", 
                title="File Upload",
                error="No file selected"
            )
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Get file details
            file_size = os.path.getsize(file_path)
            readable_size = f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024 * 1024):.1f} MB"
            file_type = filename.rsplit('.', 1)[1].lower()
            upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"File uploaded successfully: {filename}")
            
            # Extract text based on file type
            try:
                extracted_text = ""
                if file_type == 'pdf':
                    extracted_text = extract_text_from_pdf(file_path)
                elif file_type == 'docx':
                    extracted_text = extract_text_from_docx(file_path)
                else:
                    logger.error(f"Unsupported file type for text extraction: {file_type}")
                    return render_template(
                        "upload.html",
                        title="File Upload",
                        error=f"Cannot extract text from {file_type.upper()} files"
                    )
                
                # Get the total length of extracted text
                total_length = len(extracted_text)
                
                # Check if we need to truncate the text for the preview
                truncated = False
                if total_length > app.config['PREVIEW_TEXT_MAX_LENGTH']:
                    extracted_text = extracted_text[:app.config['PREVIEW_TEXT_MAX_LENGTH']] + "..."
                    truncated = True
                
                # Render the preview template with the extracted text
                return render_template(
                    "preview.html",
                    title="Text Preview",
                    filename=filename,
                    filetype=file_type.upper(),
                    extracted_text=extracted_text,
                    total_length=total_length,
                    truncated=truncated
                )
                
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
                return render_template(
                    "upload.html",
                    title="File Upload",
                    error=f"Error extracting text: {str(e)}"
                )
        else:
            logger.error("Invalid file type")
            return render_template(
                "upload.html", 
                title="File Upload",
                error="Only PDF and DOCX files are allowed"
            )
    
    # GET request - show upload form
    return render_template(
        "upload.html", 
        title="File Upload"
    )

# Route for text summarization using OpenAI
@app.route('/summarize', methods=['POST'])
def summarize_text():
    """
    Route to handle text summarization requests
    """
    logger.debug("Processing summarization request")
    
    # Get form data
    filename = request.form.get('filename', 'Unknown File')
    filetype = request.form.get('filetype', 'Unknown Type')
    extracted_text = request.form.get('extracted_text', '')
    total_length = request.form.get('total_length', 0)
    try:
        total_length = int(total_length)
    except ValueError:
        total_length = len(extracted_text)
    
    truncated = request.form.get('truncated') == 'true'
    
    # Check if we have text to summarize
    if not extracted_text or len(extracted_text.strip()) == 0:
        logger.error("No text provided for summarization")
        return render_template(
            "upload.html",
            title="File Upload",
            error="No text to summarize. Please upload a file with content."
        )
    
    # Generate summary
    summary, error = generate_summary(extracted_text)
    
    if error:
        logger.error(f"Error in summarization: {error}")
        return render_template(
            "preview.html",
            title="Text Preview",
            filename=filename,
            filetype=filetype,
            extracted_text=extracted_text,
            total_length=total_length,
            truncated=truncated,
            error=f"Failed to generate summary: {error}"
        )
    
    # Generate a unique file identifier for summary text and audio files
    base_filename = os.path.splitext(filename)[0]
    unique_id = uuid.uuid4().hex[:8]
    safe_base_filename = secure_filename(f"{base_filename}_{unique_id}")
    
    # Save the summary to a text file for downloading
    text_filename = f"{safe_base_filename}.txt"
    text_file_path = os.path.join('static', 'summaries', text_filename)
    
    # Create the summaries directory if it doesn't exist
    os.makedirs(os.path.join('static', 'summaries'), exist_ok=True)
    
    # Write the summary to a text file
    try:
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Summary of {filename}\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n\n")
            f.write(summary)
        logger.info(f"Summary text file created: {text_filename}")
    except Exception as e:
        logger.error(f"Error creating summary text file: {e}")
        text_filename = None
    
    # Generate audio from the summary text
    audio_file = None
    audio_error = None
    
    if summary:
        # Use the same base filename for audio to match the text file
        safe_audio_filename = f"{safe_base_filename}"
        
        # Convert the summary to speech
        audio_file, audio_error = text_to_speech(summary, safe_audio_filename)
        
        if audio_error:
            logger.error(f"Error in text-to-speech conversion: {audio_error}")
            # We'll continue without audio if there's an error, but we'll include the error message
    
    # Save to database
    try:
        document = Document(
            filename=filename,
            filetype=filetype,
            summary=summary,
            text_filename=text_filename,
            audio_filename=audio_file,
            upload_time=datetime.datetime.now()
        )
        db.session.add(document)
        db.session.commit()
        logger.info(f"Document record saved to database: {filename}")
    except Exception as e:
        logger.error(f"Error saving document to database: {e}")
        # Continue anyway - don't let database issues prevent summary display
    
    # Render summary template with the audio file and summary text file if available
    return render_template(
        "summary.html",
        title="Document Summary",
        filename=filename,
        filetype=filetype,
        extracted_text=extracted_text,
        total_length=total_length,
        truncated=truncated,
        summary=summary,
        audio_file=audio_file,
        audio_error=audio_error,
        text_filename=text_filename
    )

# Add upload link to the homepage
@app.context_processor
def inject_upload_url():
    return {'upload_url': url_for('upload_file')}

# Serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

# Serve audio files directly if needed
@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """
    Serve audio files directly
    """
    return send_from_directory('static/audio', filename)

# Download a summary text file
@app.route('/download/summary/<path:filename>')
def download_summary_text(filename):
    """
    Download a summary text file
    """
    logger.info(f"Downloading summary text file: {filename}")
    return send_from_directory(
        'static/summaries',
        filename,
        as_attachment=True,
        download_name=filename
    )

# Download an audio file
@app.route('/download/audio/<path:filename>')
def download_audio(filename):
    """
    Download an audio file
    """
    logger.info(f"Downloading audio file: {filename}")
    return send_from_directory(
        'static/audio',
        filename,
        as_attachment=True,
        download_name=filename
    )

# Dashboard route to view upload history
@app.route('/dashboard')
def dashboard():
    """
    Dashboard page displaying document upload history
    """
    logger.debug("Accessing dashboard route")
    
    try:
        # Query all documents from the database, ordered by upload time (newest first)
        documents = Document.query.order_by(Document.upload_time.desc()).all()
        
        # Render the dashboard template with the documents
        return render_template(
            "dashboard.html",
            title="Upload History Dashboard",
            documents=documents
        )
    except Exception as e:
        logger.error(f"Error accessing dashboard: {e}")
        return render_template(
            "index.html",
            title="Error",
            error=f"Could not load dashboard: {str(e)}",
            request=request
        )

# Error handling
@app.errorhandler(404)
def not_found_exception_handler(e):
    """
    Handle 404 errors
    """
    logger.error(f"URL {request.url} not found")
    return render_template(
        "index.html", 
        title="Page Not Found", 
        error="404 - Page not found",
        request=request
    ), 404

@app.errorhandler(500)
def server_error_handler(e):
    """
    Handle 500 errors
    """
    logger.error(f"Server error: {e}")
    return render_template(
        "index.html", 
        title="Server Error", 
        error="500 - Server error",
        request=request
    ), 500

# Run the application
if __name__ == "__main__":
    # Run Flask app
    logger.info("Starting Flask application")
    app.run(host="0.0.0.0", port=5000, debug=True)
