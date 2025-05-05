import logging
import os
import datetime
import uuid
import fitz  # PyMuPDF
import docx
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash, jsonify, session
from openai import OpenAI
from gtts import gTTS
from models import db, Document, UserSettings

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
        
        # Get user settings for language preference
        settings = get_user_settings()
        language = settings.language
        
        # Map language codes to language names for the prompt
        language_names = {
            'en': 'English',
            'fa': 'Farsi',
            'de': 'German',
            'fr': 'French',
            'es': 'Spanish'
        }
        
        # Default to English if language is not in our mapping
        language_name = language_names.get(language, 'English')
        
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        prompt = f"Summarize the following text in simple {language_name}:\n\n{text}"
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant that summarizes text clearly and concisely in {language_name}."},
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

# Helper function to get or create user settings
def get_user_settings():
    """
    Get or create user settings for the current session
    
    Returns:
        UserSettings: The user settings for the current session
    """
    # Generate a session_id if not present
    if 'session_id' not in session:
        session['session_id'] = uuid.uuid4().hex
    
    session_id = session['session_id']
    
    # Try to find existing settings
    settings = UserSettings.query.filter_by(session_id=session_id).first()
    
    # Create new settings if not found
    if not settings:
        settings = UserSettings(session_id=session_id)
        db.session.add(settings)
        db.session.commit()
        logger.info(f"Created new user settings for session: {session_id}")
    
    return settings

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
        
        # Get user settings for language and voice speed
        settings = get_user_settings()
        
        # Default to English if language not supported by gTTS
        language = settings.language if settings.language in ['en', 'de', 'fr', 'es', 'it'] else 'en'
        
        # Set the speaking rate
        slow_speech = settings.voice_speed == 'slow'
        
        # Create gTTS object with user settings
        tts = gTTS(text=text, lang=language, slow=slow_speech)
        
        # Save the audio file
        tts.save(audio_path)
        logger.info(f"Audio file created: {filename} in language: {language}")
        
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
    
    # Get the current user settings
    user_settings = get_user_settings()
    
    return render_template(
        "index.html", 
        title="Document Processor",
        request=request,
        usage_stats=user_settings
    )

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """
    Route to handle file uploads
    """
    # Get the current user settings
    user_settings = get_user_settings()
    
    if request.method == 'POST':
        logger.debug("Processing file upload")
        
        # Check if user has reached daily upload limit
        if not user_settings.can_upload_file():
            logger.warning(f"User {user_settings.session_id} has reached daily upload limit")
            return render_template(
                "upload.html", 
                title="File Upload",
                error="You've reached your daily upload limit (5 files per day). Please try again tomorrow or upgrade your account.",
                usage_stats=user_settings
            )
        
        # Check if the post request has the file part
        if 'file' not in request.files:
            logger.error("No file part in the request")
            return render_template(
                "upload.html", 
                title="File Upload",
                error="No file selected",
                usage_stats=user_settings
            )
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            logger.error("No file selected")
            return render_template(
                "upload.html", 
                title="File Upload",
                error="No file selected",
                usage_stats=user_settings
            )
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Increment the file upload counter
            user_settings.increment_file_uploads()
            db.session.commit()
            logger.info(f"Incremented file uploads for user {user_settings.session_id}: {user_settings.files_uploaded}")
            
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
                        error=f"Cannot extract text from {file_type.upper()} files",
                        usage_stats=user_settings
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
                    truncated=truncated,
                    usage_stats=user_settings
                )
                
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
                return render_template(
                    "upload.html",
                    title="File Upload",
                    error=f"Error extracting text: {str(e)}",
                    usage_stats=user_settings
                )
        else:
            logger.error("Invalid file type")
            return render_template(
                "upload.html", 
                title="File Upload",
                error="Only PDF and DOCX files are allowed",
                usage_stats=user_settings
            )
    
    # GET request - show upload form
    return render_template(
        "upload.html", 
        title="File Upload",
        usage_stats=user_settings
    )

# Route for text summarization using OpenAI
@app.route('/summarize', methods=['POST'])
def summarize_text():
    """
    Route to handle text summarization requests
    """
    logger.debug("Processing summarization request")
    
    # Get the current user settings
    user_settings = get_user_settings()
    
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
            error="No text to summarize. Please upload a file with content.",
            usage_stats=user_settings
        )
    
    # Check if user has reached summary generation limit
    if not user_settings.can_generate_summary():
        logger.warning(f"User {user_settings.session_id} has reached summary generation limit")
        return render_template(
            "preview.html",
            title="Text Preview",
            filename=filename,
            filetype=filetype,
            extracted_text=extracted_text,
            total_length=total_length,
            truncated=truncated,
            error="You've reached your summary generation limit (10 summaries total). Please upgrade your account for more.",
            usage_stats=user_settings
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
            error=f"Failed to generate summary: {error}",
            usage_stats=user_settings
        )
    
    # Increment summary counter
    user_settings.increment_summaries()
    db.session.commit()
    logger.info(f"Incremented summaries for user {user_settings.session_id}: {user_settings.summaries_generated}")
    
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
    
    # Generate audio from the summary text - if within limits
    audio_file = None
    audio_error = None
    
    if summary:
        # Estimate audio duration in minutes (average speaking rate is about 150 words per minute)
        # Split by spaces to count words, multiply by 60 seconds, divide by 150 words per minute
        estimated_audio_minutes = len(summary.split()) / 150
        
        # Check if user has audio minutes left
        if user_settings.can_generate_audio(estimated_audio_minutes):
            # Use the same base filename for audio to match the text file
            safe_audio_filename = f"{safe_base_filename}"
            
            # Convert the summary to speech
            audio_file, audio_error = text_to_speech(summary, safe_audio_filename)
            
            if audio_error:
                logger.error(f"Error in text-to-speech conversion: {audio_error}")
                # We'll continue without audio if there's an error, but we'll include the error message
            else:
                # Add the estimated audio minutes to the user's total
                user_settings.add_audio_minutes(estimated_audio_minutes)
                db.session.commit()
                logger.info(f"Added {estimated_audio_minutes:.2f} audio minutes for user {user_settings.session_id}: {user_settings.audio_minutes:.2f}")
        else:
            audio_error = "You've reached your audio generation limit (15 minutes total). Please upgrade your account for more."
            logger.warning(f"User {user_settings.session_id} has reached audio generation limit")
    
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
        text_filename=text_filename,
        usage_stats=user_settings
    )

# Add global template variables
@app.context_processor
def inject_global_variables():
    """
    Make common variables available to all templates
    """
    # Get user settings and make them available in all templates
    try:
        user_settings = get_user_settings()
    except Exception as e:
        logger.warning(f"Error getting user settings for template: {e}")
        user_settings = None
        
    return {
        'upload_url': url_for('upload_file'),
        'user_settings': user_settings,
        'theme_mode': user_settings.theme_mode if user_settings else 'dark'
    }

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
    
    # Get the current user settings
    user_settings = get_user_settings()
    
    try:
        # Query all documents from the database, ordered by upload time (newest first)
        documents = Document.query.order_by(Document.upload_time.desc()).all()
        
        # Render the dashboard template with the documents
        return render_template(
            "dashboard.html",
            title="Upload History Dashboard",
            documents=documents,
            usage_stats=user_settings
        )
    except Exception as e:
        logger.error(f"Error accessing dashboard: {e}")
        return render_template(
            "index.html",
            title="Error",
            error=f"Could not load dashboard: {str(e)}",
            request=request,
            usage_stats=user_settings
        )

# Settings route to customize user preferences
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """
    Settings page for user preferences
    """
    logger.debug("Accessing settings route")
    
    # Get the current user settings
    user_settings = get_user_settings()
    success_message = None
    error_message = None
    
    if request.method == 'POST':
        try:
            # Get form data for language, voice speed, and theme
            language = request.form.get('language', 'en')
            voice_speed = request.form.get('voice_speed', 'normal')
            theme_mode = request.form.get('theme_mode', 'dark')
            
            # Update the user settings
            user_settings.language = language
            user_settings.voice_speed = voice_speed
            user_settings.theme_mode = theme_mode
            
            # Save the changes
            db.session.commit()
            
            # Set success message
            success_message = "Settings updated successfully!"
            logger.info(f"User settings updated: language={language}, voice_speed={voice_speed}, theme_mode={theme_mode}")
            
        except Exception as e:
            error_message = f"Error updating settings: {str(e)}"
            logger.error(f"Error updating settings: {e}")
    
    # Prepare available options for the form
    languages = [
        {'code': 'en', 'name': 'English'},
        {'code': 'de', 'name': 'German'},
        {'code': 'fr', 'name': 'French'},
        {'code': 'es', 'name': 'Spanish'},
        {'code': 'fa', 'name': 'Farsi'}
    ]
    
    voice_speeds = [
        {'code': 'slow', 'name': 'Slow'},
        {'code': 'normal', 'name': 'Normal'},
        {'code': 'fast', 'name': 'Fast'}
    ]
    
    theme_modes = [
        {'code': 'dark', 'name': 'Dark Mode'},
        {'code': 'light', 'name': 'Light Mode'}
    ]
    
    # Render the settings template with the form and current settings
    return render_template(
        "settings.html",
        title="User Settings",
        user_settings=user_settings,
        languages=languages,
        voice_speeds=voice_speeds,
        theme_modes=theme_modes,
        success=success_message,
        error=error_message,
        usage_stats=user_settings
    )

# Error handling
@app.errorhandler(404)
def not_found_exception_handler(e):
    """
    Handle 404 errors
    """
    logger.error(f"URL {request.url} not found")
    
    # Get the current user settings if possible
    try:
        user_settings = get_user_settings()
    except Exception:
        user_settings = None
        
    return render_template(
        "index.html", 
        title="Page Not Found", 
        error="404 - Page not found",
        request=request,
        usage_stats=user_settings
    ), 404

@app.errorhandler(500)
def server_error_handler(e):
    """
    Handle 500 errors
    """
    logger.error(f"Server error: {e}")
    
    # Get the current user settings if possible
    try:
        user_settings = get_user_settings()
    except Exception:
        user_settings = None
        
    return render_template(
        "index.html", 
        title="Server Error", 
        error="500 - Server error",
        request=request,
        usage_stats=user_settings
    ), 500

# Run the application
if __name__ == "__main__":
    # Run Flask app
    logger.info("Starting Flask application")
    app.run(host="0.0.0.0", port=5000, debug=True)
