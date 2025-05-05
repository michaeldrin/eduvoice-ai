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
from oauth import init_oauth, auth_bp, login_required

# Set up logging for debugging
log_file_path = os.path.join('logs', 'app.log')
os.makedirs('logs', exist_ok=True)

# Configure logging with file and console handlers
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        # File handler for persistent logs
        logging.FileHandler(log_file_path),
        # Stream handler for console output
        logging.StreamHandler()
    ]
)
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

# Initialize OAuth with the app
init_oauth(app)

# Print a message to help with Google OAuth setup
replit_domain = os.environ.get("REPLIT_DEV_DOMAIN")
if replit_domain:
    print("\n====================== GOOGLE OAUTH SETUP =======================")
    print("To make Google authentication work, add this to your authorized")
    print("redirect URIs in Google Cloud Console:")
    print(f"https://{replit_domain}/callback")
    print("================================================================\n")

# Register the auth blueprint
app.register_blueprint(auth_bp)

# Register the callback route at the root level for Google OAuth
@app.route('/callback')
def oauth_callback():
    """Route to handle Google OAuth callback at root level"""
    # Log detailed information about the callback for debugging
    logger.info(f"==== ROOT LEVEL CALLBACK RECEIVED ====")
    logger.info(f"Timestamp: {datetime.datetime.now().isoformat()}")
    logger.info(f"Full callback URL: {request.url}")
    logger.info(f"HTTP Method: {request.method}")
    logger.info(f"Query parameters: {request.args}")
    logger.info(f"HTTP Headers: {dict(request.headers)}")
    
    # Display the Replit domain for verification
    replit_domain = os.environ.get('REPLIT_DEV_DOMAIN')
    if replit_domain:
        logger.info(f"Replit domain: {replit_domain}")
        
        # Compare URL with expected format
        expected_callback = f"https://{replit_domain}/callback"
        logger.info(f"Expected callback URL format: {expected_callback}")
        
        # Extract actual URL base for comparison
        from urllib.parse import urlparse
        actual_url = request.url
        parsed = urlparse(actual_url)
        actual_base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        logger.info(f"Actual URL base: {actual_base}")
        
        # Log mismatch if present
        if actual_base != expected_callback and actual_base != expected_callback.replace('https://', 'http://'):
            logger.warning(f"Potential URL mismatch - actual: {actual_base}, expected: {expected_callback}")
    
    # Check for direct error in the Google response
    if 'error' in request.args:
        error_msg = request.args.get('error')
        error_description = request.args.get('error_description', 'No description provided')
        logger.error(f"OAuth error parameter detected: {error_msg}")
        logger.error(f"Error description: {error_description}")
        
        # Handle common error cases
        if error_msg == 'redirect_uri_mismatch':
            logger.error("This is a redirect URI mismatch error - make sure Google Console has the exact URI")
            logger.error(f"Expected URI: https://{replit_domain}/callback")
            
            # Print to console for visibility
            print(f"\n============== OAUTH ERROR: REDIRECT URI MISMATCH ==============")
            print(f"The redirect URI in your Google Console doesn't match what Replit expects")
            print(f"Add exactly this URI to Google Cloud Console OAuth 2.0 Client ID settings:")
            print(f"https://{replit_domain}/callback")
            print(f"==============================================================\n")
    
    # Pass to the auth blueprint's handler
    logger.info("Forwarding callback to auth blueprint handler")
    return auth_bp.handle_google_callback()
    
# This function has been moved to combine with the existing inject_global_variables

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
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"PDF file not found: {file_path}")
            return None, "PDF file not found. Please upload the file again."
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.error(f"PDF file is empty: {file_path}")
            return None, "The uploaded PDF file is empty."
            
        # Try to open the PDF file
        try:
            pdf_document = fitz.open(file_path)
        except Exception as open_error:
            logger.error(f"Failed to open PDF file: {open_error}")
            return None, "The PDF file appears to be corrupted or in an unsupported format."
            
        # If PDF has no pages
        if len(pdf_document) == 0:
            pdf_document.close()
            logger.warning(f"PDF has no pages: {file_path}")
            return None, "The PDF file doesn't contain any pages."
            
        # Iterate through each page and extract text
        try:
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text()
                text += page_text
        except Exception as read_error:
            pdf_document.close()
            logger.error(f"Error reading PDF content: {read_error}")
            return None, "Failed to extract text from the PDF. It may be encrypted, damaged, or contain unsupported content."
        
        # Close the document
        pdf_document.close()
        
        # Check if we got any text
        if not text or len(text.strip()) == 0:
            logger.warning(f"No text extracted from PDF: {file_path}")
            return None, "No readable text found in the PDF file. It may contain only images or be password protected."
            
        return text, None
        
    except Exception as e:
        logger.exception(f"Unexpected error extracting text from PDF: {e}")
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
            
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.error(f"DOCX file is empty: {file_path}")
            return None, "The uploaded DOCX file is empty."
        
        # Try to open the DOCX file
        try:
            doc = docx.Document(file_path)
        except Exception as open_error:
            logger.error(f"Failed to open DOCX file: {open_error}")
            return None, "The DOCX file appears to be corrupted or in an unsupported format."
        
        # Iterate through each paragraph and extract text
        try:
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
                
            # If document has tables, process those too
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                    text += "\n"
                text += "\n"
        except Exception as read_error:
            logger.error(f"Error reading DOCX content: {read_error}")
            return None, "Failed to extract text from the DOCX. It may contain unsupported content."
            
        # Check if we got any text
        if not text or len(text.strip()) == 0:
            logger.warning(f"No text extracted from DOCX: {file_path}")
            return None, "No readable text found in the DOCX file."
            
        return text, None
        
    except Exception as e:
        logger.exception(f"Unexpected error extracting text from DOCX: {e}")
        return None, f"Error processing DOCX file: {str(e)}"

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
    Get or create user settings for the current session or user
    
    Returns:
        UserSettings: The user settings for the current session/user
    """
    # If user is logged in via Google OAuth, use email as session_id
    if 'user' in session and 'email' in session['user']:
        session_id = session['user']['email']
        logger.debug(f"Using Google email as session ID: {session_id}")
    else:
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
        logger.info(f"Created new user settings for session/user: {session_id}")
    
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
    
    # Get query parameters for messages and errors
    message = request.args.get('message')
    error = request.args.get('error')
    
    # Log any errors for debugging
    if error:
        logger.warning(f"Home page loaded with error: {error}")
    
    return render_template(
        "index.html", 
        title="EduVoice",
        theme_mode=user_settings.theme_mode,
        message=message,
        error=error,
        request=request,
        usage_stats=user_settings
    )

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    """
    Route to handle file uploads (requires login)
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
                error_message = None
                
                if file_type == 'pdf':
                    extracted_text, error_message = extract_text_from_pdf(file_path)
                elif file_type == 'docx':
                    extracted_text, error_message = extract_text_from_docx(file_path)
                else:
                    logger.error(f"Unsupported file type for text extraction: {file_type}")
                    return render_template(
                        "upload.html",
                        title="File Upload",
                        error=f"Cannot extract text from {file_type.upper()} files",
                        usage_stats=user_settings
                    )
                
                # Check if text extraction was successful
                if error_message or not extracted_text:
                    logger.error(f"Text extraction failed for {filename}: {error_message}")
                    return render_template(
                        "upload.html",
                        title="File Upload",
                        error=error_message or "Failed to extract text from the file.",
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
@login_required
def summarize_text():
    """
    Route to handle text summarization requests (requires login)
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
    # Check if user is logged in (from session)
    is_logged_in = 'user' in session
    
    # Get user info if logged in
    user_info = session.get('user', {}) if is_logged_in else {}
    
    # Get user settings and make them available in all templates
    try:
        user_settings = get_user_settings()
        
        # Create accessibility class string based on user settings
        accessibility_classes = []
        
        if user_settings.accessibility_mode:
            accessibility_classes.append('accessibility-enabled')
            
            # Add font size class
            if user_settings.font_size:
                accessibility_classes.append(f'font-size-{user_settings.font_size}')
                
            # Add high contrast class if enabled
            if user_settings.high_contrast:
                accessibility_classes.append('high-contrast')
                
            # Add dyslexia-friendly font class if enabled
            if user_settings.dyslexia_friendly:
                accessibility_classes.append('dyslexia-friendly')
                
            # Add line spacing class
            if user_settings.line_spacing:
                if user_settings.line_spacing == 1.5:
                    accessibility_classes.append('line-spacing-normal')
                elif user_settings.line_spacing == 2.0:
                    accessibility_classes.append('line-spacing-increased')
                else:
                    accessibility_classes.append('line-spacing-double')
                    
            # Add reduce animations class if enabled
            if user_settings.reduce_animations:
                accessibility_classes.append('reduce-animations')
        
        # Join all classes with a space
        accessibility_class_string = ' '.join(accessibility_classes)
        
    except Exception as e:
        logger.warning(f"Error getting user settings for template: {e}")
        user_settings = None
        accessibility_class_string = ''
    
    # Check if user is logged in via Google OAuth
    google_user = session.get('user', None)
    
    return {
        'upload_url': url_for('upload_file'),
        'user_settings': user_settings,
        'theme_mode': user_settings.theme_mode if user_settings else 'dark',
        'accessibility_classes': accessibility_class_string,
        'google_user': google_user,
        'is_logged_in': google_user is not None
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
@login_required
def dashboard():
    """
    Dashboard page displaying document upload history (requires login)
    """
    logger.debug("Accessing dashboard route")
    
    # Get any message from the URL parameters (e.g., welcome message)
    message = request.args.get('message')
    
    # Clear any potential redirection flags to prevent loops
    if 'from_callback' in session:
        session.pop('from_callback', None)
        logger.info("Cleared 'from_callback' flag in dashboard route")
    
    # Log user information for debugging
    if 'user' in session:
        user_email = session['user'].get('email', 'unknown')
        logger.info(f"User {user_email} is viewing the dashboard")
    
    try:
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
                usage_stats=user_settings,
                message=message,
                user=session.get('user', {})
            )
        except Exception as db_error:
            logger.error(f"Error querying documents: {db_error}")
            return render_template(
                "dashboard.html",
                title="Upload History Dashboard",
                error="Could not load documents. Please try again later.",
                documents=[],
                usage_stats=user_settings,
                message=message,
                user=session.get('user', {})
            )
    except Exception as e:
        logger.error(f"Error accessing dashboard: {e}")
        return render_template(
            "index.html",
            title="Error",
            error=f"Could not load dashboard: {str(e)}",
            request=request,
            usage_stats=None,
            user=session.get('user', {})
        )

# Settings route to customize user preferences
@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    """
    Settings page for user preferences (requires login)
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
            
            # Get accessibility settings
            accessibility_mode = 'accessibility_mode' in request.form
            font_size = request.form.get('font_size', 'medium')
            high_contrast = 'high_contrast' in request.form
            dyslexia_friendly = 'dyslexia_friendly' in request.form
            
            # Convert line spacing to float
            try:
                line_spacing = float(request.form.get('line_spacing', '1.5'))
            except ValueError:
                line_spacing = 1.5
                
            reduce_animations = 'reduce_animations' in request.form
            
            # Update the user settings
            user_settings.language = language
            user_settings.voice_speed = voice_speed
            user_settings.theme_mode = theme_mode
            
            # Update accessibility settings
            user_settings.accessibility_mode = accessibility_mode
            user_settings.font_size = font_size
            user_settings.high_contrast = high_contrast
            user_settings.dyslexia_friendly = dyslexia_friendly
            user_settings.line_spacing = line_spacing
            user_settings.reduce_animations = reduce_animations
            
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
# Add a route for reporting errors
@app.route('/report-error', methods=['POST'])
def report_error():
    """
    Handle error reports from users
    """
    try:
        error_code = request.form.get('error_code', 'Unknown')
        error_message = request.form.get('error_message', 'Unknown')
        user_description = request.form.get('user_description', '')
        user_email = request.form.get('user_email', '')
        
        # Log the error report
        logger.warning(
            f"User error report received:\n"
            f"- Error code: {error_code}\n"
            f"- Error message: {error_message}\n"
            f"- User description: {user_description}\n"
            f"- User email: {user_email}"
        )
        
        # Get user settings
        user_settings = get_user_settings()
        
        # Flash a success message
        flash('Thank you for reporting the issue. Our team will look into it.', 'success')
        
        # Redirect to the homepage
        return redirect(url_for('home_page'))
        
    except Exception as e:
        logger.error(f"Error processing error report: {e}")
        return redirect(url_for('home_page'))

# Common function to handle errors
def handle_error(e, code, message, description=None, show_details=False):
    """
    Common function to handle various errors
    
    Args:
        e: The exception
        code: HTTP status code
        message: User-friendly message
        description: Additional descriptive text
        show_details: Whether to show detailed error info
    
    Returns:
        Rendered error template
    """
    error_id = uuid.uuid4().hex[:8]
    
    # Log the error with a unique ID for reference
    logger.error(f"Error ID: {error_id} - Code: {code} - {message} - Details: {str(e)}")
    
    # Get the current user settings if possible
    try:
        user_settings = get_user_settings()
        theme_mode = user_settings.theme_mode
    except Exception as settings_error:
        logger.warning(f"Error getting user settings for template: {settings_error}")
        user_settings = None
        theme_mode = "dark"
    
    # Render the error template
    return render_template(
        "error.html",
        title=f"Error {code}",
        code=code,
        message=message,
        description=description,
        details=str(e) if e else None,
        show_details=show_details,
        theme_mode=theme_mode,
        request=request,
        usage_stats=user_settings
    ), code

@app.errorhandler(400)
def bad_request_error(e):
    """Handle 400 Bad Request errors"""
    return handle_error(
        e, 
        400, 
        "Bad Request", 
        "The server cannot process your request due to invalid syntax or parameters."
    )

@app.errorhandler(401)
def unauthorized_error(e):
    """Handle 401 Unauthorized errors"""
    return handle_error(
        e, 
        401, 
        "Unauthorized", 
        "You don't have permission to access this resource."
    )

@app.errorhandler(403)
def forbidden_error(e):
    """Handle 403 Forbidden errors"""
    return handle_error(
        e, 
        403, 
        "Forbidden", 
        "You don't have permission to access this resource."
    )

@app.errorhandler(404)
def not_found_exception_handler(e):
    """Handle 404 Not Found errors"""
    return handle_error(
        e, 
        404, 
        "Page Not Found", 
        f"The requested URL ({request.path}) was not found on this server."
    )

@app.errorhandler(413)
def request_entity_too_large_error(e):
    """Handle 413 Request Entity Too Large errors"""
    return handle_error(
        e, 
        413, 
        "File Too Large", 
        f"The file you're trying to upload is too large. Maximum allowed size is {app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024):.1f} MB."
    )

@app.errorhandler(500)
def server_error_handler(e):
    """Handle 500 Internal Server Error"""
    # In production, show detailed error information for debugging
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    return handle_error(
        e, 
        500, 
        "Server Error", 
        "An unexpected error occurred on our servers. Our team has been notified and is working to fix the issue.",
        show_details=debug_mode
    )

@app.errorhandler(Exception)
def unhandled_exception(e):
    """Handle any unhandled exceptions"""
    logger.exception("Unhandled exception occurred")
    return handle_error(
        e, 
        500, 
        "Unexpected Error", 
        "An unexpected error occurred. Our technical team has been notified.",
        show_details=False
    )

# Guest access route
@app.route('/guest')
def guest_access():
    """
    Allow limited access to the application as a guest user
    """
    logger.debug("Guest access requested")
    
    # Create a guest session with limited access
    session['user'] = {
        'id': f"guest_{uuid.uuid4().hex[:8]}",
        'name': 'Guest User',
        'email': 'guest@example.com',
        'picture': None,
        'logged_in_at': datetime.datetime.now().isoformat(),
        'auth_provider': 'guest',
        'is_guest': True
    }
    
    # Get or create guest user settings
    user_settings = get_user_settings()
    
    # Flash a message about guest limitations
    flash('You are using EduVoice as a guest. Some features may be limited. Log in with Google for full access.', 'info')
    
    # Redirect to dashboard with limited functionality
    return redirect(url_for('dashboard', message="Welcome, Guest User!"))

# Run the application
if __name__ == "__main__":
    # Run Flask app
    logger.info("Starting Flask application")
    app.run(host="0.0.0.0", port=5000, debug=True)
