import logging
import os
import datetime
import uuid
import traceback  # For enhanced error reporting
import requests  # For API calls and error handling
import fitz  # PyMuPDF
import docx
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash, jsonify, session
from openai import OpenAI
# Import OpenAI exceptions - using try/except since error types might vary between versions
from gtts import gTTS
from models import db, Document, UserSettings, ChatMessage
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
app.debug = True  # Enable debug mode to get detailed error messages

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

# Helper function to validate OpenAI API key and handle errors
def translate_text(text, target_language, source_language='en'):
    """
    Translate text to target language using OpenAI
    
    Args:
        text (str): Text to translate
        target_language (str): Target language code (e.g., 'es', 'fr', 'de')
        source_language (str, optional): Source language code. Defaults to 'en'.
        
    Returns:
        tuple: (translated_text, error_message)
    """
    if not text:
        return None, "No text provided for translation"
        
    # Get language names
    language_names = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean'
    }
    
    target_language_name = language_names.get(target_language, target_language)
    source_language_name = language_names.get(source_language, source_language)
    
    # Skip translation if source and target are the same
    if target_language == source_language:
        return text, None
    
    # Get OpenAI client
    client, error = validate_openai_api()
    if error:
        return None, f"API Error: {error}"
    
    try:
        logger.info(f"Translating text from {source_language_name} to {target_language_name}")
        
        # Build system message
        system_message = f"You are a professional translator. Translate the following text from {source_language_name} to {target_language_name}. Preserve formatting, maintain the same tone, and ensure accuracy."
        
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": text}
            ],
            max_tokens=1500,
            temperature=0.1,  # Lower temperature for more deterministic output
            timeout=30  # Add timeout to prevent long-running requests
        )
        
        translated_text = response.choices[0].message.content.strip()
        logger.info(f"Translation completed successfully")
        
        return translated_text, None
        
    except Exception as e:
        error_message = handle_openai_error(e)
        logger.error(f"Translation error: {error_message}")
        return None, f"Translation error: {error_message}"

def validate_openai_api(api_key=None):
    """
    Validate OpenAI API key and handle common errors
    
    Args:
        api_key (str, optional): API key to validate. If None, get from environment.
        
    Returns:
        tuple: (OpenAI client or None, error message or None)
    """
    try:
        # Get API key from environment if not provided
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            
        # Check if API key exists
        if not api_key:
            logger.error("OpenAI API key not found in environment variables")
            return None, "OpenAI API key not configured. Please contact the administrator."
            
        # Validate API key format (should start with 'sk-')
        if not isinstance(api_key, str) or not api_key.startswith("sk-"):
            logger.error("Invalid OpenAI API key format")
            return None, "Invalid OpenAI API key format. Please check your API key and try again."
            
        # Create OpenAI client
        client = OpenAI(api_key=api_key)
        return client, None
        
    except Exception as e:
        logger.exception(f"Error validating OpenAI API key: {e}")
        return None, f"Error initializing OpenAI client: {str(e)}"

# Helper function to handle OpenAI API errors
def handle_openai_error(error):
    """
    Handle OpenAI API errors and return user-friendly error messages
    
    Args:
        error (Exception): The error to handle
        
    Returns:
        str: User-friendly error message
    """
    error_str = str(error)
    error_type = type(error).__name__
    
    # Log the detailed error
    logger.error(f"OpenAI API error ({error_type}): {error_str}")
    
    # Check for common error patterns
    if "authentication" in error_str.lower() or "auth" in error_type.lower():
        return "Invalid API key. Please verify your OpenAI API key and try again."
    elif "rate" in error_str.lower() and "limit" in error_str.lower():
        return "You've reached the OpenAI API rate limit. Please try again later or upgrade your API plan."
    elif "insufficient_quota" in error_str.lower():
        return "Your OpenAI API account has insufficient quota. Please check your billing details or upgrade your plan."
    elif "invalid" in error_str.lower() and "request" in error_str.lower():
        return "Invalid request to OpenAI API. The document might be too large or contain unsupported content."
    elif "connection" in error_str.lower() or "network" in error_str.lower():
        return "Could not connect to OpenAI API. Please check your internet connection and try again."
    else:
        # Generic fallback message
        logger.exception(f"Unexpected OpenAI error: {error}")
        return f"OpenAI API error: {error_str}"

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

# Helper function to generate a chat response for document Q&A
def generate_chat_response(document_id, user_message):
    """
    Generate a response to a user question about a document using OpenAI API
    
    Args:
        document_id (int): The ID of the document to chat about
        user_message (str): The user's question or message
        
    Returns:
        tuple: (response_text, error_message)
    """
    # Initialize these variables to track if we need to roll back transactions on error
    user_chat_message = None
    assistant_chat_message = None
    transaction_started = False
    
    try:
        # Input validation
        if not document_id:
            logger.error("Missing document_id parameter")
            return None, "Missing document ID"
            
        if not user_message or not user_message.strip():
            logger.error("Empty user message")
            return None, "Please enter a question or message to continue the conversation."
            
        # Get the document from database with error handling
        try:
            document = Document.query.get(document_id)
            if not document:
                logger.error(f"Document not found: {document_id}")
                return None, "Document not found. It may have been deleted."
                
            if not document.text_content:
                logger.error(f"Document has no content: {document_id}")
                return None, "This document has no text content to chat about. Please try with a different document."
        except Exception as db_error:
            logger.exception(f"Database error retrieving document: {db_error}")
            return None, "Could not access document information. Please try again later."
        
        # Get OpenAI client
        client, error = validate_openai_api()
        if error:
            logger.error(f"OpenAI API key validation failed: {error}")
            
            # Include API key instructions in debug mode
            if app.debug:
                error_details = f"{error} Please check that your OPENAI_API_KEY is set correctly in the Replit Secrets."
                return None, error_details
            return None, error
            
        # Get document and user language preference
        try:
            # First try to use the document language
            language = document.language
            
            # If not set, fall back to user settings
            if not language:
                settings = get_user_settings()
                language = settings.language
        except Exception as settings_error:
            logger.error(f"Error retrieving language settings: {settings_error}")
            language = 'en'  # Default to English on error
        
        # Map language codes to language names for the prompt
        language_names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi'
        }
        
        # Default to English if language is not in our mapping
        language_name = language_names.get(language, 'English')
        
        # Get all previous chat messages for context with error handling
        try:
            chat_history = ChatMessage.query.filter_by(document_id=document_id).order_by(ChatMessage.created_at).all()
        except Exception as history_error:
            logger.exception(f"Error retrieving chat history: {history_error}")
            chat_history = []  # Use empty history as fallback
        
        # Get document content and truncate if needed
        max_content_length = 10000  # Limit to avoid token limits
        document_content = document.text_content[:max_content_length]
        if len(document.text_content) > max_content_length:
            document_content += " [Content truncated due to length...]"
        
        # Create the system prompt
        system_prompt = f"""You are an intelligent assistant that helps users understand the content of documents.
        Respond in {language_name}.
        The document content is provided below. Use it to answer the user's questions.
        Be concise, accurate, and helpful. If you don't know the answer based on the document, say so.
        Do not make up information that is not in the document.
        
        DOCUMENT CONTENT:
        {document_content}
        """
        
        # Build conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history for context (up to the last 10 messages)
        # Only include most recent messages to stay within token limits
        if chat_history:
            for message in chat_history[-10:]:
                messages.append({
                    "role": "user" if message.message_type == "user" else "assistant",
                    "content": message.content
                })
        
        # Add the current user message
        messages.append({"role": "user", "content": user_message})
        
        # Start a transaction for database operations
        transaction_started = True
        
        # First, save the user message immediately, so we have a record even if the API call fails
        user_chat_message = ChatMessage(
            document_id=document_id,
            user_id=session.get('user', {}).get('email', 'guest'),
            message_type='user',
            content=user_message
        )
        
        db.session.add(user_chat_message)
        db.session.commit()
        logger.debug(f"User message saved to database for document {document_id}")
        
        try:
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=500,
                timeout=30  # Add timeout to prevent long-running requests
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
            # Save only the assistant response to the database (user message already saved)
            assistant_chat_message = ChatMessage(
                document_id=document_id,
                user_id=session.get('user', {}).get('email', 'guest'),
                message_type='assistant',
                content=response_text
            )
            
            db.session.add(assistant_chat_message)
            db.session.commit()
            transaction_started = False
            
            logger.info(f"Chat response generated for document {document_id}")
            return response_text, None
            
        except Exception as api_error:
            # Handle API errors
            error_message = handle_openai_error(api_error)
            logger.error(f"OpenAI API error in chat response: {error_message}")
            
            # Save an error response to the database (user message already saved)
            error_response = ChatMessage(
                document_id=document_id,
                user_id=session.get('user', {}).get('email', 'guest'),
                message_type='assistant',
                content=f"I'm sorry, I couldn't generate a response at this time due to a technical issue. Please try again in a moment."
            )
            
            db.session.add(error_response)
            db.session.commit()
            transaction_started = False
            
            if app.debug:
                # Return detailed error in debug mode
                return None, f"API Error: {error_message}"
            else:
                return None, "The AI service is currently unavailable. Please try again later."
        
    except Exception as e:
        # General error handling
        logger.exception(f"Unexpected error in chat response generation: {e}")
        
        # If we were in the middle of a transaction, roll it back
        if transaction_started:
            try:
                db.session.rollback()
            except Exception as rollback_error:
                logger.error(f"Error during transaction rollback: {rollback_error}")
        
        if app.debug:
            # Return detailed error in debug mode
            return None, f"Error: {str(e)}"
        else:
            return None, "An unexpected error occurred. Please try again later."

# Helper function to generate an initial welcome message for document chat
def generate_initial_chat_message(document):
    """
    Generate an initial welcome message for a document chat
    
    Args:
        document (Document): The document to generate a welcome message for
        
    Returns:
        str: The welcome message
    """
    try:
        if not document or not document.text_content:
            return None
            
        # Get OpenAI client
        client, error = validate_openai_api()
        if error:
            logger.error(f"OpenAI API key validation failed: {error}")
            return None
            
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
        
        # Create the system prompt
        system_prompt = f"""You are an intelligent assistant that helps users understand documents.
        Respond in {language_name}.
        You need to create a brief, friendly welcome message for a document chat.
        Include 2-3 specific suggested questions the user could ask about this document.
        
        DOCUMENT SUMMARY:
        Title: {document.filename}
        Type: {document.filetype}
        Content preview: {document.text_content[:2000] if document.text_content else "No text content available."}
        """
        
        try:
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Please create a welcome message for this document chat."}
                ],
                max_tokens=350
            )
            
            # Extract the response text
            welcome_message = response.choices[0].message.content
            
            # Save the assistant message to the database
            assistant_chat_message = ChatMessage(
                document_id=document.id,
                user_id=document.user_id or 'system',
                message_type='assistant',
                content=welcome_message
            )
            
            db.session.add(assistant_chat_message)
            db.session.commit()
            
            logger.info(f"Initial chat message generated for document {document.id}")
            return welcome_message
            
        except Exception as api_error:
            error_message = handle_openai_error(api_error)
            logger.error(f"OpenAI API error in welcome message: {error_message}")
            
            # Create a fallback welcome message
            fallback_message = f"Welcome to the document chat for '{document.filename}'. I'm currently experiencing some technical difficulties with my AI service. Please try asking a question anyway, and I'll do my best to help when the service is restored."
            
            # Save the fallback message to the database
            assistant_chat_message = ChatMessage(
                document_id=document.id,
                user_id=document.user_id or 'system',
                message_type='assistant',
                content=fallback_message
            )
            
            db.session.add(assistant_chat_message)
            db.session.commit()
            
            return fallback_message
        
    except Exception as e:
        logger.exception(f"Error generating initial chat message: {e}")
        return None

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
        
        # Get OpenAI client
        client, error = validate_openai_api()
        if error:
            logger.error(f"OpenAI API key validation failed: {error}")
            
            # FALLBACK LOGIC: Return a generic summary message when API fails
            if app.debug:
                # In debug mode, include the error message
                logger.warning("Using fallback summary due to API error")
                fallback_summary = f"[FALLBACK SUMMARY USED - API ERROR: {error}]\n\n"
                fallback_summary += "This is a computer-generated summary of the document you uploaded. "
                fallback_summary += "The summary focuses on the key points and main ideas presented in the text. "
                fallback_summary += "For a more detailed analysis, please read the original document."
                return fallback_summary, None
            else:
                # In production, just return the error
                return None, error
            
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
        
        try:
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
                max_tokens=500,
                timeout=20  # Add timeout to prevent long-running requests
            )
            
            # Extract the summary from the response
            summary = response.choices[0].message.content
            
            # Return the summary
            if truncated_for_api:
                summary += "\n\n(Note: The original text was truncated due to length constraints before summarization.)"
                
            return summary, None
            
        except Exception as api_error:
            error_message = handle_openai_error(api_error)
            logger.error(f"OpenAI API error in summary generation: {error_message}")
            
            # FALLBACK LOGIC: Provide a generic summary for testing when API fails
            if app.debug:
                logger.warning("Using fallback summary due to API error")
                fallback_summary = f"[FALLBACK SUMMARY USED - API ERROR: {error_message}]\n\n"
                fallback_summary += "This is a fallback summary of the document you uploaded. "
                fallback_summary += "The original content contained approximately {len(text)} characters. "
                fallback_summary += "We're currently unable to generate a real summary due to API issues."
                return fallback_summary, None
            else:
                return None, error_message
        
    except Exception as e:
        logger.exception(f"Error generating summary with OpenAI: {e}")
        
        # FALLBACK LOGIC: Return a generic error message with detailed logging
        if app.debug:
            fallback_summary = f"[DEBUG MODE FALLBACK - GENERAL ERROR: {str(e)}]\n\n"
            fallback_summary += "This is a placeholder summary due to an error in the summarization process. "
            fallback_summary += "Check the application logs for more details about this error."
            return fallback_summary, None
        else:
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
    else:
        # Check if daily counters need to be reset
        if settings.check_daily_reset():
            db.session.commit()
            logger.info(f"Reset daily usage counters for user: {session_id}")
    
    return settings

# Helper function for automatically processing a document
def auto_process_document(document):
    """
    Automatically process a document by generating a summary, TTS audio, and initial chat message
    
    Args:
        document (Document): The document to process
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    processing_success = False
    try:
        # Input validation
        if not document:
            logger.error("Cannot auto-process: document is None")
            return False
            
        if not document.text_content:
            logger.error(f"Cannot auto-process document {document.id}: no text content")
            # Make sure the document shows as processed even if there's no content
            # This prevents repeated processing attempts for empty documents
            try:
                document.auto_processed = True
                db.session.commit()
            except Exception as db_error:
                logger.exception(f"Database error while flagging empty document as processed: {db_error}")
                db.session.rollback()
            return False
            
        # Start auto-processing
        logger.info(f"Starting auto-processing for document {document.id}: {document.filename}")
        
        # Generate summary if not already done
        if not document.summary:
            try:
                logger.info(f"Generating summary for document {document.id}")
                summary, error = generate_summary(document.text_content)
                if error or not summary:
                    logger.error(f"Failed to generate summary: {error}")
                    # Continue with partial processing
                else:
                    document.summary = summary
                    
                    # Save summary to a text file with error handling
                    try:
                        summaries_dir = os.path.join('static', 'summaries')
                        os.makedirs(summaries_dir, exist_ok=True)
                        
                        summary_filename = f"summary_{uuid.uuid4().hex[:8]}.txt"
                        summary_path = os.path.join(summaries_dir, summary_filename)
                        
                        with open(summary_path, 'w', encoding='utf-8') as summary_file:
                            if summary:  # Extra check to prevent None errors
                                summary_file.write(summary)
                            
                        document.text_filename = summary_filename
                        db.session.commit()  # Save progress incrementally
                        logger.info(f"Summary file created: {summary_filename}")
                    except Exception as file_error:
                        logger.exception(f"Error saving summary file: {file_error}")
                        # Continue with partial processing
            except Exception as summary_error:
                logger.exception(f"Error during summary generation: {summary_error}")
                # Continue with partial processing
            
        # Generate audio if not already done and if we have a summary
        if document.summary and not document.audio_filename:
            try:
                logger.info(f"Generating audio for document {document.id}")
                audio_filename, error = text_to_speech(document.summary)
                if error or not audio_filename:
                    logger.error(f"Failed to generate audio: {error}")
                    # Continue processing even if audio fails
                else:
                    document.audio_filename = audio_filename
                    db.session.commit()  # Save progress incrementally
                    logger.info(f"Audio file created: {audio_filename}")
            except Exception as audio_error:
                logger.exception(f"Error during audio generation: {audio_error}")
                # Continue with partial processing
        
        # Generate initial chat message
        try:
            logger.info(f"Generating initial chat message for document {document.id}")
            welcome_message = generate_initial_chat_message(document)
            if not welcome_message:
                logger.warning(f"Failed to generate welcome message for document {document.id}")
                # Continue processing even if welcome message fails
        except Exception as chat_error:
            logger.exception(f"Error generating initial chat message: {chat_error}")
            # Continue with partial processing
            
        # Mark document as auto-processed regardless of partial failures
        try:
            document.auto_processed = True
            db.session.commit()
            logger.info(f"Auto-processing complete for document {document.id}")
            processing_success = True
        except Exception as db_error:
            logger.exception(f"Database error while finalizing auto-processing: {db_error}")
            db.session.rollback()
            
        return processing_success
        
    except Exception as e:
        logger.exception(f"Unhandled exception during auto-processing of document: {e}")
        try:
            # Still try to mark as processed to prevent repeated processing attempts
            if document:
                document.auto_processed = True
                db.session.commit()
        except Exception:
            db.session.rollback()
        return processing_success

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
            base_filename = f"speech_{uuid.uuid4().hex[:8]}"
        else:
            # Strip any path and extension
            base_filename = os.path.splitext(os.path.basename(filename))[0]
            
        # Add document name if available (from a filename like 'document_123.pdf')
        if '_' in base_filename and not base_filename.startswith('speech_'):
            # Extract the document name for better identification
            doc_name = base_filename.split('.')[0]
            filename = f"{doc_name}_{uuid.uuid4().hex[:8]}.mp3"
        else:
            filename = f"{base_filename}.mp3"
            
        # Full path to the audio file
        audio_path = os.path.join(audio_dir, filename)
        
        # Limit text length for TTS if needed (gTTS has limits)
        max_tts_length = 5000  # Characters
        truncated = False
        if len(text) > max_tts_length:
            text = text[:max_tts_length] + "... Text has been truncated for audio conversion."
            truncated = True
            logger.info(f"Text truncated for TTS (length: {len(text)})")
        
        # Get user settings for language and voice speed
        settings = get_user_settings()
        
        # Default to English if language not supported by gTTS
        supported_languages = ['en', 'de', 'fr', 'es', 'it', 'ru', 'zh-CN', 'ja']
        language = settings.language if settings.language in supported_languages else 'en'
        
        # Set the speaking rate
        slow_speech = settings.voice_speed == 'slow'
        
        try:
            # Create gTTS object with user settings
            tts = gTTS(text=text, lang=language, slow=slow_speech)
            
            # Save the audio file
            tts.save(audio_path)
            
            # Check if the file was created successfully and has content
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                raise Exception("Failed to create audio file or file is empty")
                
            logger.info(f"Audio file created: {filename} in language: {language}" + 
                        (" (truncated)" if truncated else ""))
            
            # Return the filename (without the full path)
            return filename, None
            
        except requests.exceptions.RequestException as network_error:
            # Network-related errors (like connectivity issues)
            logger.error(f"Network error during TTS API call: {network_error}")
            return None, "Could not connect to the text-to-speech service. Please check your internet connection."
            
        except ValueError as value_error:
            # Invalid values (like unsupported language)
            logger.error(f"Value error in TTS: {value_error}")
            return None, f"Invalid parameter for text-to-speech: {str(value_error)}"
        
    except IOError as io_error:
        # File-related errors
        logger.error(f"File I/O error during TTS processing: {io_error}")
        return None, "Could not create the audio file due to a file system error."
        
    except Exception as e:
        # General catch-all with detailed logging
        logger.exception(f"Unexpected error in text-to-speech conversion: {e}")
        
        if app.debug:
            # In debug mode, return detailed error info
            return None, f"Error converting text to speech: {str(e)}"
        else:
            # In production, return a user-friendly message
            return None, "An error occurred while creating the audio file. Please try again later."

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
        usage_stats=user_settings,
        user=session.get('user', {})  # Ensure user is always defined
    )

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    """
    Route to handle file uploads (requires login)
    """
    try:
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
                try:
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
                            # More user-friendly error message
                            fallback_error = f"Could not read this {file_type.upper()} file. It may be password-protected, contain only images, or be in an unsupported format."
                            return render_template(
                                "upload.html",
                                title="File Upload",
                                error=error_message or fallback_error,
                                usage_stats=user_settings
                            )
                        
                        # Get the total length of extracted text
                        total_length = len(extracted_text)
                        
                        # Check if we need to truncate the text for the preview
                        truncated = False
                        preview_text = extracted_text
                        if total_length > app.config['PREVIEW_TEXT_MAX_LENGTH']:
                            preview_text = extracted_text[:app.config['PREVIEW_TEXT_MAX_LENGTH']] + "... (truncated for preview)"
                            truncated = True
                            logger.info(f"Text truncated for preview: {total_length} characters")
                        
                        # Render the preview template with the extracted text
                        return render_template(
                            "preview.html",
                            title="Text Preview",
                            filename=filename,
                            filetype=file_type.upper(),
                            extracted_text=preview_text,
                            total_length=total_length,
                            truncated=truncated,
                            usage_stats=user_settings
                        )
                        
                    except Exception as e:
                        logger.exception(f"Error processing file {filename} content: {str(e)}")
                        # Rollback database transaction if any
                        db.session.rollback()
                        error_detail = str(e)
                        # Clean up detailed error messages for user display
                        user_error = f"Error extracting text: Could not process {file_type.upper()} file content"
                        # Log the detailed error for debugging
                        logger.error(f"Detailed error: {error_detail}")
                        return render_template(
                            "upload.html",
                            title="File Upload",
                            error=user_error,
                            usage_stats=user_settings
                        )
                        
                except Exception as save_error:
                    logger.exception(f"Error saving file {file.filename}: {str(save_error)}")
                    db.session.rollback()
                    return render_template(
                        "upload.html",
                        title="File Upload",
                        error=f"Could not save file. {str(save_error)}",
                        usage_stats=user_settings
                    )
            else:
                logger.error(f"Invalid file type: {file.filename if file and file.filename else 'unknown'}")
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
        
    except Exception as e:
        # This catch-all exception handler prevents 500 errors
        logger.exception(f"Unhandled exception in upload route: {str(e)}")
        db.session.rollback()  # Ensure any failed database transactions are rolled back
        
        # Return direct error details in debug mode
        if app.debug:
            # Return a detailed error response for debugging
            error_detail = f"Error in upload route: {str(e)}"
            return f"<h1>Server Error (500)</h1><pre>{error_detail}</pre>", 500
        else:
            # In production, show the user-friendly error page
            return render_template(
                "error.html",
                title="Server Error",
                code=500,
                message="Something went wrong with your request.",
                description="Our technical team has been notified. Please try again later.",
                show_details=False
            ), 500

# Route for text summarization using OpenAI
@app.route('/summarize', methods=['GET', 'POST'])
@login_required
def summarize_text():
    """
    Route to handle text summarization requests (requires login)
    """
    logger.debug("Processing summarization request")
    
    # Redirect GET requests to the upload page with a helpful message
    if request.method == 'GET':
        flash("To summarize text, please upload a document or paste text from the upload page.", "info")
        return redirect(url_for('upload_file'))
    
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
        # Get the current user's ID (email) or session ID for guest users
        user_id = None
        if 'user' in session and 'email' in session['user']:
            user_id = session['user']['email']
            logger.info(f"Associating document with user: {user_id}")
        else:
            # Use session ID for guest users
            user_id = session.get('session_id', uuid.uuid4().hex)
            logger.info(f"Associating document with guest session: {user_id}")
            
        document = Document(
            user_id=user_id,  # Associate document with user
            filename=filename,
            filetype=filetype,
            summary=summary,
            text_content=extracted_text,  # Store the full text content for chat
            text_filename=text_filename,
            audio_filename=audio_file,
            upload_time=datetime.datetime.now(),
            auto_processed=True  # Mark as processed since we're doing it here
        )
        db.session.add(document)
        db.session.commit()
        logger.info(f"Document record saved to database: {filename}")
        
        # Store document ID in session for the chat feature in summary page
        session['last_document_id'] = document.id
        
        # Generate initial chat message for document chat
        try:
            welcome_message = generate_initial_chat_message(document)
            if welcome_message:
                logger.info(f"Generated initial chat message for document {document.id}")
            else:
                logger.warning(f"Failed to generate initial chat message for document {document.id}")
        except Exception as chat_error:
            logger.error(f"Error generating initial chat message: {chat_error}")
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
    
    # Import the UserSettings class for its testing mode flag
    from models import UserSettings
    
    return {
        'upload_url': url_for('upload_file'),
        'user_settings': user_settings,
        'theme_mode': user_settings.theme_mode if user_settings else 'dark',
        'accessibility_classes': accessibility_class_string,
        'google_user': google_user,
        'is_logged_in': google_user is not None,
        'testing_mode': UserSettings.TESTING_MODE  # Add testing mode flag to all templates
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

# Document chat page
@app.route('/document/<int:document_id>/chat')
@login_required
def document_chat(document_id):
    """
    Chat with a specific document using AI
    """
    logger.debug(f"Accessing document chat for document ID: {document_id}")
    
    try:
        # Get the document
        document = Document.query.get_or_404(document_id)
        
        # Security check: make sure the current user owns this document
        current_user_id = None
        if 'user' in session:
            current_user_id = session['user'].get('email')
        else:
            current_user_id = session.get('session_id', '')
            
        if document.user_id != current_user_id:
            logger.warning(f"Unauthorized access attempt to document {document_id} by {current_user_id}")
            flash("You don't have permission to access this document.", "error")
            return redirect(url_for('dashboard'))
            
        # Get the chat messages for this document
        chat_messages = ChatMessage.query.filter_by(document_id=document_id).order_by(ChatMessage.created_at).all()
        
        # If no chat messages exist yet, auto-generate a welcome message
        if not chat_messages and document.text_content:
            try:
                logger.info(f"Generating initial welcome message for document {document_id}")
                welcome_message = generate_initial_chat_message(document)
                if welcome_message:
                    # Reload the chat messages to include the new welcome message
                    chat_messages = ChatMessage.query.filter_by(document_id=document_id).order_by(ChatMessage.created_at).all()
            except Exception as e:
                logger.error(f"Error generating welcome message: {e}")
        
        # Get user settings
        user_settings = get_user_settings()
        
        # Render the chat template
        return render_template(
            "document_chat.html",
            title=f"Chat with {document.filename}",
            document=document,
            chat_messages=chat_messages,
            usage_stats=user_settings
        )
        
    except Exception as e:
        logger.error(f"Error accessing document chat: {e}")
        flash(f"Error accessing document chat: {str(e)}", "error")
        return redirect(url_for('dashboard'))

# API endpoint for document chat
@app.route('/api/document/<int:document_id>/chat', methods=['POST'])
@login_required
def document_chat_api(document_id):
    """
    API endpoint for document chat interactions
    """
    try:
        # Log API request
        logger.info(f"Document chat API request received for document ID: {document_id}")
        
        # Validate document_id format
        if not isinstance(document_id, int) or document_id <= 0:
            logger.warning(f"Invalid document ID format: {document_id}")
            return jsonify({'error': "Invalid document ID"}), 400
        
        try:
            # Get the document
            document = Document.query.get_or_404(document_id)
            logger.debug(f"Retrieved document {document.id}: {document.filename}")
            
            # Check if document has text content
            if not document.text_content:
                logger.warning(f"Document {document_id} has no text content for chat")
                return jsonify({'error': "This document has no content to chat about"}), 400
        except Exception as db_error:
            logger.exception(f"Database error retrieving document {document_id}: {db_error}")
            return jsonify({'error': "Could not retrieve the document. It may have been deleted."}), 404
            
        # Security check: make sure the current user owns this document
        try:
            current_user_id = None
            if 'user' in session:
                current_user_id = session['user'].get('email')
            else:
                current_user_id = session.get('session_id', '')
                
            if document.user_id != current_user_id:
                logger.warning(f"Unauthorized API access attempt to document {document_id} by {current_user_id}")
                return jsonify({'error': "You don't have permission to access this document."}), 403
        except Exception as auth_error:
            logger.exception(f"Authentication error in document chat API: {auth_error}")
            return jsonify({'error': "Authentication error. Please log in again."}), 401
        
        # Get the user message from the request
        try:
            data = request.get_json()
            if not data:
                logger.warning("No JSON data in request")
                return jsonify({'error': 'Invalid request format. Expected JSON data.'}), 400
                
            if 'message' not in data:
                logger.warning("Missing 'message' field in request data")
                return jsonify({'error': 'Missing required field: message'}), 400
                
            user_message = data['message']
            
            # Check message content
            if not user_message or not isinstance(user_message, str) or len(user_message.strip()) == 0:
                logger.warning(f"Empty or invalid message content: {user_message}")
                return jsonify({'error': 'Please provide a non-empty message'}), 400
        except Exception as json_error:
            logger.exception(f"Error parsing request JSON: {json_error}")
            return jsonify({'error': 'Invalid JSON format in request'}), 400
        
        # Generate a response with enhanced error handling
        try:
            response, error = generate_chat_response(document_id, user_message)
            
            if error:
                # Check for known error patterns to provide better errors
                if "api key" in error.lower() or "apikey" in error.lower():
                    # API key related error
                    logger.error(f"OpenAI API key error: {error}")
                    return jsonify({'error': "AI service unavailable due to API configuration issues. Please try again later or contact support."}), 503
                elif "quota" in error.lower() or "limit" in error.lower() or "rate" in error.lower():
                    # Rate limit or quota error
                    logger.error(f"OpenAI API quota/rate limit error: {error}")
                    return jsonify({'error': "AI service temporarily unavailable due to usage limits. Please try again in a few minutes."}), 429
                else:
                    # Other errors
                    logger.error(f"Error generating chat response: {error}")
                    return jsonify({'error': error}), 500
        except Exception as ai_error:
            logger.exception(f"Unexpected error in AI response generation: {ai_error}")
            return jsonify({'error': "Failed to generate AI response"}), 500
                
        # Get all chat messages for this document
        try:
            chat_messages = ChatMessage.query.filter_by(document_id=document_id).order_by(ChatMessage.created_at).all()
            messages_json = [message.to_dict() for message in chat_messages]
            logger.debug(f"Retrieved {len(messages_json)} chat messages for document {document_id}")
        except Exception as chat_error:
            logger.exception(f"Error retrieving chat messages: {chat_error}")
            # Continue with empty chat history if there's an error
            logger.info("Continuing with empty chat history due to retrieval error")
            messages_json = []
        
        # Success response
        return jsonify({
            'success': True,
            'response': response,
            'messages': messages_json
        })
        
    except Exception as e:
        logger.exception(f"Unhandled exception in document chat API: {e}")
        if app.debug:
            # In debug mode, include the detailed error
            return jsonify({
                'error': f"An error occurred: {str(e)}",
                'trace': traceback.format_exc()
            }), 500
        else:
            # In production, send a generic error
            return jsonify({'error': "An unexpected error occurred. Please try again later."}), 500

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
            # Get the current user's ID (email)
            current_user_id = None
            if 'user' in session:
                current_user_id = session['user'].get('email')
                
            # Query only documents belonging to the current user, ordered by upload time (newest first)
            if current_user_id:
                documents = Document.query.filter_by(user_id=current_user_id).order_by(Document.upload_time.desc()).all()
                logger.info(f"Found {len(documents)} documents for user {current_user_id}")
            else:
                # Fallback to session-based ID for guests
                session_id = session.get('session_id', '')
                documents = Document.query.filter_by(user_id=session_id).order_by(Document.upload_time.desc()).all()
                logger.info(f"Found {len(documents)} documents for guest session {session_id}")
            
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

@app.errorhandler(405)
def method_not_allowed_error(e):
    """Handle 405 Method Not Allowed errors"""
    # Get the list of allowed methods if available
    methods = getattr(e, 'valid_methods', None)
    methods_str = f"Allowed methods: {', '.join(methods)}" if methods else "Method not allowed for this URL"
    
    # Log the detailed error
    logger.error(f"Method Not Allowed: {request.method} {request.path} - {methods_str}")
    
    return handle_error(
        e,
        405,
        "Method Not Allowed",
        f"The request method {request.method} is not allowed for this URL. {methods_str}",
        show_details=app.debug
    )

@app.errorhandler(500)
def server_error_handler(e):
    """Handle 500 Internal Server Error"""
    # In production, show detailed error information for debugging
    debug_mode = os.environ.get('FLASK_ENV') == 'development' or app.debug
    
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
    logger.exception(f"Unhandled exception occurred: {str(e)}")
    
    # Include more details in debug mode
    if app.debug:
        error_details = f"Details: {str(e)}\n\nTraceback: {traceback.format_exc()}"
    else:
        error_details = "An unexpected error occurred. Our technical team has been notified."
    
    return handle_error(
        e, 
        500, 
        "Unexpected Error", 
        error_details,
        show_details=app.debug
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
