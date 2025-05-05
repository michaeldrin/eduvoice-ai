import logging
import os
import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app instance
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}
app.secret_key = 'your-secret-key'  # Needed for flash messages

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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
            
            # Get file details for the success page
            file_size = os.path.getsize(file_path)
            readable_size = f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024 * 1024):.1f} MB"
            file_type = filename.rsplit('.', 1)[1].upper()
            upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"File uploaded successfully: {filename}")
            
            return render_template(
                "upload_success.html",
                title="Upload Successful",
                filename=filename,
                filetype=file_type,
                filesize=readable_size,
                upload_time=upload_time
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

# Add upload link to the homepage
@app.context_processor
def inject_upload_url():
    return {'upload_url': url_for('upload_file')}

# Serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

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
