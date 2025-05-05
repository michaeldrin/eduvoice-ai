import logging
from flask import Flask, render_template, request, send_from_directory

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app instance
app = Flask(__name__)

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
