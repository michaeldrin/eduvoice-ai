from main import app as fastapi_app
import uvicorn
from fastapi.middleware.wsgi import WSGIMiddleware
from flask import Flask

# Create a Flask app for WSGI compatibility with Gunicorn
flask_app = Flask(__name__)

# Wrap the FastAPI app with WSGIMiddleware
@flask_app.route('/', defaults={'path': ''})
@flask_app.route('/<path:path>')
def catch_all(path):
    return WSGIMiddleware(fastapi_app)({"PATH_INFO": "/" + path} if path else {"PATH_INFO": "/"}, lambda s, r: None)

# This is what Gunicorn will import
app = flask_app

if __name__ == "__main__":
    uvicorn.run(fastapi_app, host="0.0.0.0", port=5000)