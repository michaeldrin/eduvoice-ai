# FastAPI with Jinja2 Templates

A simple Python FastAPI application with basic routing and Jinja2 templating.

## Features

- Basic FastAPI application structure
- Homepage route (GET "/") that returns HTML welcome message
- Server running with uvicorn
- Basic HTML template system using Jinja2
- Error handling for 404 and 500 errors
- Bootstrap-based responsive UI

## Project Structure

- `main.py`: Entry point for the application
- `templates/`: Directory containing Jinja2 HTML templates
- `static/`: Directory for static assets (CSS, JavaScript, etc.)

## How to Run

1. Install the required packages:
```bash
pip install fastapi uvicorn jinja2
