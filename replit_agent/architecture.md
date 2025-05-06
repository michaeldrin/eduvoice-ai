# Architecture

## Overview

EduVoice is a document processing application built with Flask that allows users to upload, analyze, and interact with document content. The system uses AI capabilities to extract information, generate summaries, and provide a chat interface for document-based questions. It supports accessibility features and includes OAuth-based authentication.

## System Architecture

The application follows a monolithic architecture with a clear separation between the presentation layer (templates), application logic (Flask routes and controllers), and data storage (PostgreSQL database via SQLAlchemy).

```
┌────────────────────────────────────────────┐
│                  Client                     │
└───────────────────────┬────────────────────┘
                        │
┌───────────────────────▼────────────────────┐
│                 Flask Server                │
│  ┌──────────────┐  ┌─────────────────────┐ │
│  │   Routes &   │  │    Templates &      │ │
│  │ Controllers  │◄─┤     Static Files    │ │
│  └──────┬───────┘  └─────────────────────┘ │
│         │                                   │
│  ┌──────▼───────┐  ┌─────────────────────┐ │
│  │    Models    │◄─┤    SQLAlchemy ORM   │ │
│  └──────┬───────┘  └─────────────────────┘ │
└─────────┼───────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────┐
│             PostgreSQL Database             │
└────────────────────────────────────────────┘
```

## Key Components

### Frontend

1. **Templates**: The application uses Jinja2 templating engine for rendering HTML pages. Templates are located in the `templates/` directory and include:
   - Main pages (dashboard, document preview, upload, settings)
   - Components (navbar, floating action button)
   - Error pages

2. **Static Assets**:
   - CSS: Custom styles and accessibility features
   - JavaScript: Client-side functionality for accessibility, audio visualization, and UI components
   - Images and SVG files for branding and illustrations

### Backend

1. **Main Application (main.py)**:
   - Entry point for the Flask application
   - Contains route definitions and controller logic
   - Handles file processing, API integration, and business logic

2. **Models (models.py)**:
   - Defines database schemas using SQLAlchemy ORM
   - Key models include:
     - Document: Stores uploaded document information and content
     - ChatMessage: Stores conversation history for document chat
     - DocumentQuestion: Stores extracted or generated questions
     - UserSettings: Stores user preferences and settings
     - UserNote: Stores user annotations for documents

3. **Authentication (oauth.py)**:
   - Handles OAuth integration with Google
   - Manages user sessions and authentication state
   - Provides login protection for routes

4. **Entry Points**:
   - asgi.py: ASGI entry point for uvicorn
   - wsgi.py: WSGI entry point for gunicorn

### Data Storage

The application uses PostgreSQL for data persistence, with SQLAlchemy as the ORM layer. Database connection is configured in the main application with connection pooling and health checks.

## Data Flow

1. **Document Processing Flow**:
   ```
   Upload → Text Extraction → Analysis → Storage → Presentation
   ```
   
   - User uploads a document (PDF or DOCX)
   - System extracts text using PyMuPDF (for PDFs) or python-docx (for DOCX)
   - Text is analyzed by OpenAI API for summarization
   - Results are stored in the database and presented to the user

2. **Document Chat Flow**:
   ```
   User Question → Context Preparation → AI Processing → Response Storage → Display
   ```
   
   - User asks a question about the document
   - System prepares document content as context
   - Question and context are sent to OpenAI API
   - Response is stored as a chat message and displayed to the user

3. **Authentication Flow**:
   ```
   Login Request → OAuth Redirect → Callback Processing → Session Creation
   ```
   
   - User initiates login
   - System redirects to Google OAuth
   - Google returns control with authentication token
   - System creates a user session

## External Dependencies

1. **OpenAI API**:
   - Used for text analysis, summarization, and chat functionality
   - Integration handled in main.py

2. **Google OAuth**:
   - Used for user authentication
   - Configured in oauth.py

3. **Document Processing Libraries**:
   - PyMuPDF (fitz): PDF processing
   - python-docx: DOCX processing

4. **Text-to-Speech**:
   - gTTS (Google Text-to-Speech): Used for generating audio from summaries

## Deployment Strategy

The application is configured for deployment on Replit with the following components:

1. **Runtime Environment**:
   - Python 3.11
   - PostgreSQL 16

2. **Application Server**:
   - Gunicorn as the WSGI server
   - Configured to run with auto-scaling

3. **Dependencies**:
   - Managed through pyproject.toml and uv.lock
   - Additional system packages configured in .replit Nix configuration

4. **Configuration**:
   - Environment variables for API keys and secrets
   - Domain detection for OAuth callbacks based on environment

5. **Monitoring**:
   - Custom logging to file (logs/app.log) and console
   - Configurable log levels for different environments

## Security Considerations

1. **Authentication**: OAuth-based authentication with Google provides secure user identification.

2. **Input Validation**: File uploads are validated for type and size limits.

3. **API Key Protection**: OpenAI and OAuth keys are stored as environment variables.

4. **Error Handling**: Comprehensive error handling to prevent information disclosure.

5. **Database Security**: Connection pooling with timeouts and health checks.

## Accessibility

The application includes comprehensive accessibility features:

1. **Visual Adaptations**: Configurable font sizes, contrast settings, and line spacing

2. **Reading Aids**: Reading ruler and dyslexia-friendly fonts

3. **Animation Controls**: Option to reduce animations for users with sensitivities

4. **Theme Support**: Light and dark mode options

These features are implemented via CSS classes and JavaScript controls in the settings interface.