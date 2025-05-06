from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class ChatMessage(db.Model):
    """Model for document chat messages"""
    __tablename__ = 'chat_message'
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    session_id = db.Column(db.String(255), nullable=False)  # Session ID to track conversations
    message_type = db.Column(db.String(10), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    has_attachment = db.Column(db.Boolean, default=False)  # Flag to indicate if message has an attachment
    attachment_filename = db.Column(db.String(255), nullable=True)  # Stored filename of attachment
    attachment_original_filename = db.Column(db.String(255), nullable=True)  # Original filename of attachment
    attachment_type = db.Column(db.String(50), nullable=True)  # Type of attachment (pdf, docx, txt, image)
    attachment_text = db.Column(db.Text, nullable=True)  # Extracted text from attachment
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ChatMessage {self.id}>'
        
    def to_dict(self):
        """Convert chat message to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'document_id': self.document_id,
            'session_id': self.session_id,
            'message_type': self.message_type,
            'content': self.content,
            'has_attachment': self.has_attachment,
            'attachment_filename': self.attachment_filename,
            'attachment_original_filename': self.attachment_original_filename,
            'attachment_type': self.attachment_type,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

class Document(db.Model):
    """Model for uploaded documents and their summaries"""
    __tablename__ = 'document'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(255), nullable=True)  # For future use with user authentication
    session_id = db.Column(db.String(255), nullable=False)  # Session ID to track ownership
    filename = db.Column(db.String(255), nullable=False)
    filetype = db.Column(db.String(50), nullable=False)
    summary = db.Column(db.Text, nullable=True)
    text_content = db.Column(db.Text, nullable=True)  # Store the extracted text content
    interaction_tips = db.Column(db.Text, nullable=True)  # Store personalized interaction tips
    learning_suggestions = db.Column(db.Text, nullable=True)  # Store next learning suggestions as JSON
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    chat_messages = db.relationship('ChatMessage', backref='document', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Document {self.filename}>'
    
    def to_dict(self):
        """Convert document to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'filename': self.filename,
            'filetype': self.filetype,
            'summary': self.summary,
            'has_content': self.text_content is not None,
            'has_tips': self.interaction_tips is not None,
            'has_suggestions': self.learning_suggestions is not None,
            'upload_time': self.upload_time.strftime('%Y-%m-%d %H:%M:%S')
        }