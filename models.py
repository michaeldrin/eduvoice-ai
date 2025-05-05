from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Document(db.Model):
    """Model for uploaded documents and their summaries"""
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filetype = db.Column(db.String(50), nullable=False)
    summary = db.Column(db.Text, nullable=True)
    text_filename = db.Column(db.String(255), nullable=True)
    audio_filename = db.Column(db.String(255), nullable=True)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Document {self.filename}>'
    
    def to_dict(self):
        """Convert document to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'filename': self.filename,
            'filetype': self.filetype,
            'summary': self.summary,
            'text_filename': self.text_filename,
            'audio_filename': self.audio_filename,
            'upload_time': self.upload_time.strftime('%Y-%m-%d %H:%M:%S')
        }

class UserSettings(db.Model):
    """Model for user settings and preferences"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), unique=True, nullable=False)
    language = db.Column(db.String(10), default='en')  # Default: English
    voice_speed = db.Column(db.String(10), default='normal')  # Slow, Normal, Fast
    theme_mode = db.Column(db.String(10), default='dark')  # Dark or Light
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<UserSettings {self.session_id}>'
    
    def to_dict(self):
        """Convert settings to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'language': self.language,
            'voice_speed': self.voice_speed,
            'theme_mode': self.theme_mode,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        }