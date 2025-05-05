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