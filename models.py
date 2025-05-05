from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Document(db.Model):
    """Model for uploaded documents and their summaries"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(255), nullable=True)  # User's email or guest ID
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
            'user_id': self.user_id,
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
    
    # Accessibility settings
    accessibility_mode = db.Column(db.Boolean, default=False)  # Enable/disable accessibility features
    font_size = db.Column(db.String(10), default='medium')  # Small, Medium, Large, X-Large
    high_contrast = db.Column(db.Boolean, default=False)  # High contrast mode
    dyslexia_friendly = db.Column(db.Boolean, default=False)  # Dyslexia-friendly font
    line_spacing = db.Column(db.Float, default=1.5)  # Line spacing factor
    reduce_animations = db.Column(db.Boolean, default=False)  # Reduce UI animations
    
    # Usage tracking fields
    files_uploaded = db.Column(db.Integer, default=0)  # Count of files uploaded
    summaries_generated = db.Column(db.Integer, default=0)  # Count of summaries
    audio_minutes = db.Column(db.Float, default=0.0)  # Total minutes of audio generated
    last_reset_date = db.Column(db.Date, default=datetime.utcnow().date())  # Last counter reset
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Usage limits as class constants
    MAX_FILES_PER_DAY = 5
    MAX_SUMMARIES = 10
    MAX_AUDIO_MINUTES = 15.0
    
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
            
            # Accessibility settings
            'accessibility_mode': self.accessibility_mode,
            'font_size': self.font_size,
            'high_contrast': self.high_contrast,
            'dyslexia_friendly': self.dyslexia_friendly,
            'line_spacing': self.line_spacing,
            'reduce_animations': self.reduce_animations,
            
            # Usage stats
            'files_uploaded': self.files_uploaded,
            'summaries_generated': self.summaries_generated,
            'audio_minutes': self.audio_minutes,
            'last_reset_date': self.last_reset_date.strftime('%Y-%m-%d') if self.last_reset_date else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def check_daily_reset(self):
        """Reset daily counters if last reset was not today"""
        today = datetime.utcnow().date()
        if self.last_reset_date < today:
            self.files_uploaded = 0
            self.last_reset_date = today
            return True
        return False
    
    def can_upload_file(self):
        """Check if user can upload more files today"""
        self.check_daily_reset()
        return self.files_uploaded < self.MAX_FILES_PER_DAY
    
    def can_generate_summary(self):
        """Check if user can generate more summaries"""
        return self.summaries_generated < self.MAX_SUMMARIES
    
    def can_generate_audio(self, minutes_to_add=0.0):
        """Check if user can generate more audio"""
        return (self.audio_minutes + minutes_to_add) <= self.MAX_AUDIO_MINUTES
    
    def increment_file_uploads(self):
        """Increment file uploads counter"""
        self.check_daily_reset()
        self.files_uploaded += 1
        
    def increment_summaries(self):
        """Increment summaries counter"""
        self.summaries_generated += 1
    
    def add_audio_minutes(self, minutes):
        """Add minutes to audio counter"""
        if minutes > 0:
            self.audio_minutes += minutes