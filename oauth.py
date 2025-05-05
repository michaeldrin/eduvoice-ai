import os
import json
from flask import Blueprint, redirect, url_for, session, current_app, request, render_template
from authlib.integrations.flask_client import OAuth

# Initialize OAuth
oauth = OAuth()

# Create the blueprint
auth_bp = Blueprint('auth', __name__)

def init_oauth(app):
    """Initialize OAuth with the Flask app"""
    oauth.init_app(app)
    
    # Register the Google OAuth provider
    oauth.register(
        name='google',
        client_id=os.environ.get('GOOGLE_OAUTH_CLIENT_ID'),
        client_secret=os.environ.get('GOOGLE_OAUTH_CLIENT_SECRET'),
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={
            'scope': 'openid email profile'
        }
    )


@auth_bp.route('/login')
def login():
    """Redirect to Google for OAuth login"""
    # Generate a random state for security
    redirect_uri = url_for('auth.callback', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.route('/callback')
def callback():
    """Callback endpoint for Google OAuth"""
    try:
        # Get the token
        token = oauth.google.authorize_access_token()
        
        # Get user info
        user_info = token.get('userinfo')
        if user_info:
            # Store user info in session
            session['user'] = {
                'id': user_info.get('sub'),
                'email': user_info.get('email'),
                'name': user_info.get('name'),
                'picture': user_info.get('picture'),
                'logged_in_at': json.dumps(dict(token.get('expires_at')))
            }
            
            current_app.logger.info(f"User logged in: {user_info.get('email')}")
        else:
            current_app.logger.error("Failed to get user info from token")
            return redirect(url_for('home_page', error="Failed to get user info"))
        
        return redirect(url_for('home_page'))
    except Exception as e:
        current_app.logger.error(f"Error in OAuth callback: {str(e)}")
        return redirect(url_for('home_page', error=f"Authentication error: {str(e)}"))


@auth_bp.route('/logout')
def logout():
    """Log out user by clearing session"""
    session.pop('user', None)
    return redirect(url_for('home_page'))


@auth_bp.route('/profile')
def profile():
    """Display user profile information"""
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    
    from models import db, UserSettings
    
    # Get user settings based on user email (create if doesn't exist)
    user_settings = UserSettings.query.filter_by(
        session_id=session['user']['email']
    ).first()
    
    if not user_settings:
        user_settings = UserSettings(session_id=session['user']['email'])
        db.session.add(user_settings)
        db.session.commit()
    
    # Pass user settings to template
    return render_template(
        "profile.html",
        title="User Profile",
        user=session['user'],
        usage_stats=user_settings,
        theme_mode=user_settings.theme_mode
    )


# Decorator to require login for routes
def login_required(view_func):
    """Decorator to require login for views"""
    def wrapped_view(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('auth.login'))
        return view_func(*args, **kwargs)
    wrapped_view.__name__ = view_func.__name__
    return wrapped_view