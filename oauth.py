import os
import json
import datetime
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
    # Get the Replit domain from environment variables
    replit_domain = os.environ.get('REPLIT_DEV_DOMAIN')
    if replit_domain:
        # Force the HTTPS protocol for the redirect URI
        redirect_uri = f"https://{replit_domain}/callback"
        current_app.logger.info(f"Using Replit redirect URI: {redirect_uri}")
    else:
        # Fallback to default url_for behavior
        redirect_uri = url_for('auth.callback', _external=True)
        current_app.logger.info(f"Using standard redirect URI: {redirect_uri}")
    
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.route('/callback')
def callback():
    """Callback endpoint for Google OAuth"""
    try:
        # Detailed logging for debugging
        current_app.logger.info(f"Callback received: {request.url}")
        
        # Get the token
        token = oauth.google.authorize_access_token()
        current_app.logger.info("Token retrieved successfully")
        
        # Get user info from token
        resp = oauth.google.get('https://openidconnect.googleapis.com/v1/userinfo')
        user_info = resp.json()
        
        if user_info and 'email' in user_info:
            # Store user info in session
            session['user'] = {
                'id': user_info.get('sub'),
                'email': user_info.get('email'),
                'name': user_info.get('name', user_info.get('email')),
                'picture': user_info.get('picture', ''),
                'logged_in_at': datetime.datetime.now().isoformat()
            }
            
            current_app.logger.info(f"User logged in: {user_info.get('email')}")
        else:
            current_app.logger.error(f"Failed to get user info: {user_info}")
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


# Method to be called from the root-level callback route
def handle_google_callback():
    """Method to handle Google OAuth callback from root level"""
    return callback()

# Decorator to require login for routes
def login_required(view_func):
    """Decorator to require login for views"""
    def wrapped_view(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('auth.login'))
        return view_func(*args, **kwargs)
    wrapped_view.__name__ = view_func.__name__
    return wrapped_view