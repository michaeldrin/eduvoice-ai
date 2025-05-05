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
    
    # Get the Google OAuth client ID to check if it's configured
    client_id = os.environ.get('GOOGLE_OAUTH_CLIENT_ID')
    client_secret = os.environ.get('GOOGLE_OAUTH_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        current_app.logger.error("Google OAuth credentials not found in environment variables")
        missing = []
        if not client_id:
            missing.append("GOOGLE_OAUTH_CLIENT_ID")
        if not client_secret:
            missing.append("GOOGLE_OAUTH_CLIENT_SECRET")
        return redirect(url_for('home_page', error=f"Google OAuth is not properly configured. Missing: {', '.join(missing)}"))
    
    # Log the available callback paths
    current_app.logger.debug(f"Auth blueprint callback: {url_for('auth.callback', _external=True)}")
    current_app.logger.debug(f"Root callback: {url_for('oauth_callback', _external=True)}")
    
    # For Replit, we need to use the specific format required by Google OAuth
    if replit_domain:
        # This is the format Google OAuth expects for Replit
        redirect_uri = f"https://{replit_domain}/callback"
        current_app.logger.info(f"Using Replit redirect URI: {redirect_uri}")
        
        # Print to console for easier setup
        print(f"\n====================== GOOGLE OAUTH SETUP =======================")
        print(f"To make Google authentication work, add this to your authorized")
        print(f"redirect URIs in Google Cloud Console:")
        print(f"{redirect_uri}")
        print(f"================================================================\n")
    else:
        # Fallback to default url_for behavior (for local development)
        redirect_uri = url_for('oauth_callback', _external=True) 
        current_app.logger.info(f"Using standard redirect URI: {redirect_uri}")
    
    # Log OAuth configuration for debugging
    current_app.logger.info(f"OAuth config: client_id={client_id[:5]}..., redirect_uri={redirect_uri}")
    
    try:
        return oauth.google.authorize_redirect(redirect_uri)
    except Exception as e:
        current_app.logger.error(f"Error in OAuth redirect: {str(e)}")
        return redirect(url_for('home_page', error=f"Error starting authentication: {str(e)}"))


@auth_bp.route('/callback')
def callback():
    """Callback endpoint for Google OAuth"""
    try:
        # Detailed logging for debugging
        current_app.logger.info(f"Callback received: {request.url}")
        current_app.logger.info(f"Callback args: {request.args}")
        
        # Check for error parameter in the callback URL
        if 'error' in request.args:
            error_msg = request.args.get('error')
            error_description = request.args.get('error_description', 'No description provided')
            current_app.logger.error(f"OAuth error in callback: {error_msg}")
            current_app.logger.error(f"Error description: {error_description}")
            return redirect(url_for('home_page', error=f"Google authentication error: {error_msg} - {error_description}"))
            
        # Check for HTTPS - Google OAuth requires HTTPS for the callback
        if not request.url.startswith('https://') and not request.url.startswith('http://localhost'):
            current_app.logger.error(f"Callback URL must use HTTPS: {request.url}")
            current_app.logger.error("If running on Replit, make sure you're using the HTTPS URL.")
            return redirect(url_for('home_page', error="OAuth error: Callback must use HTTPS. Please use the HTTPS URL for this application."))
        
        # Check for state parameter (important for CSRF protection)
        if 'state' not in request.args:
            current_app.logger.error("Missing state parameter in callback")
            return redirect(url_for('home_page', error="OAuth error: Missing state parameter. This may indicate a CSRF attack."))
        
        # Get the token
        try:
            token = oauth.google.authorize_access_token()
            current_app.logger.info("Token retrieved successfully")
            # Log token details for debugging (excluding the actual token value for security)
            if token:
                token_info = {k: v for k, v in token.items() if k != 'access_token' and k != 'id_token'}
                current_app.logger.info(f"Token info: {token_info}")
            else:
                current_app.logger.warning("Token is empty or None")
        except Exception as token_error:
            # Log detailed error information
            current_app.logger.error(f"Failed to get token: {str(token_error)}")
            current_app.logger.error(f"Error type: {type(token_error).__name__}")
            
            # Check for specific error types to provide better error messages
            if hasattr(token_error, 'description'):
                current_app.logger.error(f"Error description: {token_error.description}")
            
            # Extract error message from OAuth error response if available
            error_details = str(token_error)
            if hasattr(token_error, 'response') and hasattr(token_error.response, 'json'):
                try:
                    error_json = token_error.response.json()
                    current_app.logger.error(f"OAuth error response: {error_json}")
                    if 'error_description' in error_json:
                        error_details = f"{error_json.get('error', 'OAuth Error')}: {error_json.get('error_description')}"
                except Exception as json_err:
                    current_app.logger.error(f"Failed to parse error response: {str(json_err)}")
            
            return redirect(url_for('home_page', error=f"Authentication failed: Could not get token. Make sure the redirect URI is correctly set in Google console. Error: {error_details}"))
        
        # Get user info from token
        try:
            resp = oauth.google.get('https://openidconnect.googleapis.com/v1/userinfo')
            user_info = resp.json()
            current_app.logger.info(f"User info retrieved: {user_info.get('email', 'No email')}")
        except Exception as userinfo_error:
            current_app.logger.error(f"Failed to get user info: {str(userinfo_error)}")
            return redirect(url_for('home_page', error=f"Authentication failed: Could not get user info: {str(userinfo_error)}"))
        
        if user_info and 'email' in user_info:
            # Store user info in session
            session['user'] = {
                'id': user_info.get('sub'),
                'email': user_info.get('email'),
                'name': user_info.get('name', user_info.get('email')),
                'picture': user_info.get('picture', ''),
                'logged_in_at': datetime.datetime.now().isoformat()
            }
            
            current_app.logger.info(f"User logged in successfully: {user_info.get('email')}")
            # Include success message in the redirect
            return redirect(url_for('home_page', message=f"Welcome, {user_info.get('name', user_info.get('email'))}!"))
        else:
            current_app.logger.error(f"Failed to get valid user info: {user_info}")
            return redirect(url_for('home_page', error="Failed to get user info. Please try again."))
    
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