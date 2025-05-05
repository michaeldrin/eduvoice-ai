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
    
    # Get credentials with error checking
    client_id = os.environ.get('GOOGLE_OAUTH_CLIENT_ID')
    client_secret = os.environ.get('GOOGLE_OAUTH_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        app.logger.error("Google OAuth credentials missing. Please check environment variables.")
        missing = []
        if not client_id:
            missing.append("GOOGLE_OAUTH_CLIENT_ID")
        if not client_secret:
            missing.append("GOOGLE_OAUTH_CLIENT_SECRET")
        app.logger.error(f"Missing OAuth credentials: {', '.join(missing)}")
    
    # Log the domain used for Replit
    replit_domain = os.environ.get('REPLIT_DEV_DOMAIN') 
    if replit_domain:
        app.logger.info(f"Detected Replit environment with domain: {replit_domain}")
        # Print clear instructions for the redirect URI setup
        print(f"\n====================== GOOGLE OAUTH SETUP =======================")
        print(f"To make Google authentication work, add this EXACT URI to your")
        print(f"authorized redirect URIs in Google Cloud Console:")
        print(f"https://{replit_domain}/callback")
        print(f"================================================================\n")
    
    # Register the Google OAuth provider with explicit parameters
    oauth.register(
        name='google',
        client_id=client_id,
        client_secret=client_secret,
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={
            'scope': 'openid email profile',
            'prompt': 'select_account',  # Always ask which account to use
            'access_type': 'online'      # We don't need offline access
        }
    )


@auth_bp.route('/login')
def login():
    """Redirect to Google for OAuth login"""
    # Verify OAuth credentials are available
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
    
    # Determine the correct callback URL based on environment
    replit_domain = os.environ.get('REPLIT_DEV_DOMAIN')
    
    # For Replit, construct the callback URL using the REPLIT_DEV_DOMAIN
    if replit_domain:
        # This must exactly match what's in your Google Cloud Console
        redirect_uri = f"https://{replit_domain}/callback"
        
        # Log the exact URI for debugging purposes
        current_app.logger.info(f"Using Replit callback URI: {redirect_uri}")
        current_app.logger.info("If login fails with 403 error, verify this exact URI is added to Google Cloud Console")
        
        # Print instructions to console to make it very clear
        print(f"\n====================== GOOGLE OAUTH SETUP =======================")
        print(f"IMPORTANT: Add this exact URI to your Google Cloud Console")
        print(f"(OAuth 2.0 Client IDs → Authorized redirect URIs):")
        print(f"")
        print(f"{redirect_uri}")
        print(f"")
        print(f"If you continue to experience 403 errors:")
        print(f"1. Ensure there are no extra spaces or characters in the URI")
        print(f"2. If your app is in 'Testing' mode, add your email to test users")
        print(f"3. Make sure Google+ API or People API is enabled")
        print(f"================================================================\n")
    else:
        # For local development - though this won't work for Google OAuth typically
        redirect_uri = url_for('oauth_callback', _external=True, _scheme='https')
        current_app.logger.info(f"Using local development redirect URI: {redirect_uri}")
    
    try:
        # Save the redirect URI used in the session for verification during callback
        session['oauth_redirect_uri'] = redirect_uri
        
        # Include explicit parameters for more reliable OAuth flow
        return oauth.google.authorize_redirect(
            redirect_uri=redirect_uri,
            prompt='select_account'  # Always show account selector
        )
    except Exception as e:
        current_app.logger.error(f"Error in OAuth redirect: {str(e)}")
        current_app.logger.error(f"Error type: {type(e).__name__}")
        # Create a more detailed error message for debugging
        error_details = f"Type: {type(e).__name__}, Message: {str(e)}"
        return redirect(url_for('home_page', error=f"Error starting authentication: {error_details}"))


@auth_bp.route('/callback')
def callback():
    """Callback endpoint for Google OAuth"""
    try:
        # Extremely detailed logging for debugging 403 errors
        current_app.logger.info(f"==================== OAUTH CALLBACK ====================")
        current_app.logger.info(f"Callback received at: {datetime.datetime.now().isoformat()}")
        current_app.logger.info(f"Full URL: {request.url}")
        current_app.logger.info(f"Request method: {request.method}")
        current_app.logger.info(f"URL parameters: {request.args}")
        current_app.logger.info(f"Headers: {dict(request.headers)}")
        
        # Check for explicit error response from Google
        if 'error' in request.args:
            error_msg = request.args.get('error')
            error_description = request.args.get('error_description', 'No description provided')
            current_app.logger.error(f"OAUTH ERROR RESPONSE: {error_msg}")
            current_app.logger.error(f"Error description: {error_description}")
            
            # Log helpful information for common errors
            if error_msg == 'access_denied':
                current_app.logger.error("The user declined to grant access to your application")
            elif error_msg == 'redirect_uri_mismatch':
                current_app.logger.error("The redirect URI in the request does not match the one registered in Google Console")
                current_app.logger.error(f"Expected URI from session: {session.get('oauth_redirect_uri', 'Not found in session')}")
                # Print the instructions again since this is the most common error
                replit_domain = os.environ.get('REPLIT_DEV_DOMAIN')
                if replit_domain:
                    uri = f"https://{replit_domain}/callback"
                    print(f"\n====================== OAUTH ERROR: REDIRECT URI MISMATCH =======================")
                    print(f"Add this EXACT redirect URI to Google Cloud Console:")
                    print(f"{uri}")
                    print(f"=============================================================================\n")
            
            return redirect(url_for('home_page', error=f"Google authentication error: {error_msg} - {error_description}"))
            
        # Validate HTTPS (Google OAuth requires HTTPS)
        if not request.url.startswith('https://') and not request.url.startswith('http://localhost'):
            current_app.logger.error(f"Non-HTTPS callback URL: {request.url}")
            return redirect(url_for('home_page', error="OAuth error: HTTPS is required. Please use the secure URL."))
        
        # Verify state parameter to prevent CSRF attacks
        if 'state' not in request.args:
            current_app.logger.error("Missing state parameter in callback - potential CSRF attack")
            return redirect(url_for('home_page', error="OAuth error: Missing state parameter. This may indicate a security issue."))
        
        # Get the token with enhanced error handling
        try:
            # Log what we're about to do
            current_app.logger.info("Attempting to obtain access token from Google")
            
            # Get the token
            token = oauth.google.authorize_access_token()
            
            # Success - log what we received (securely)
            current_app.logger.info("✓ Token successfully retrieved from Google")
            if token:
                # Log token info excluding sensitive parts
                safe_token_info = {
                    k: v for k, v in token.items() 
                    if k not in ['access_token', 'id_token', 'refresh_token']
                }
                # Log expiration time if available
                if 'expires_at' in token:
                    import time
                    expires_at = token['expires_at']
                    current_time = time.time()
                    expiry_delta = expires_at - current_time
                    safe_token_info['expires_in_seconds'] = expiry_delta
                    
                current_app.logger.info(f"Token metadata: {safe_token_info}")
            else:
                current_app.logger.warning("Token response is empty or None - this is unexpected")
                
        except Exception as token_error:
            # Comprehensive error logging for token acquisition failures
            current_app.logger.error("✗ Failed to get token from Google")
            current_app.logger.error(f"Error type: {type(token_error).__name__}")
            current_app.logger.error(f"Error message: {str(token_error)}")
            
            # Try to extract more detailed information from the error
            error_details = str(token_error)
            
            # Check for OAuth-specific error attributes
            if hasattr(token_error, 'description'):
                error_description = getattr(token_error, 'description', 'No description available')
                current_app.logger.error(f"OAuth error description: {error_description}")
                error_details = f"{type(token_error).__name__}: {error_description}"
            
            # Check for HTTP response details if available
            if hasattr(token_error, 'response'):
                response = getattr(token_error, 'response', None)
                if response:
                    current_app.logger.error(f"Response status code: {getattr(response, 'status_code', 'Unknown')}")
                    
                    # Try to get JSON details from response
                    try:
                        if hasattr(response, 'json') and callable(response.json):
                            error_json = response.json()
                            current_app.logger.error(f"Full error response: {error_json}")
                            
                            # Extract the most relevant parts of the error
                            if 'error' in error_json:
                                if isinstance(error_json['error'], str):
                                    error_code = error_json['error']
                                    error_desc = error_json.get('error_description', '')
                                    error_details = f"{error_code}: {error_desc}"
                                elif isinstance(error_json['error'], dict):
                                    error_code = error_json['error'].get('code', '')
                                    error_message = error_json['error'].get('message', '')
                                    error_details = f"Code {error_code}: {error_message}"
                    except Exception as parse_err:
                        current_app.logger.error(f"Failed to parse error response: {str(parse_err)}")
                        
                        # Try to get text content if JSON parsing failed
                        if hasattr(response, 'text'):
                            current_app.logger.error(f"Response text: {getattr(response, 'text', 'No text content')[:1000]}")
            
            # Display a user-friendly error message with detailed context
            return redirect(url_for('home_page', error=f"""
                Authentication failed: Could not get access token.
                
                Error: {error_details}
                
                Please check:
                1. The redirect URI is exactly correct in Google Cloud Console
                2. Your application has the necessary API permissions
                3. Your account has access if the app is in testing mode
            """.strip().replace('\n', ' ')))
        
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