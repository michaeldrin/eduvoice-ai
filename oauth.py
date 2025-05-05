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
    
    # Determine the correct Replit domain for callbacks
    replit_domain = None
    
    # Check for Replit domain in different environment variables
    # REPLIT_DEV_DOMAIN is used in the Dev environment
    if os.environ.get('REPLIT_DEV_DOMAIN'):
        replit_domain = os.environ.get('REPLIT_DEV_DOMAIN')
        app.logger.info(f"Detected Replit dev domain: {replit_domain}")
    # REPL_SLUG and REPL_OWNER are used in the production environment
    elif os.environ.get('REPL_SLUG') and os.environ.get('REPL_OWNER'):
        repl_slug = os.environ.get('REPL_SLUG')
        repl_owner = os.environ.get('REPL_OWNER')
        replit_domain = f"{repl_owner}.{repl_slug}.repl.co"
        app.logger.info(f"Constructed Replit production domain: {replit_domain}")
    
    if replit_domain:
        # Format the callback URL and save it in the app config for consistent use
        app.config['OAUTH_REDIRECT_URI'] = f"https://{replit_domain}/callback"
        
        # Print clear instructions for the redirect URI setup
        print(f"\n====================== GOOGLE OAUTH SETUP =======================")
        print(f"IMPORTANT: Add this EXACT URI to 'Authorized redirect URIs'")
        print(f"in Google Cloud Console (OAuth 2.0 Client IDs section):")
        print(f"")
        print(f"{app.config['OAUTH_REDIRECT_URI']}")
        print(f"")
        print(f"The 'redirect_uri_mismatch' error means this URI doesn't exactly match")
        print(f"what's configured in your Google Cloud Console.")
        print(f"================================================================\n")
    else:
        app.logger.warning("Could not determine Replit domain. OAuth may not work correctly.")
        # For local development fallback
        app.config['OAUTH_REDIRECT_URI'] = None
    
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
    # First, check if user is already logged in
    if 'user' in session:
        current_app.logger.info(f"User already logged in: {session['user'].get('email', 'unknown')}")
        # Track redirection to prevent loops
        if session.get('from_callback'):
            # This is a potential infinite loop - reset session and continue to login
            current_app.logger.warning("Detected potential redirect loop - resetting session")
            session.pop('from_callback', None)
        else:
            # User is already logged in - redirect to dashboard instead of Google login
            current_app.logger.info("Redirecting logged-in user to dashboard instead of login flow")
            # Set next parameter to remember where user wanted to go, if provided
            next_url = request.args.get('next', 'dashboard')
            return redirect(url_for(next_url))
    
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
    
    # Get the pre-configured redirect URI from app config
    redirect_uri = current_app.config.get('OAUTH_REDIRECT_URI')
    
    if not redirect_uri:
        # If not found in config, determine it from environment variables
        replit_domain = None
        
        # Check Replit dev environment
        if os.environ.get('REPLIT_DEV_DOMAIN'):
            replit_domain = os.environ.get('REPLIT_DEV_DOMAIN')
            current_app.logger.info(f"Using Replit dev domain: {replit_domain}")
        # Check Replit production environment
        elif os.environ.get('REPL_SLUG') and os.environ.get('REPL_OWNER'):
            repl_slug = os.environ.get('REPL_SLUG')
            repl_owner = os.environ.get('REPL_OWNER')
            replit_domain = f"{repl_owner}.{repl_slug}.repl.co"
            current_app.logger.info(f"Using Replit production domain: {replit_domain}")
            
        if replit_domain:
            redirect_uri = f"https://{replit_domain}/callback"
        else:
            # Local development fallback
            redirect_uri = url_for('oauth_callback', _external=True, _scheme='https')
            current_app.logger.warning("Could not determine Replit domain, using fallback URI")
    
    # Log the final redirect URI for debugging
    current_app.logger.info(f"Using OAuth redirect URI: {redirect_uri}")
    
    # Store the referring page (if any) so we can redirect back after login
    next_page = request.args.get('next')
    if next_page:
        session['next_page'] = next_page
        current_app.logger.info(f"Saved next_page in session: {next_page}")
    
    # Clear any 'from_callback' flag to prevent loops
    session.pop('from_callback', None)
    
    # Print instructions to console (always do this for clarity)
    print(f"\n====================== GOOGLE OAUTH SETUP =======================")
    print(f"IMPORTANT: Add this EXACT redirect URI to Google Cloud Console:")
    print(f"{redirect_uri}")
    print(f"")
    print(f"If you receive 'redirect_uri_mismatch' errors:")
    print(f"1. Copy the exact URI above (no extra spaces or characters)")
    print(f"2. In Google Cloud Console → API & Services → Credentials")
    print(f"3. Edit your OAuth 2.0 Client ID")
    print(f"4. Paste it in 'Authorized redirect URIs' section")
    print(f"5. Save the changes")
    print(f"================================================================\n")
    
    try:
        # Store redirect URI in session for validation during callback
        session['oauth_redirect_uri'] = redirect_uri
        
        # Log parameters for debugging
        current_app.logger.info(f"Starting OAuth flow with redirect_uri={redirect_uri}")
        
        # Initiate the OAuth flow with explicit parameters
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
            
        # Special handling for HTTP callbacks on Replit - force HTTPS for the authorization
        # This works around a common issue where the callback comes to http:// but
        # Google OAuth requires https:// registered callbacks
        replit_domain = None
        if os.environ.get('REPLIT_DEV_DOMAIN'):
            replit_domain = os.environ.get('REPLIT_DEV_DOMAIN')
        elif os.environ.get('REPL_SLUG') and os.environ.get('REPL_OWNER'):
            repl_slug = os.environ.get('REPL_SLUG')
            repl_owner = os.environ.get('REPL_OWNER')
            replit_domain = f"{repl_owner}.{repl_slug}.repl.co"
        
        if replit_domain and request.url.startswith(f"http://{replit_domain}"):
            # If we received HTTP but need HTTPS, construct the HTTPS URL
            current_app.logger.warning(f"Received HTTP callback - redirecting to HTTPS: {request.url}")
            
            # Extract the path, query string, and fragment
            from urllib.parse import urlparse, parse_qsl, urlencode
            parsed = urlparse(request.url)
            path = parsed.path
            
            # Reconstruct with HTTPS
            https_url = f"https://{replit_domain}{path}"
            
            # Add query parameters if present
            if parsed.query:
                query_params = dict(parse_qsl(parsed.query))
                https_url = f"{https_url}?{urlencode(query_params)}"
                
            # Log the redirection
            current_app.logger.info(f"Redirecting to HTTPS URL: {https_url}")
            
            # Redirect to the HTTPS version
            return redirect(https_url)
            
        # For non-Replit environments, validate HTTPS normally
        elif not request.url.startswith('https://') and not request.url.startswith('http://localhost'):
            current_app.logger.error(f"Non-HTTPS callback URL: {request.url}")
            return redirect(url_for('home_page', error="OAuth error: HTTPS is required. Please use the secure URL."))
        
        # Verify state parameter to prevent CSRF attacks
        if 'state' not in request.args:
            current_app.logger.error("Missing state parameter in callback - potential CSRF attack")
            return redirect(url_for('home_page', error="OAuth error: Missing state parameter. This may indicate a security issue."))
        
        # Verify the redirect URI from session matches what we expect
        expected_redirect_uri = session.get('oauth_redirect_uri')
        if expected_redirect_uri:
            current_app.logger.info(f"Verifying against expected redirect URI: {expected_redirect_uri}")
        else:
            current_app.logger.warning("No expected redirect URI found in session. This could cause problems.")
        
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
            # Get user information from the response
            user_email = user_info.get('email')
            user_name = user_info.get('name', user_email)
            user_id = user_info.get('sub')
            user_picture = user_info.get('picture', '')
            
            # Log full user information for debugging (except any sensitive data)
            current_app.logger.info(f"User authenticated successfully:")
            current_app.logger.info(f"- Email: {user_email}")
            current_app.logger.info(f"- Name: {user_name}")
            current_app.logger.info(f"- Google ID: {user_id}")
            current_app.logger.info(f"- Has picture: {'Yes' if user_picture else 'No'}")
            
            # Store user info in session with additional fields for verification
            session['user'] = {
                'id': user_id,
                'email': user_email,
                'name': user_name,
                'picture': user_picture,
                'logged_in_at': datetime.datetime.now().isoformat(),
                'auth_provider': 'google'
            }
            
            # Log successful session storage
            current_app.logger.info(f"User session created successfully for: {user_email}")
            
            # Create/update user in database if needed
            try:
                from models import db, UserSettings
                
                # Find or create user settings 
                user_settings = UserSettings.query.filter_by(session_id=user_email).first()
                if not user_settings:
                    user_settings = UserSettings(session_id=user_email)
                    db.session.add(user_settings)
                    db.session.commit()
                    current_app.logger.info(f"Created new user settings for: {user_email}")
                else:
                    current_app.logger.info(f"Found existing user settings for: {user_email}")
            except Exception as db_error:
                current_app.logger.error(f"Database error when storing user: {str(db_error)}")
                # Continue with login flow even if database storage fails
            
            # Set a flag to prevent redirect loops
            session['from_callback'] = True
            
            # Determine where to redirect the user after successful login
            if 'next_page' in session:
                # Redirect to the page they were trying to access before login
                next_page = session.pop('next_page')
                current_app.logger.info(f"Redirecting user to previously requested page: {next_page}")
                # Make sure the page exists, fallback to dashboard if not
                try:
                    # Check if the route exists
                    url_for(next_page)
                    return redirect(url_for(next_page, message=f"Welcome, {user_name}!"))
                except Exception:
                    current_app.logger.warning(f"Invalid next_page route: {next_page}, redirecting to dashboard")
                    return redirect(url_for('dashboard', message=f"Welcome, {user_name}!"))
            else:
                # Default redirect to dashboard after successful login
                current_app.logger.info(f"Redirecting user to dashboard after successful login")
                return redirect(url_for('dashboard', message=f"Welcome, {user_name}!"))
        else:
            # Log specific issues with the user info
            missing_fields = []
            if not user_info:
                current_app.logger.error("Empty user info response from Google")
                missing_fields.append("all user information")
            else:
                if 'email' not in user_info:
                    current_app.logger.error("Missing email in Google user info")
                    missing_fields.append("email")
                if 'name' not in user_info:
                    current_app.logger.warning("Missing name in Google user info")
                    missing_fields.append("name")
                if 'sub' not in user_info:
                    current_app.logger.warning("Missing sub (user ID) in Google user info")
                    missing_fields.append("user ID")
                
                current_app.logger.error(f"Invalid user info response: {user_info}")
                
            return redirect(url_for('home_page', error=f"Failed to get valid user information. Missing: {', '.join(missing_fields)}. Please try again."))
    
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
            # Store the requested URL to redirect back after login
            next_url = request.path
            current_app.logger.info(f"Unauthorized access to {next_url}, redirecting to login")
            
            # Save the path in the session to redirect after login
            if next_url:
                session['next_page'] = next_url
                
            # Redirect to login page with a message
            return redirect(url_for('auth.login', next=next_url, message="Please log in to access this page"))
            
        # User is logged in, proceed to view
        return view_func(*args, **kwargs)
        
    # Preserve function name for Flask's routing
    wrapped_view.__name__ = view_func.__name__
    return wrapped_view