"""
Authentication module for Cake's Promptfoo integration
"""
import os
from typing import Optional, Dict, Any
from cake_auth import CakeAuthClient, CakeAuthConfig
from dotenv import load_dotenv
import urllib.parse

load_dotenv()

class PromptfooAuth:
    """Handle authentication for Promptfoo using Cake's auth system"""
    
    def __init__(self):
        self.base_url = os.getenv('PROMPTFOO_REMOTE_API_BASE_URL', 'https://promptfoo.dev.aws.kflow.ai')
        self.app_url = os.getenv('PROMPTFOO_REMOTE_APP_BASE_URL', 'https://promptfoo.dev.aws.kflow.ai')
        
        try:
            # Initialize Cake auth from environment
            self.config = CakeAuthConfig.from_env()
            self.client = CakeAuthClient(self.config)
            self.auth_available = True
        except Exception as e:
            print(f"⚠️  Could not initialize Cake auth: {e}")
            self.auth_available = False
            self.config = None
            self.client = None
        
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Promptfoo API calls"""
        # Get the auth token from Cake's auth system
        token = self.client.get_token()
        
        return {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    def get_auth_callback_url(self, service_name: str = "promptfoo") -> str:
        """Get the OAuth callback URL for the service"""
        # Cake uses a specific pattern for OAuth callbacks
        cluster_base = os.getenv('CLUSTER_BASE_NAME', 'dev.aws.kflow.ai')
        return f"https://oauth.{cluster_base}/callback?service={service_name}"
    
    def get_authenticated_url(self, target_url: str) -> str:
        """Build authenticated redirect URL using Cake's OAuth pattern"""
        # Cake's OAuth flow expects: /auth?rd=<encoded_target_url>
        cluster_base = os.getenv('CLUSTER_BASE_NAME', 'dev.aws.kflow.ai')
        encoded_target = urllib.parse.quote(target_url, safe='')
        return f"https://oauth.{cluster_base}/auth?rd={encoded_target}"
    
    def configure_promptfoo_env(self):
        """Configure environment variables for Promptfoo with authentication"""
        # Check for manual JWT token first
        manual_token = os.getenv('CAKE_JWT_TOKEN')
        if manual_token:
            print("🔑 Using manual JWT token")
            os.environ['PROMPTFOO_AUTH_TOKEN'] = manual_token
            os.environ['PROMPTFOO_SHARE_API_BASE_URL'] = self.base_url
            os.environ['PROMPTFOO_SHARE_APP_BASE_URL'] = self.app_url
            os.environ['PROMPTFOO_API_HEADERS'] = f'Authorization: Bearer {manual_token}'
            print(f"✓ Authentication configured for {self.base_url}")
            return True
            
        if not self.auth_available:
            print("⚠️  Cake auth not available - running without authentication")
            print("To enable auth, you can:")
            print("  1. Set CAKE_JWT_TOKEN with a manual token")
            print("  2. Set CAKE_DEX_INFERENCE_PASSWORD")
            print("  3. Or configure AWS credentials for Secrets Manager")
            return
            
        try:
            # Check which auth method is available
            auth_method = self.config.get_recommended_auth_method()
            print(f"Using auth method: {auth_method}")
            
            # Get auth token
            token = self.client.get_token()
            
            if token:
                # Set the auth token for Promptfoo
                os.environ['PROMPTFOO_AUTH_TOKEN'] = token
                
                # For remote eval, Promptfoo needs these env vars
                os.environ['PROMPTFOO_SHARE_API_BASE_URL'] = self.base_url
                os.environ['PROMPTFOO_SHARE_APP_BASE_URL'] = self.app_url
                
                # Also set headers for the API calls
                os.environ['PROMPTFOO_API_HEADERS'] = f'Authorization: Bearer {token}'
                
                print(f"✓ Authentication configured for {self.base_url}")
                return True
            else:
                print("⚠️  No auth token retrieved - authentication may fail")
                return False
                
        except Exception as e:
            print(f"⚠️  Authentication setup failed: {e}")
            
            # If it's a browser flow error, suggest the browser auth
            if "browser" in str(e).lower():
                print("\nTry browser authentication:")
                print(f"  1. Run: python -m cake_auth login")
                print(f"  2. Complete authentication in browser")
                print(f"  3. Retry the evaluation")
            
            return False
