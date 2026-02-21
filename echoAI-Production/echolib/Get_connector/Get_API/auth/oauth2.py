"""
OAuth2 authentication implementation.
Supports multiple grant types including client credentials, authorization code, and refresh tokens.
"""

from typing import Dict, Any, Optional
import httpx
import time
from echolib.Get_connector.Get_API.auth.base import AuthBase
from echolib.Get_connector.Get_API.models.config import OAuth2GrantType
from urllib.parse import parse_qs


class OAuth2Auth(AuthBase):
    """
    OAuth2 authentication strategy.
    
    Supports:
    - Client Credentials flow
    - Authorization Code flow
    - Refresh Token flow
    
    Automatically handles token refresh when expired.
    """
    
    def __init__(
        self,
        grant_type: OAuth2GrantType,
        token_url: str,
        client_id: str,
        client_secret: str = "",
        scope: Optional[str] = None,
        authorization_url: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        code: Optional[str] = None,
        device_code_url: Optional[str] = None,
        device_code: Optional[str] = None,
        refresh_token: Optional[str] = None,
        access_token: Optional[str] = None,
        token_expiry: Optional[int] = None,
        send_client_secret_in_device_code: bool = False,
        custom_token_params: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Initialize OAuth2 authentication.
        
        Args:
            grant_type: OAuth2 grant type
            token_url: Token endpoint URL
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret (not used for device code)
            scope: Requested scopes (optional)
            authorization_url: Authorization endpoint (for auth code flow)
            redirect_uri: Redirect URI (for auth code flow)
            code: Authorization code (for auth code flow)
            device_code_url: Device code endpoint (for device code flow)
            device_code: Device code (for device code flow)
            refresh_token: Refresh token
            access_token: Current access token (cached)
            token_expiry: Token expiry timestamp (cached)
            
        Raises:
            ValueError: If required parameters are missing
        """
        if not token_url or not client_id:
            raise ValueError("token_url and client_id are required")
        
        # Client secret required for all except device code
        if grant_type != OAuth2GrantType.DEVICE_CODE:
            if not client_secret:
                raise ValueError(f"{grant_type.value} requires client_secret")
        
        self.grant_type = grant_type
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.authorization_url = authorization_url
        self.redirect_uri = redirect_uri
        self.code = code
        self.device_code_url = device_code_url
        self.device_code = device_code
        self.refresh_token = refresh_token
        self.access_token = access_token
        self.token_expiry = token_expiry
        self.send_client_secret_in_device_code = send_client_secret_in_device_code
        self.custom_token_params = custom_token_params or {}
    
    def _is_token_expired(self) -> bool:
        """
        Check if the current access token is expired.
        
        Returns:
            True if token is expired or missing, False otherwise
        """
        if not self.access_token:
            return True
        
        if not self.token_expiry:
            return False
        
        # Add 60 second buffer before actual expiry
        return int(time.time()) >= (self.token_expiry - 60)
    
    def _fetch_token(self) -> Dict[str, Any]:
        """
        Fetch a new access token from the OAuth2 provider.
        
        Returns:
            Token response dictionary
            
        Raises:
            ValueError: If token fetch fails
        """
        from urllib.parse import parse_qs
        
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        if self.grant_type == OAuth2GrantType.CLIENT_CREDENTIALS:
            data["grant_type"] = "client_credentials"
            if self.scope:
                data["scope"] = self.scope
        
        elif self.grant_type == OAuth2GrantType.AUTHORIZATION_CODE:
            if not self.code or not self.redirect_uri:
                raise ValueError("Authorization code and redirect_uri required for auth code flow")
            data["grant_type"] = "authorization_code"
            data["code"] = self.code
            data["redirect_uri"] = self.redirect_uri
        
        elif self.grant_type == OAuth2GrantType.REFRESH_TOKEN:
            if not self.refresh_token:
                raise ValueError("Refresh token required for refresh token flow")
            data["grant_type"] = "refresh_token"
            data["refresh_token"] = self.refresh_token
        
        elif self.grant_type == OAuth2GrantType.DEVICE_CODE:
            if not self.device_code:
                raise ValueError("Device code required for device code flow")
            data["grant_type"] = "urn:ietf:params:oauth:grant-type:device_code"
            data["device_code"] = self.device_code
        
        try:
            with httpx.Client() as client:
                response = client.post(
                    self.token_url,
                    data=data,
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json"
                    }
                )
                response.raise_for_status()
                
                # Parse response based on Content-Type
                content_type = response.headers.get("content-type", "").lower()
                
                if "application/json" in content_type:
                    token_data = response.json()
                elif "application/x-www-form-urlencoded" in content_type:
                    parsed = parse_qs(response.text)
                    token_data = {k: v[0] if isinstance(v, list) and len(v) == 1 else v 
                                for k, v in parsed.items()}
                else:
                    try:
                        token_data = response.json()
                    except Exception:
                        parsed = parse_qs(response.text)
                        token_data = {k: v[0] if isinstance(v, list) and len(v) == 1 else v 
                                    for k, v in parsed.items()}
                
                if "access_token" not in token_data or not token_data["access_token"]:
                    raise ValueError("OAuth2 response missing access_token")
                
                return token_data
        
        except httpx.HTTPError as e:
            error_msg = f"OAuth2 token exchange failed\n"
            if hasattr(e, 'response') and e.response:
                error_msg += f"Raw response: {e.response.text}\n"
            error_msg += f"Error: {str(e)}"
            raise ValueError(error_msg)
        except Exception as e:
            raise ValueError(f"Unexpected error fetching OAuth2 token: {str(e)}")
    
    def _update_token(self, token_response: Dict[str, Any]) -> None:
        """
        Update internal token state from OAuth2 response.
        
        Args:
            token_response: OAuth2 token response
        """
        self.access_token = token_response.get("access_token")
        
        if "expires_in" in token_response:
            self.token_expiry = int(time.time()) + token_response["expires_in"]
        
        if "refresh_token" in token_response:
            self.refresh_token = token_response["refresh_token"]
    
    def _request_device_code(self) -> Dict[str, Any]:
        """
        Request device code from provider.
        
        Returns:
            Device code response with device_code, user_code, verification_uri
            
        Raises:
            ValueError: If request fails
        """
        from urllib.parse import parse_qs
        
        # Build request data based on what provider needs
        data = {"client_id": self.client_id}
        
        # Some providers need client_secret, some don't
        if self.send_client_secret_in_device_code and self.client_secret:
            data["client_secret"] = self.client_secret
        
        if self.scope:
            data["scope"] = self.scope
        
        # Add any custom params the provider needs
        if self.custom_token_params:
            data.update(self.custom_token_params)
        
        try:
            with httpx.Client() as client:
                response = client.post(
                    self.device_code_url,
                    data=data,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/x-www-form-urlencoded"
                    }
                )
                
                # If request fails, show what we sent for debugging
                if response.status_code >= 400:
                    error_msg = f"Device code request failed\n"
                    error_msg += f"URL: {self.device_code_url}\n"
                    error_msg += f"Data sent: {data}\n"
                    error_msg += f"Status: {response.status_code}\n"
                    error_msg += f"Response: {response.text}"
                    raise ValueError(error_msg)
                
                response.raise_for_status()
                
                # Try to parse response - handle both JSON and form-encoded
                content_type = response.headers.get("content-type", "").lower()
                
                if "application/json" in content_type:
                    return response.json()
                elif "application/x-www-form-urlencoded" in content_type:
                    parsed = parse_qs(response.text)
                    return {k: v[0] if isinstance(v, list) and len(v) == 1 else v 
                            for k, v in parsed.items()}
                else:
                    # Try JSON first, fallback to form-encoded
                    try:
                        return response.json()
                    except Exception:
                        parsed = parse_qs(response.text)
                        return {k: v[0] if isinstance(v, list) and len(v) == 1 else v 
                                for k, v in parsed.items()}
        
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to request device code: {str(e)}")


    def _poll_device_token(self, device_code: str, interval: int = 5) -> Dict[str, Any]:
        """
        Poll for access token using device code.
        
        Args:
            device_code: Device code from initial request
            interval: Polling interval in seconds
            
        Returns:
            Token response
            
        Raises:
            ValueError: If polling fails or times out
        """
        from urllib.parse import parse_qs
        import time
        
        data = {
            "client_id": self.client_id,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
        }
        
        max_attempts = 60  # 5 minutes max
        attempt = 0
        
        while attempt < max_attempts:
            try:
                with httpx.Client() as client:
                    response = client.post(
                        self.token_url,
                        data=data,
                        headers={
                            "Accept": "application/json"
                        }
                    )
                    
                    content_type = response.headers.get("content-type", "").lower()
                    
                    if "application/json" in content_type:
                        token_data = response.json()
                    elif "application/x-www-form-urlencoded" in content_type:
                        parsed = parse_qs(response.text)
                        token_data = {k: v[0] if isinstance(v, list) and len(v) == 1 else v 
                                    for k, v in parsed.items()}
                    else:
                        try:
                            token_data = response.json()
                        except Exception:
                            parsed = parse_qs(response.text)
                            token_data = {k: v[0] if isinstance(v, list) and len(v) == 1 else v 
                                        for k, v in parsed.items()}
                    
                    # Check for errors
                    if "error" in token_data:
                        error_code = token_data["error"]
                        
                        if error_code == "authorization_pending":
                            # User hasn't authorized yet, continue polling
                            time.sleep(interval)
                            attempt += 1
                            continue
                        elif error_code == "slow_down":
                            # Provider wants us to slow down
                            interval += 5
                            time.sleep(interval)
                            attempt += 1
                            continue
                        elif error_code == "expired_token":
                            raise ValueError("Device code expired. Please restart the flow.")
                        elif error_code == "access_denied":
                            raise ValueError("User denied authorization.")
                        else:
                            raise ValueError(f"OAuth2 error: {error_code}")
                    
                    # Success
                    if "access_token" in token_data:
                        return token_data
                    
                    time.sleep(interval)
                    attempt += 1
            
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"Device code polling failed: {str(e)}")
        
        raise ValueError("Device code authorization timeout. User did not authorize in time.")
    
    def apply(
        self,
        headers: Dict[str, str],
        params: Dict[str, Any],
        client: httpx.Client
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        """
        Apply OAuth2 authentication.
        
        Args:
            headers: Request headers
            params: Query parameters
            client: HTTP client
            
        Returns:
            Modified (headers, params) with OAuth2 access token
            
        Raises:
            ValueError: If token acquisition fails
        """
        # Refresh token if needed
        if self._is_token_expired():
            token_response = self._fetch_token()
            self._update_token(token_response)
        
        if not self.access_token:
            raise ValueError("Failed to obtain access token")
        
        headers = headers.copy()
        headers["Authorization"] = f"Bearer {self.access_token}"
        return headers, params.copy()
    
    def refresh_if_needed(self) -> None:
        """
        Refresh the access token if expired.
        
        Raises:
            ValueError: If refresh fails
        """
        if self._is_token_expired():
            token_response = self._fetch_token()
            self._update_token(token_response)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "type": "oauth2",
            "grant_type": self.grant_type.value,
            "token_url": self.token_url,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": self.scope,
            "authorization_url": self.authorization_url,
            "redirect_uri": self.redirect_uri,
            "code": self.code,
            "device_code_url": self.device_code_url,
            "device_code": self.device_code,
            "refresh_token": self.refresh_token,
            "access_token": self.access_token,
            "token_expiry": self.token_expiry
        }
    
    @classmethod
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OAuth2Auth':
        """
        Deserialize from dictionary.
        
        Args:
            data: Dictionary containing auth configuration
            
        Returns:
            OAuth2Auth instance
            
        Raises:
            ValueError: If required fields are missing
        """
        required_fields = ["grant_type", "token_url", "client_id", "client_secret"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        return cls(
            grant_type=OAuth2GrantType(data["grant_type"]),
            token_url=data["token_url"],
            client_id=data["client_id"],
            client_secret=data["client_secret"],
            scope=data.get("scope"),
            authorization_url=data.get("authorization_url"),
            redirect_uri=data.get("redirect_uri"),
            code=data.get("code"),
            device_code_url=data.get("device_code_url"),
            device_code=data.get("device_code"),
            refresh_token=data.get("refresh_token"),
            access_token=data.get("access_token"),
            token_expiry=data.get("token_expiry")
        )
