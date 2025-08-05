"""
API Client Tool - Tool for making HTTP requests to external APIs.
Provides comprehensive API interaction capabilities with authentication and error handling.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid
import base64
from urllib.parse import urljoin, urlparse

from ...core.interfaces import ToolInterface, ToolResult, ToolParameter, ToolExecutionStatus


class APIClientTool(ToolInterface):
    """
    Tool for making HTTP requests to external APIs with comprehensive features.
    
    Features:
    - Multiple HTTP methods (GET, POST, PUT, DELETE, PATCH)
    - Authentication support (Basic, Bearer, API Key)
    - Request/response logging
    - Retry mechanisms
    - Timeout handling
    - Response caching
    """
    
    def __init__(self, tool_id: str, config: Dict[str, Any]):
        super().__init__(tool_id, config)
        self._logger = logging.getLogger(__name__)
        self._default_timeout = config.get('default_timeout_seconds', 30)
        self._max_retries = config.get('max_retries', 3)
        self._retry_delay = config.get('retry_delay_seconds', 1)
        self._enable_caching = config.get('enable_caching', True)
        self._cache_ttl_seconds = config.get('cache_ttl_seconds', 300)  # 5 minutes
        self._max_response_size = config.get('max_response_size_bytes', 10 * 1024 * 1024)  # 10MB
        self._allowed_hosts = config.get('allowed_hosts', [])  # Empty means all hosts allowed
        self._blocked_hosts = config.get('blocked_hosts', [])
        
        # Request/response cache and history
        self._response_cache: Dict[str, Dict[str, Any]] = {}
        self._request_history: List[Dict[str, Any]] = []
        self._max_history_size = config.get('max_history_size', 100)
        
        # Default headers
        self._default_headers = {
            'User-Agent': f'APIClientTool/{tool_id}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        self._default_headers.update(config.get('default_headers', {}))
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute API request with given parameters.
        
        Args:
            parameters: Request parameters including URL, method, headers, etc.
            
        Returns:
            ToolResult with API response
        """
        start_time = datetime.now()
        request_id = str(uuid.uuid4())
        
        try:
            # Validate parameters
            if not self.validate_parameters(parameters):
                return ToolResult(
                    status=ToolExecutionStatus.INVALID_INPUT,
                    data=None,
                    error_message="Invalid parameters provided"
                )
            
            url = parameters.get('url', '')
            method = parameters.get('method', 'GET').upper()
            headers = parameters.get('headers', {})
            body = parameters.get('body', None)
            auth = parameters.get('auth', None)
            timeout = parameters.get('timeout', self._default_timeout)
            enable_cache = parameters.get('enable_cache', self._enable_caching)
            max_retries = parameters.get('max_retries', self._max_retries)
            
            self._logger.info(f"Executing {method} request to {url} (ID: {request_id})")
            
            # Security checks
            security_check = self._perform_security_check(url)
            if not security_check['allowed']:
                return ToolResult(
                    status=ToolExecutionStatus.FAILED,
                    data=None,
                    error_message=f"Request blocked: {security_check['reason']}"
                )
            
            # Check cache if enabled and method is GET
            if enable_cache and method == 'GET':
                cache_key = self._generate_cache_key(url, headers)
                cached_response = self._get_cached_response(cache_key)
                if cached_response:
                    self._logger.debug(f"Returning cached response for {url}")
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    result = {
                        'request_id': request_id,
                        'from_cache': True,
                        'execution_time': execution_time,
                        **cached_response
                    }
                    
                    return ToolResult(
                        status=ToolExecutionStatus.SUCCESS,
                        data=result,
                        execution_time=execution_time
                    )
            
            # Prepare request
            request_data = await self._prepare_request(url, method, headers, body, auth)
            
            # Execute request with retries
            response_data = await self._execute_request_with_retries(
                request_data, timeout, max_retries
            )
            
            # Cache response if applicable
            if enable_cache and method == 'GET' and response_data.get('success', False):
                cache_key = self._generate_cache_key(url, headers)
                self._cache_response(cache_key, response_data)
            
            # Record request in history
            self._record_request_history(request_id, request_data, response_data)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            response_data['request_id'] = request_id
            response_data['from_cache'] = False
            response_data['execution_time'] = execution_time
            
            status = ToolExecutionStatus.SUCCESS if response_data.get('success', False) else ToolExecutionStatus.FAILED
            
            return ToolResult(
                status=status,
                data=response_data,
                execution_time=execution_time,
                metadata={
                    'request_id': request_id,
                    'method': method,
                    'url': url,
                    'status_code': response_data.get('status_code', 0)
                }
            )
            
        except Exception as e:
            self._logger.error(f"API request failed (ID: {request_id}): {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                status=ToolExecutionStatus.FAILED,
                data=None,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def get_parameters(self) -> List[ToolParameter]:
        """Return list of parameters this tool accepts."""
        return [
            ToolParameter(
                name="url",
                type="string",
                required=True,
                description="URL to make the request to"
            ),
            ToolParameter(
                name="method",
                type="string",
                required=False,
                description="HTTP method (GET, POST, PUT, DELETE, PATCH)",
                default_value="GET"
            ),
            ToolParameter(
                name="headers",
                type="object",
                required=False,
                description="HTTP headers as key-value pairs",
                default_value={}
            ),
            ToolParameter(
                name="body",
                type="string",
                required=False,
                description="Request body (for POST, PUT, PATCH)",
                default_value=None
            ),
            ToolParameter(
                name="auth",
                type="object",
                required=False,
                description="Authentication configuration",
                default_value=None
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                required=False,
                description="Request timeout in seconds",
                default_value=30
            ),
            ToolParameter(
                name="enable_cache",
                type="boolean",
                required=False,
                description="Enable response caching for GET requests",
                default_value=True
            ),
            ToolParameter(
                name="max_retries",
                type="integer",
                required=False,
                description="Maximum number of retry attempts",
                default_value=3
            ),
            ToolParameter(
                name="follow_redirects",
                type="boolean",
                required=False,
                description="Follow HTTP redirects",
                default_value=True
            )
        ]
    
    def get_description(self) -> str:
        """Return description of what this tool does."""
        return ("API Client Tool - Makes HTTP requests to external APIs with support for "
                "multiple methods, authentication, caching, retries, and comprehensive "
                "error handling. Supports REST APIs and other HTTP-based services.")
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate if parameters are correct for this tool."""
        # Check required parameters
        if 'url' not in parameters:
            return False
        
        url = parameters.get('url', '')
        if not isinstance(url, str) or not url.strip():
            return False
        
        # Validate URL format
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
        except Exception:
            return False
        
        # Validate method
        method = parameters.get('method', 'GET').upper()
        allowed_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
        if method not in allowed_methods:
            return False
        
        # Validate timeout
        timeout = parameters.get('timeout', self._default_timeout)
        if not isinstance(timeout, int) or timeout <= 0 or timeout > 300:  # Max 5 minutes
            return False
        
        return True
    
    def _perform_security_check(self, url: str) -> Dict[str, Any]:
        """
        Perform security checks on the URL.
        
        Args:
            url: URL to check
            
        Returns:
            Security check result
        """
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            
            # Check blocked hosts
            if self._blocked_hosts:
                for blocked_host in self._blocked_hosts:
                    if blocked_host.lower() in hostname.lower():
                        return {
                            'allowed': False,
                            'reason': f'Host {hostname} is blocked'
                        }
            
            # Check allowed hosts (if specified)
            if self._allowed_hosts:
                allowed = False
                for allowed_host in self._allowed_hosts:
                    if allowed_host.lower() in hostname.lower():
                        allowed = True
                        break
                
                if not allowed:
                    return {
                        'allowed': False,
                        'reason': f'Host {hostname} is not in allowed hosts list'
                    }
            
            # Check for localhost/private IP ranges (basic check)
            if hostname in ['localhost', '127.0.0.1', '0.0.0.0'] or hostname.startswith('192.168.') or hostname.startswith('10.'):
                return {
                    'allowed': False,
                    'reason': 'Local/private network access not allowed'
                }
            
            return {
                'allowed': True,
                'reason': 'Security checks passed'
            }
            
        except Exception as e:
            return {
                'allowed': False,
                'reason': f'Security check failed: {str(e)}'
            }
    
    async def _prepare_request(self, url: str, method: str, headers: Dict[str, Any], 
                             body: Optional[str], auth: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare request data.
        
        Args:
            url: Request URL
            method: HTTP method
            headers: Request headers
            body: Request body
            auth: Authentication configuration
            
        Returns:
            Prepared request data
        """
        # Combine default headers with provided headers
        combined_headers = self._default_headers.copy()
        combined_headers.update(headers or {})
        
        # Handle authentication
        if auth:
            auth_headers = await self._prepare_authentication(auth)
            combined_headers.update(auth_headers)
        
        # Prepare body
        prepared_body = None
        if body and method in ['POST', 'PUT', 'PATCH']:
            if isinstance(body, dict):
                prepared_body = json.dumps(body)
                combined_headers['Content-Type'] = 'application/json'
            else:
                prepared_body = str(body)
        
        return {
            'url': url,
            'method': method,
            'headers': combined_headers,
            'body': prepared_body
        }
    
    async def _prepare_authentication(self, auth: Dict[str, Any]) -> Dict[str, str]:
        """
        Prepare authentication headers.
        
        Args:
            auth: Authentication configuration
            
        Returns:
            Authentication headers
        """
        auth_headers = {}
        auth_type = auth.get('type', '').lower()
        
        if auth_type == 'basic':
            username = auth.get('username', '')
            password = auth.get('password', '')
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            auth_headers['Authorization'] = f'Basic {credentials}'
            
        elif auth_type == 'bearer':
            token = auth.get('token', '')
            auth_headers['Authorization'] = f'Bearer {token}'
            
        elif auth_type == 'api_key':
            key_name = auth.get('key_name', 'X-API-Key')
            key_value = auth.get('key_value', '')
            auth_headers[key_name] = key_value
            
        elif auth_type == 'custom':
            custom_headers = auth.get('headers', {})
            auth_headers.update(custom_headers)
        
        return auth_headers
    
    async def _execute_request_with_retries(self, request_data: Dict[str, Any], 
                                          timeout: int, max_retries: int) -> Dict[str, Any]:
        """
        Execute request with retry logic.
        
        Args:
            request_data: Prepared request data
            timeout: Request timeout
            max_retries: Maximum retry attempts
            
        Returns:
            Response data
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    # Wait before retry
                    wait_time = self._retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    await asyncio.sleep(min(wait_time, 30))  # Max 30 seconds
                    self._logger.info(f"Retrying request (attempt {attempt + 1}/{max_retries + 1})")
                
                response_data = await self._execute_single_request(request_data, timeout)
                
                # Check if we should retry based on status code
                status_code = response_data.get('status_code', 0)
                if attempt < max_retries and self._should_retry(status_code):
                    continue
                
                return response_data
                
            except Exception as e:
                last_error = e
                self._logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries:
                    break
        
        # All retries failed
        return {
            'success': False,
            'error': f'Request failed after {max_retries + 1} attempts: {str(last_error)}',
            'status_code': 0,
            'headers': {},
            'response_data': None
        }
    
    async def _execute_single_request(self, request_data: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """
        Execute a single HTTP request.
        
        Args:
            request_data: Request data
            timeout: Request timeout
            
        Returns:
            Response data
        """
        # Simulate HTTP request execution
        # In a real implementation, this would use aiohttp or similar
        
        url = request_data['url']
        method = request_data['method']
        
        # Simulate request delay
        await asyncio.sleep(0.1)
        
        # Simulate different responses based on URL patterns
        if 'api.github.com' in url:
            response_data = await self._simulate_github_api_response(method, url)
        elif 'jsonplaceholder.typicode.com' in url:
            response_data = await self._simulate_jsonplaceholder_response(method, url)
        elif 'httpbin.org' in url:
            response_data = await self._simulate_httpbin_response(method, url, request_data)
        else:
            response_data = await self._simulate_generic_response(method, url)
        
        return response_data
    
    async def _simulate_github_api_response(self, method: str, url: str) -> Dict[str, Any]:
        """Simulate GitHub API response."""
        if method == 'GET' and '/users/' in url:
            return {
                'success': True,
                'status_code': 200,
                'headers': {'content-type': 'application/json'},
                'response_data': {
                    'login': 'octocat',
                    'id': 1,
                    'name': 'The Octocat',
                    'company': '@github',
                    'blog': 'https://github.blog',
                    'public_repos': 8
                },
                'response_size': 256
            }
        else:
            return {
                'success': True,
                'status_code': 200,
                'headers': {'content-type': 'application/json'},
                'response_data': {'message': 'GitHub API response'},
                'response_size': 64
            }
    
    async def _simulate_jsonplaceholder_response(self, method: str, url: str) -> Dict[str, Any]:
        """Simulate JSONPlaceholder API response."""
        if method == 'GET' and '/posts' in url:
            return {
                'success': True,
                'status_code': 200,
                'headers': {'content-type': 'application/json'},
                'response_data': [
                    {
                        'userId': 1,
                        'id': 1,
                        'title': 'Sample Post',
                        'body': 'This is a sample post from JSONPlaceholder.'
                    }
                ],
                'response_size': 128
            }
        elif method == 'POST':
            return {
                'success': True,
                'status_code': 201,
                'headers': {'content-type': 'application/json'},
                'response_data': {
                    'id': 101,
                    'title': 'Created Post',
                    'body': 'Post created successfully'
                },
                'response_size': 96
            }
        else:
            return {
                'success': True,
                'status_code': 200,
                'headers': {'content-type': 'application/json'},
                'response_data': {'message': 'JSONPlaceholder response'},
                'response_size': 64
            }
    
    async def _simulate_httpbin_response(self, method: str, url: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate HTTPBin response."""
        return {
            'success': True,
            'status_code': 200,
            'headers': {'content-type': 'application/json'},
            'response_data': {
                'method': method,
                'url': url,
                'headers': request_data.get('headers', {}),
                'data': request_data.get('body', ''),
                'origin': '192.168.1.1'
            },
            'response_size': 512
        }
    
    async def _simulate_generic_response(self, method: str, url: str) -> Dict[str, Any]:
        """Simulate generic API response."""
        return {
            'success': True,
            'status_code': 200,
            'headers': {'content-type': 'application/json'},
            'response_data': {
                'message': 'Success',
                'method': method,
                'url': url,
                'timestamp': datetime.now().isoformat()
            },
            'response_size': 128
        }
    
    def _should_retry(self, status_code: int) -> bool:
        """Determine if request should be retried based on status code."""
        # Retry on server errors and specific client errors
        retry_codes = [429, 500, 502, 503, 504]
        return status_code in retry_codes
    
    def _generate_cache_key(self, url: str, headers: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        # Simple cache key based on URL and relevant headers
        relevant_headers = {k: v for k, v in headers.items() if k.lower() in ['authorization', 'accept']}
        cache_data = {'url': url, 'headers': relevant_headers}
        return str(hash(json.dumps(cache_data, sort_keys=True)))
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        if cache_key not in self._response_cache:
            return None
        
        cached_data = self._response_cache[cache_key]
        cache_time = cached_data.get('cache_time', 0)
        
        # Check if cache is expired
        if time.time() - cache_time > self._cache_ttl_seconds:
            del self._response_cache[cache_key]
            return None
        
        return cached_data.get('response')
    
    def _cache_response(self, cache_key: str, response_data: Dict[str, Any]):
        """Cache response data."""
        self._response_cache[cache_key] = {
            'response': response_data,
            'cache_time': time.time()
        }
        
        # Simple cache size management
        if len(self._response_cache) > 100:  # Max 100 cached responses
            # Remove oldest cache entry
            oldest_key = min(self._response_cache.keys(), 
                           key=lambda k: self._response_cache[k]['cache_time'])
            del self._response_cache[oldest_key]
    
    def _record_request_history(self, request_id: str, request_data: Dict[str, Any], 
                              response_data: Dict[str, Any]):
        """Record request in history."""
        history_entry = {
            'request_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'method': request_data['method'],
            'url': request_data['url'],
            'status_code': response_data.get('status_code', 0),
            'success': response_data.get('success', False),
            'response_size': response_data.get('response_size', 0)
        }
        
        self._request_history.append(history_entry)
        
        # Maintain history size limit
        if len(self._request_history) > self._max_history_size:
            self._request_history.pop(0)
    
    def get_request_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent request history."""
        return self._request_history[-limit:] if self._request_history else []
    
    def clear_cache(self):
        """Clear response cache."""
        self._response_cache.clear()
        self._logger.info("Response cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._response_cache),
            'cache_ttl_seconds': self._cache_ttl_seconds,
            'max_cache_size': 100
        }
    
    async def initialize(self) -> bool:
        """Initialize the API client tool."""
        try:
            self._logger.info(f"Initializing APIClientTool {self.tool_id}")
            
            # Validate configuration
            if self._default_timeout <= 0:
                self._default_timeout = 30
            
            if self._max_retries < 0:
                self._max_retries = 3
            
            if self._cache_ttl_seconds <= 0:
                self._cache_ttl_seconds = 300
            
            self.is_available = True
            self._logger.info(f"APIClientTool {self.tool_id} initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize APIClientTool {self.tool_id}: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup API client tool resources."""
        try:
            self._logger.info(f"Cleaning up APIClientTool {self.tool_id}")
            
            # Clear caches and history
            self._response_cache.clear()
            self._request_history.clear()
            
            self.is_available = False
            self._logger.info(f"APIClientTool {self.tool_id} cleanup completed")
            return True
            
        except Exception as e:
            self._logger.error(f"Error during APIClientTool cleanup: {e}")
            return False