"""
OpenAI Adapter - Adapter for integrating with OpenAI's API services.
Provides unified interface for OpenAI models including GPT, embeddings, and other services.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from datetime import datetime
import uuid

# Note: In a real implementation, you would install and import the OpenAI library
# from openai import AsyncOpenAI


class OpenAIAdapter:
    """
    Adapter for OpenAI API services with support for chat completions, embeddings, and more.
    
    Features:
    - Chat completions (GPT models)
    - Text embeddings
    - Function calling
    - Streaming responses
    - Token usage tracking
    - Rate limiting
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._logger = logging.getLogger(__name__)
        self._api_key = config.get('api_key', '')
        self._base_url = config.get('base_url', 'https://api.openai.com/v1')
        self._organization = config.get('organization', None)
        self._project = config.get('project', None)
        self._default_model = config.get('default_model', 'gpt-3.5-turbo')
        self._default_embedding_model = config.get('default_embedding_model', 'text-embedding-ada-002')
        self._max_retries = config.get('max_retries', 3)
        self._timeout = config.get('timeout_seconds', 60)
        
        # Usage tracking
        self._usage_stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'requests_by_model': {},
            'errors': 0
        }
        
        # Rate limiting
        self._request_timestamps = []
        self._max_requests_per_minute = config.get('max_requests_per_minute', 60)
        
        # Model pricing (approximate, in USD per 1K tokens)
        self._model_pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-32k': {'input': 0.06, 'output': 0.12},
            'gpt-3.5-turbo': {'input': 0.0010, 'output': 0.0020},
            'gpt-3.5-turbo-16k': {'input': 0.003, 'output': 0.004},
            'text-embedding-ada-002': {'input': 0.0001, 'output': 0.0}
        }
        
        # Initialize client (simulated)
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            # In a real implementation:
            # self._client = AsyncOpenAI(
            #     api_key=self._api_key,
            #     base_url=self._base_url,
            #     organization=self._organization,
            #     project=self._project,
            #     max_retries=self._max_retries,
            #     timeout=self._timeout
            # )
            
            # Simulated client for demo
            self._client = type('MockClient', (), {})()
            self._logger.info("OpenAI adapter initialized (simulated)")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    async def chat_completion(self, messages: List[Dict[str, str]], 
                            model: Optional[str] = None,
                            temperature: float = 0.7,
                            max_tokens: Optional[int] = None,
                            functions: Optional[List[Dict[str, Any]]] = None,
                            function_call: Optional[Union[str, Dict[str, str]]] = None,
                            stream: bool = False,
                            **kwargs) -> Dict[str, Any]:
        """
        Create a chat completion using OpenAI's API.
        
        Args:
            messages: List of message dictionaries
            model: Model to use (defaults to configured default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            functions: Function definitions for function calling
            function_call: Function call configuration
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            Chat completion response
        """
        try:
            await self._check_rate_limit()
            
            model = model or self._default_model
            request_id = str(uuid.uuid4())
            
            self._logger.info(f"Creating chat completion with {model} (ID: {request_id})")
            
            # Prepare request parameters
            request_params = {
                'model': model,
                'messages': messages,
                'temperature': temperature,
                'max_tokens': max_tokens,
                **kwargs
            }
            
            if functions:
                request_params['functions'] = functions
            if function_call:
                request_params['function_call'] = function_call
            
            # Simulate API call
            start_time = datetime.now()
            
            if stream:
                return await self._simulate_streaming_completion(request_params, request_id)
            else:
                response = await self._simulate_completion(request_params, request_id)
                
                # Track usage
                await self._track_usage(model, response)
                
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                
                self._logger.info(f"Chat completion completed in {response_time:.2f}s (ID: {request_id})")
                
                return {
                    'id': request_id,
                    'response': response,
                    'model': model,
                    'response_time': response_time,
                    'usage': response.get('usage', {}),
                    'timestamp': start_time.isoformat()
                }
                
        except Exception as e:
            self._usage_stats['errors'] += 1
            self._logger.error(f"Chat completion failed: {e}")
            raise
    
    async def create_embedding(self, text: Union[str, List[str]], 
                             model: Optional[str] = None,
                             **kwargs) -> Dict[str, Any]:
        """
        Create embeddings for text input.
        
        Args:
            text: Text or list of texts to embed
            model: Embedding model to use
            **kwargs: Additional parameters
            
        Returns:
            Embedding response
        """
        try:
            await self._check_rate_limit()
            
            model = model or self._default_embedding_model
            request_id = str(uuid.uuid4())
            
            # Convert single text to list
            texts = [text] if isinstance(text, str) else text
            
            self._logger.info(f"Creating embeddings for {len(texts)} text(s) with {model} (ID: {request_id})")
            
            start_time = datetime.now()
            
            # Simulate embedding generation
            embeddings = []
            for i, input_text in enumerate(texts):
                embedding = await self._simulate_embedding(input_text, model)
                embeddings.append({
                    'object': 'embedding',
                    'index': i,
                    'embedding': embedding
                })
            
            # Simulate usage data
            total_tokens = sum(len(t.split()) * 1.3 for t in texts)  # Rough token estimate
            usage = {
                'prompt_tokens': int(total_tokens),
                'total_tokens': int(total_tokens)
            }
            
            response = {
                'object': 'list',
                'data': embeddings,
                'model': model,
                'usage': usage
            }
            
            # Track usage
            await self._track_usage(model, response)
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            self._logger.info(f"Embeddings created in {response_time:.2f}s (ID: {request_id})")
            
            return {
                'id': request_id,
                'response': response,
                'model': model,
                'response_time': response_time,
                'text_count': len(texts),
                'timestamp': start_time.isoformat()
            }
            
        except Exception as e:
            self._usage_stats['errors'] += 1
            self._logger.error(f"Embedding creation failed: {e}")
            raise
    
    async def list_models(self) -> Dict[str, Any]:
        """
        List available models.
        
        Returns:
            List of available models
        """
        try:
            # Simulate model listing
            models = [
                {
                    'id': 'gpt-4',
                    'object': 'model',
                    'created': 1687882411,
                    'owned_by': 'openai',
                    'capabilities': ['chat', 'completion']
                },
                {
                    'id': 'gpt-4-32k',
                    'object': 'model',
                    'created': 1687882411,
                    'owned_by': 'openai',
                    'capabilities': ['chat', 'completion']
                },
                {
                    'id': 'gpt-3.5-turbo',
                    'object': 'model',
                    'created': 1677610602,
                    'owned_by': 'openai',
                    'capabilities': ['chat', 'completion']
                },
                {
                    'id': 'gpt-3.5-turbo-16k',
                    'object': 'model',
                    'created': 1683758102,
                    'owned_by': 'openai',
                    'capabilities': ['chat', 'completion']
                },
                {
                    'id': 'text-embedding-ada-002',
                    'object': 'model',
                    'created': 1671217299,
                    'owned_by': 'openai',
                    'capabilities': ['embedding']
                }
            ]
            
            return {
                'object': 'list',
                'data': models
            }
            
        except Exception as e:
            self._logger.error(f"Failed to list models: {e}")
            raise
    
    async def _simulate_completion(self, request_params: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Simulate chat completion response."""
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        messages = request_params.get('messages', [])
        model = request_params.get('model', 'gpt-3.5-turbo')
        functions = request_params.get('functions', [])
        
        # Generate simulated response based on last message
        last_message = messages[-1] if messages else {'content': ''}
        user_content = last_message.get('content', '')
        
        # Check if this might be a function call
        if functions and any(func_name in user_content.lower() for func in functions for func_name in [func.get('name', '').lower()]):
            # Simulate function call response
            function_name = functions[0].get('name', 'unknown_function')
            response_content = None
            function_call = {
                'name': function_name,
                'arguments': json.dumps({'query': user_content[:100]})
            }
        else:
            # Simulate regular chat response
            if 'hello' in user_content.lower():
                response_content = "Hello! How can I assist you today?"
            elif 'weather' in user_content.lower():
                response_content = "I'd be happy to help with weather information. Could you please specify a location?"
            elif 'code' in user_content.lower() or 'programming' in user_content.lower():
                response_content = "I can help you with programming questions. What language or concept would you like to explore?"
            elif 'analysis' in user_content.lower() or 'analyze' in user_content.lower():
                response_content = "I can help analyze data, text, or concepts. Please provide more details about what you'd like me to analyze."
            else:
                response_content = f"I understand you're asking about: {user_content[:50]}{'...' if len(user_content) > 50 else ''}. How can I help you with this?"
            
            function_call = None
        
        # Simulate token usage
        input_tokens = sum(len(msg.get('content', '').split()) * 1.3 for msg in messages)
        output_tokens = len(response_content.split()) * 1.3 if response_content else 20
        
        choice = {
            'index': 0,
            'message': {
                'role': 'assistant',
                'content': response_content
            },
            'finish_reason': 'stop'
        }
        
        if function_call:
            choice['message']['function_call'] = function_call
            choice['finish_reason'] = 'function_call'
        
        return {
            'id': f'chatcmpl-{request_id[:8]}',
            'object': 'chat.completion',
            'created': int(datetime.now().timestamp()),
            'model': model,
            'choices': [choice],
            'usage': {
                'prompt_tokens': int(input_tokens),
                'completion_tokens': int(output_tokens),
                'total_tokens': int(input_tokens + output_tokens)
            }
        }
    
    async def _simulate_streaming_completion(self, request_params: Dict[str, Any], request_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Simulate streaming chat completion response."""
        model = request_params.get('model', 'gpt-3.5-turbo')
        
        # Simulate streaming response
        response_text = "This is a simulated streaming response from OpenAI. Each chunk arrives progressively."
        words = response_text.split()
        
        for i, word in enumerate(words):
            chunk = {
                'id': f'chatcmpl-{request_id[:8]}',
                'object': 'chat.completion.chunk',
                'created': int(datetime.now().timestamp()),
                'model': model,
                'choices': [{
                    'index': 0,
                    'delta': {
                        'content': word + ' ' if i < len(words) - 1 else word
                    },
                    'finish_reason': None
                }]
            }
            
            yield chunk
            await asyncio.sleep(0.1)  # Simulate streaming delay
        
        # Final chunk
        final_chunk = {
            'id': f'chatcmpl-{request_id[:8]}',
            'object': 'chat.completion.chunk',
            'created': int(datetime.now().timestamp()),
            'model': model,
            'choices': [{
                'index': 0,
                'delta': {},
                'finish_reason': 'stop'
            }]
        }
        
        yield final_chunk
    
    async def _simulate_embedding(self, text: str, model: str) -> List[float]:
        """Simulate embedding generation."""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Generate deterministic embedding based on text
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to embedding-like vector
        embedding_dim = 1536  # Standard dimension for Ada-002
        embedding = []
        
        for i in range(embedding_dim):
            # Use hash and position to generate consistent values
            seed = int(text_hash[i % len(text_hash)], 16) + i
            value = (seed % 1000 - 500) / 500.0  # Normalize to [-1, 1]
            embedding.append(value)
        
        # Normalize the embedding
        import math
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = datetime.now().timestamp()
        
        # Remove timestamps older than 1 minute
        self._request_timestamps = [
            ts for ts in self._request_timestamps 
            if current_time - ts < 60
        ]
        
        if len(self._request_timestamps) >= self._max_requests_per_minute:
            wait_time = 60 - (current_time - self._request_timestamps[0])
            if wait_time > 0:
                self._logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        self._request_timestamps.append(current_time)
    
    async def _track_usage(self, model: str, response: Dict[str, Any]):
        """Track API usage and costs."""
        usage = response.get('usage', {})
        
        if usage:
            self._usage_stats['total_requests'] += 1
            self._usage_stats['total_tokens'] += usage.get('total_tokens', 0)
            
            # Track by model
            if model not in self._usage_stats['requests_by_model']:
                self._usage_stats['requests_by_model'][model] = {
                    'requests': 0,
                    'tokens': 0,
                    'cost': 0.0
                }
            
            model_stats = self._usage_stats['requests_by_model'][model]
            model_stats['requests'] += 1
            model_stats['tokens'] += usage.get('total_tokens', 0)
            
            # Calculate cost
            if model in self._model_pricing:
                pricing = self._model_pricing[model]
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                
                cost = (input_tokens * pricing['input'] + output_tokens * pricing['output']) / 1000
                model_stats['cost'] += cost
                self._usage_stats['total_cost'] += cost
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return self._usage_stats.copy()
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models."""
        return list(self._model_pricing.keys())
    
    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int = 0) -> float:
        """
        Estimate cost for a request.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        if model not in self._model_pricing:
            return 0.0
        
        pricing = self._model_pricing[model]
        return (input_tokens * pricing['input'] + output_tokens * pricing['output']) / 1000
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the OpenAI service.
        
        Returns:
            Health check results
        """
        try:
            # Simulate health check by listing models
            models = await self.list_models()
            
            return {
                'status': 'healthy',
                'service': 'OpenAI API',
                'models_available': len(models.get('data', [])),
                'rate_limit_remaining': self._max_requests_per_minute - len(self._request_timestamps),
                'total_requests': self._usage_stats['total_requests'],
                'error_rate': self._usage_stats['errors'] / max(self._usage_stats['total_requests'], 1),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'service': 'OpenAI API',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self._usage_stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'requests_by_model': {},
            'errors': 0
        }
        self._logger.info("Usage statistics reset")
    
    async def cleanup(self):
        """Cleanup adapter resources."""
        try:
            # Close client connections if needed
            if hasattr(self._client, 'close'):
                await self._client.close()
            
            self._logger.info("OpenAI adapter cleaned up")
            
        except Exception as e:
            self._logger.error(f"Error during OpenAI adapter cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            # Note: In async context, proper cleanup should be done explicitly
            pass
        except Exception:
            pass