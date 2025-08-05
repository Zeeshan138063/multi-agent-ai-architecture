"""
Web Search Tool - Tool for performing web searches and retrieving information.
Demonstrates tool interface implementation with web search capabilities.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re

from ...core.interfaces import ToolInterface, ToolResult, ToolParameter, ToolExecutionStatus


class WebSearchTool(ToolInterface):
    """
    Tool for performing web searches and information retrieval.
    
    Capabilities:
    - Web search queries
    - Result filtering and ranking
    - Content extraction
    - Search result caching
    """
    
    def __init__(self, tool_id: str, config: Dict[str, Any]):
        super().__init__(tool_id, config)
        self._logger = logging.getLogger(__name__)
        self._search_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl_seconds = config.get('cache_ttl_seconds', 3600)  # 1 hour
        self._max_results = config.get('max_results', 10)
        self._search_engines = config.get('search_engines', ['google', 'bing', 'duckduckgo'])
        self._enable_content_extraction = config.get('enable_content_extraction', True)
        
        # Simulated search results for demonstration
        self._demo_results = self._load_demo_results()
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute web search with given parameters.
        
        Args:
            parameters: Search parameters including query, filters, etc.
            
        Returns:
            ToolResult with search results
        """
        start_time = datetime.now()
        
        try:
            # Validate parameters
            if not self.validate_parameters(parameters):
                return ToolResult(
                    status=ToolExecutionStatus.INVALID_INPUT,
                    data=None,
                    error_message="Invalid parameters provided"
                )
            
            query = parameters.get('query', '')
            search_engine = parameters.get('search_engine', 'google')
            max_results = min(parameters.get('max_results', self._max_results), self._max_results)
            include_snippets = parameters.get('include_snippets', True)
            filter_domain = parameters.get('filter_domain', None)
            
            self._logger.info(f"Executing web search: '{query}' using {search_engine}")
            
            # Check cache first
            cache_key = self._generate_cache_key(query, search_engine, max_results)
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result:
                self._logger.debug(f"Returning cached result for query: {query}")
                execution_time = (datetime.now() - start_time).total_seconds()
                cached_result['execution_time'] = execution_time
                cached_result['from_cache'] = True
                
                return ToolResult(
                    status=ToolExecutionStatus.SUCCESS,
                    data=cached_result,
                    execution_time=execution_time
                )
            
            # Perform search
            search_results = await self._perform_search(
                query, search_engine, max_results, include_snippets, filter_domain
            )
            
            # Cache the results
            self._cache_result(cache_key, search_results)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            search_results['execution_time'] = execution_time
            search_results['from_cache'] = False
            
            return ToolResult(
                status=ToolExecutionStatus.SUCCESS,
                data=search_results,
                execution_time=execution_time,
                metadata={
                    'query': query,
                    'search_engine': search_engine,
                    'results_count': len(search_results.get('results', []))
                }
            )
            
        except Exception as e:
            self._logger.error(f"Web search execution failed: {e}")
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
                name="query",
                type="string",
                required=True,
                description="Search query string"
            ),
            ToolParameter(
                name="search_engine",
                type="string",
                required=False,
                description="Search engine to use (google, bing, duckduckgo)",
                default_value="google"
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                required=False,
                description="Maximum number of results to return",
                default_value=10
            ),
            ToolParameter(
                name="include_snippets",
                type="boolean",
                required=False,
                description="Include content snippets in results",
                default_value=True
            ),
            ToolParameter(
                name="filter_domain",
                type="string",
                required=False,
                description="Filter results to specific domain",
                default_value=None
            ),
            ToolParameter(
                name="safe_search",
                type="boolean",
                required=False,
                description="Enable safe search filtering",
                default_value=True
            ),
            ToolParameter(
                name="language",
                type="string",
                required=False,
                description="Language preference for results",
                default_value="en"
            )
        ]
    
    def get_description(self) -> str:
        """Return description of what this tool does."""
        return ("Web Search Tool - Performs web searches across multiple search engines "
                "and returns ranked results with content snippets and metadata. "
                "Supports result filtering, caching, and content extraction.")
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate if parameters are correct for this tool."""
        # Check required parameters
        if 'query' not in parameters:
            return False
        
        query = parameters.get('query', '')
        if not isinstance(query, str) or len(query.strip()) == 0:
            return False
        
        # Validate optional parameters
        search_engine = parameters.get('search_engine', 'google')
        if search_engine not in self._search_engines:
            return False
        
        max_results = parameters.get('max_results', 10)
        if not isinstance(max_results, int) or max_results <= 0 or max_results > 100:
            return False
        
        return True
    
    async def _perform_search(self, query: str, search_engine: str, max_results: int,
                            include_snippets: bool, filter_domain: Optional[str]) -> Dict[str, Any]:
        """
        Perform the actual web search.
        
        Args:
            query: Search query
            search_engine: Search engine to use
            max_results: Maximum results to return
            include_snippets: Whether to include content snippets
            filter_domain: Domain filter if specified
            
        Returns:
            Search results dictionary
        """
        # Simulate search delay
        await asyncio.sleep(0.5)
        
        # Get demo results based on query
        demo_results = self._get_demo_results_for_query(query)
        
        # Apply domain filter if specified
        if filter_domain:
            demo_results = [r for r in demo_results if filter_domain.lower() in r['url'].lower()]
        
        # Limit results
        demo_results = demo_results[:max_results]
        
        # Process results based on preferences
        processed_results = []
        for i, result in enumerate(demo_results):
            processed_result = {
                'rank': i + 1,
                'title': result['title'],
                'url': result['url'],
                'domain': self._extract_domain(result['url']),
                'relevance_score': result.get('relevance_score', 0.8)
            }
            
            if include_snippets:
                processed_result['snippet'] = result.get('snippet', '')
                processed_result['content_preview'] = result.get('content_preview', '')
            
            # Add metadata
            processed_result['metadata'] = {
                'search_engine': search_engine,
                'indexed_date': result.get('indexed_date', datetime.now().isoformat()),
                'content_type': result.get('content_type', 'text/html'),
                'language': result.get('language', 'en')
            }
            
            processed_results.append(processed_result)
        
        # Calculate search statistics
        search_stats = {
            'total_results_found': len(processed_results),
            'search_time_ms': 500,  # Simulated search time
            'search_engine_used': search_engine,
            'query_processed': query,
            'results_filtered': filter_domain is not None
        }
        
        return {
            'query': query,
            'results': processed_results,
            'search_stats': search_stats,
            'suggestions': self._generate_search_suggestions(query),
            'related_queries': self._generate_related_queries(query)
        }
    
    def _get_demo_results_for_query(self, query: str) -> List[Dict[str, Any]]:
        """Get demo results based on query keywords."""
        query_lower = query.lower()
        
        # Return different demo results based on query content
        if any(keyword in query_lower for keyword in ['python', 'programming', 'code']):
            return self._demo_results['programming']
        elif any(keyword in query_lower for keyword in ['ai', 'machine learning', 'artificial']):
            return self._demo_results['ai']
        elif any(keyword in query_lower for keyword in ['weather', 'climate', 'temperature']):
            return self._demo_results['weather']
        elif any(keyword in query_lower for keyword in ['news', 'current events', 'today']):
            return self._demo_results['news']
        else:
            return self._demo_results['general']
    
    def _load_demo_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load demo search results for different categories."""
        return {
            'programming': [
                {
                    'title': 'Python Official Documentation',
                    'url': 'https://docs.python.org/',
                    'snippet': 'Python is a programming language that lets you work quickly and integrate systems more effectively.',
                    'content_preview': 'Welcome to Python.org. Python is a programming language...',
                    'relevance_score': 0.95,
                    'content_type': 'text/html',
                    'language': 'en'
                },
                {
                    'title': 'Learn Python Programming - Codecademy',
                    'url': 'https://www.codecademy.com/learn/learn-python',
                    'snippet': 'Learn Python, a powerful language used by sites like YouTube and Dropbox.',
                    'content_preview': 'Start your Python journey with interactive lessons...',
                    'relevance_score': 0.88,
                    'content_type': 'text/html',
                    'language': 'en'
                },
                {
                    'title': 'Python Tutorial - W3Schools',
                    'url': 'https://www.w3schools.com/python/',
                    'snippet': 'Well organized and easy to understand Web building tutorials with lots of examples.',
                    'content_preview': 'Python is a popular programming language...',
                    'relevance_score': 0.82,
                    'content_type': 'text/html',
                    'language': 'en'
                }
            ],
            'ai': [
                {
                    'title': 'What is Artificial Intelligence (AI)? | IBM',
                    'url': 'https://www.ibm.com/cloud/learn/what-is-artificial-intelligence',
                    'snippet': 'Artificial intelligence leverages computers and machines to mimic problem-solving and decision-making.',
                    'content_preview': 'Artificial intelligence (AI) refers to the simulation...',
                    'relevance_score': 0.93,
                    'content_type': 'text/html',
                    'language': 'en'
                },
                {
                    'title': 'Machine Learning | Coursera',
                    'url': 'https://www.coursera.org/learn/machine-learning',
                    'snippet': 'Learn Machine Learning online with courses like Machine Learning and Deep Learning.',
                    'content_preview': 'Machine learning is the science of getting computers...',
                    'relevance_score': 0.89,
                    'content_type': 'text/html',
                    'language': 'en'
                },
                {
                    'title': 'AI News - The latest artificial intelligence news',
                    'url': 'https://www.artificialintelligence-news.com/',
                    'snippet': 'The latest artificial intelligence news, AI research and reports.',
                    'content_preview': 'Stay updated with the latest developments in AI...',
                    'relevance_score': 0.85,
                    'content_type': 'text/html',
                    'language': 'en'
                }
            ],
            'weather': [
                {
                    'title': 'Weather.com - Weather Forecast & Reports',
                    'url': 'https://weather.com/',
                    'snippet': 'Get the latest weather forecast, weather radar and weather reports.',
                    'content_preview': 'Get current weather conditions and forecasts...',
                    'relevance_score': 0.91,
                    'content_type': 'text/html',
                    'language': 'en'
                },
                {
                    'title': 'AccuWeather - Weather Forecasts',
                    'url': 'https://www.accuweather.com/',
                    'snippet': 'Get AccuWeather alerts as they happen with our browser notifications.',
                    'content_preview': 'Superior accuracy with detailed local weather...',
                    'relevance_score': 0.87,
                    'content_type': 'text/html',
                    'language': 'en'
                }
            ],
            'news': [
                {
                    'title': 'CNN - Breaking News, Latest News and Videos',
                    'url': 'https://www.cnn.com/',
                    'snippet': 'View the latest news and breaking news today for U.S., world, weather, entertainment.',
                    'content_preview': 'Breaking news and the latest headlines...',
                    'relevance_score': 0.90,
                    'content_type': 'text/html',
                    'language': 'en'
                },
                {
                    'title': 'BBC News - Home',
                    'url': 'https://www.bbc.com/news',
                    'snippet': 'Visit BBC News for up-to-the-minute news, breaking news, video, audio and feature stories.',
                    'content_preview': 'The latest global news and analysis...',
                    'relevance_score': 0.88,
                    'content_type': 'text/html',
                    'language': 'en'
                }
            ],
            'general': [
                {
                    'title': 'Wikipedia - The Free Encyclopedia',
                    'url': 'https://en.wikipedia.org/',
                    'snippet': 'Wikipedia is a free online encyclopedia with millions of articles.',
                    'content_preview': 'Wikipedia is a multilingual online encyclopedia...',
                    'relevance_score': 0.75,
                    'content_type': 'text/html',
                    'language': 'en'
                },
                {
                    'title': 'Google Search',
                    'url': 'https://www.google.com/',
                    'snippet': 'Search the world\'s information, including webpages, images, videos and more.',
                    'content_preview': 'Google offers search capabilities...',
                    'relevance_score': 0.70,
                    'content_type': 'text/html',
                    'language': 'en'
                }
            ]
        }
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        import re
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        return match.group(1) if match else url
    
    def _generate_search_suggestions(self, query: str) -> List[str]:
        """Generate search suggestions based on query."""
        suggestions = []
        query_words = query.lower().split()
        
        # Simple suggestion generation based on common patterns
        if 'python' in query_words:
            suggestions.extend(['python tutorial', 'python examples', 'python documentation'])
        elif 'ai' in query_words or 'artificial intelligence' in query.lower():
            suggestions.extend(['machine learning', 'deep learning', 'neural networks'])
        elif 'weather' in query_words:
            suggestions.extend(['weather forecast', 'weather radar', 'weather alerts'])
        else:
            suggestions.extend([f'{query} tutorial', f'{query} guide', f'{query} examples'])
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _generate_related_queries(self, query: str) -> List[str]:
        """Generate related queries based on the search query."""
        related = []
        query_lower = query.lower()
        
        if 'python' in query_lower:
            related.extend(['javascript programming', 'java tutorials', 'web development'])
        elif 'ai' in query_lower:
            related.extend(['data science', 'robotics', 'computer vision'])
        elif 'weather' in query_lower:
            related.extend(['climate change', 'meteorology', 'weather patterns'])
        else:
            # Generic related queries
            related.extend([f'how to {query}', f'{query} tips', f'best {query}'])
        
        return related[:3]  # Return top 3 related queries
    
    def _generate_cache_key(self, query: str, search_engine: str, max_results: int) -> str:
        """Generate cache key for search results."""
        return f"{search_engine}:{query.lower()}:{max_results}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached search result if available and not expired."""
        if cache_key not in self._search_cache:
            return None
        
        cached_data = self._search_cache[cache_key]
        cache_time = cached_data.get('cache_time', 0)
        
        # Check if cache is expired
        if datetime.now().timestamp() - cache_time > self._cache_ttl_seconds:
            del self._search_cache[cache_key]
            return None
        
        return cached_data.get('data')
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache search result."""
        self._search_cache[cache_key] = {
            'data': result,
            'cache_time': datetime.now().timestamp()
        }
        
        # Simple cache size management
        if len(self._search_cache) > 100:  # Max 100 cached searches
            # Remove oldest cache entry
            oldest_key = min(self._search_cache.keys(), 
                           key=lambda k: self._search_cache[k]['cache_time'])
            del self._search_cache[oldest_key]
    
    async def initialize(self) -> bool:
        """Initialize the web search tool."""
        try:
            self._logger.info(f"Initializing WebSearchTool {self.tool_id}")
            
            # Validate configuration
            if self._max_results <= 0:
                self._max_results = 10
            
            if self._cache_ttl_seconds <= 0:
                self._cache_ttl_seconds = 3600
            
            self.is_available = True
            self._logger.info(f"WebSearchTool {self.tool_id} initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize WebSearchTool {self.tool_id}: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup web search tool resources."""
        try:
            self._logger.info(f"Cleaning up WebSearchTool {self.tool_id}")
            
            # Clear cache
            self._search_cache.clear()
            
            self.is_available = False
            self._logger.info(f"WebSearchTool {self.tool_id} cleanup completed")
            return True
            
        except Exception as e:
            self._logger.error(f"Error during WebSearchTool cleanup: {e}")
            return False