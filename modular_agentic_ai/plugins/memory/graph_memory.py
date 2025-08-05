"""
Graph Memory - Memory system using graph structure for relationship-based storage and retrieval.
Provides graph-based storage with nodes, edges, and relationship queries.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
import uuid
from dataclasses import dataclass
from enum import Enum

from ...core.interfaces import MemoryInterface, MemoryEntry, MemoryQuery, MemorySearchResult


class NodeType(Enum):
    """Types of nodes in the graph."""
    MEMORY = "memory"
    CONCEPT = "concept"
    ENTITY = "entity"
    EVENT = "event"
    RELATIONSHIP = "relationship"


class EdgeType(Enum):
    """Types of edges in the graph."""
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    CAUSED_BY = "caused_by"
    LEADS_TO = "leads_to"
    SIMILAR_TO = "similar_to"
    OPPOSITE_TO = "opposite_to"
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"


@dataclass
class GraphNode:
    """Node in the memory graph."""
    id: str
    node_type: NodeType
    data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    properties: Dict[str, Any]


@dataclass
class GraphEdge:
    """Edge in the memory graph."""
    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float
    properties: Dict[str, Any]
    created_at: datetime


class GraphMemory(MemoryInterface):
    """
    Graph-based memory system for storing relationships and hierarchical data.
    
    Features:
    - Node and edge storage
    - Relationship queries
    - Graph traversal
    - Subgraph extraction
    - Graph analytics
    """
    
    def __init__(self, memory_id: str, config: Dict[str, Any]):
        super().__init__(memory_id, config)
        self._logger = logging.getLogger(__name__)
        self._max_nodes = config.get('max_nodes', 10000)
        self._max_edges = config.get('max_edges', 50000)
        self._enable_indexing = config.get('enable_indexing', True)
        self._auto_create_relationships = config.get('auto_create_relationships', True)
        self._relationship_threshold = config.get('relationship_threshold', 0.7)
        
        # Graph storage
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, GraphEdge] = {}
        self._adjacency_list: Dict[str, List[str]] = {}  # node_id -> [edge_ids]
        self._reverse_adjacency_list: Dict[str, List[str]] = {}  # node_id -> [edge_ids]
        
        # Indexes for efficient querying
        self._node_type_index: Dict[NodeType, Set[str]] = {}
        self._edge_type_index: Dict[EdgeType, Set[str]] = {}
        self._property_index: Dict[str, Dict[Any, Set[str]]] = {}
        
        # Memory entry mapping
        self._memory_to_node: Dict[str, str] = {}  # memory_entry_id -> node_id
        self._node_to_memory: Dict[str, str] = {}  # node_id -> memory_entry_id
        
        # Analytics
        self._query_history: List[Dict[str, Any]] = []
        self._graph_metrics: Dict[str, Any] = {}
    
    async def store(self, entry: MemoryEntry) -> bool:
        """
        Store a memory entry as a graph node with potential relationships.
        
        Args:
            entry: Memory entry to store
            
        Returns:
            True if storage successful
        """
        try:
            self._logger.debug(f"Storing memory entry as graph node: {entry.id}")
            
            # Check capacity
            if len(self._nodes) >= self._max_nodes:
                await self._cleanup_old_nodes()
            
            # Create graph node for memory entry
            node_id = f"memory_{entry.id}"
            node = GraphNode(
                id=node_id,
                node_type=NodeType.MEMORY,
                data={
                    'content': entry.content,
                    'tags': entry.tags,
                    'importance_score': entry.importance_score
                },
                created_at=entry.timestamp,
                updated_at=entry.timestamp,
                properties=entry.metadata.copy()
            )
            
            # Store node
            await self._add_node(node)
            
            # Map memory entry to node
            self._memory_to_node[entry.id] = node_id
            self._node_to_memory[node_id] = entry.id
            
            # Create relationships with existing nodes
            if self._auto_create_relationships:
                await self._create_automatic_relationships(node)
            
            # Extract and create concept nodes
            await self._extract_concepts(node)
            
            self._logger.debug(f"Successfully stored memory entry as node: {node_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Error storing memory entry {entry.id}: {e}")
            return False
    
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a memory entry by ID.
        
        Args:
            entry_id: ID of the memory entry
            
        Returns:
            Memory entry if found
        """
        try:
            node_id = self._memory_to_node.get(entry_id)
            if not node_id:
                return None
            
            node = self._nodes.get(node_id)
            if not node:
                return None
            
            # Convert node back to memory entry
            return MemoryEntry(
                id=entry_id,
                content=node.data.get('content', ''),
                timestamp=node.created_at,
                tags=node.data.get('tags', []),
                metadata=node.properties.copy(),
                importance_score=node.data.get('importance_score', 0.5)
            )
            
        except Exception as e:
            self._logger.error(f"Error retrieving memory entry {entry_id}: {e}")
            return None
    
    async def search(self, query: MemoryQuery) -> MemorySearchResult:
        """
        Search memory entries using graph queries and traversal.
        
        Args:
            query: Search query with parameters
            
        Returns:
            Search results with ranked entries
        """
        start_time = datetime.now()
        
        try:
            self._logger.debug(f"Searching graph memory with query: {query.query}")
            
            # Parse query for graph-specific operations
            search_results = await self._execute_graph_search(query)
            
            # Convert nodes back to memory entries
            result_entries = []
            for node_id, score in search_results:
                memory_id = self._node_to_memory.get(node_id)
                if memory_id:
                    entry = await self.retrieve(memory_id)
                    if entry:
                        # Add graph-specific metadata
                        entry.metadata['graph_score'] = score
                        entry.metadata['node_id'] = node_id
                        entry.metadata['node_connections'] = len(self._adjacency_list.get(node_id, []))
                        result_entries.append(entry)
            
            # Limit results
            result_entries = result_entries[:query.limit]
            
            end_time = datetime.now()
            search_time = (end_time - start_time).total_seconds()
            
            # Record search in history
            search_record = {
                'timestamp': start_time.isoformat(),
                'query': query.query,
                'results_count': len(result_entries),
                'search_time': search_time,
                'query_type': 'graph_search'
            }
            self._query_history.append(search_record)
            
            return MemorySearchResult(
                entries=result_entries,
                total_count=len(search_results),
                search_time=search_time,
                metadata={
                    'graph_nodes_searched': len(self._nodes),
                    'graph_edges_traversed': len(self._edges),
                    'search_method': 'graph_traversal'
                }
            )
            
        except Exception as e:
            self._logger.error(f"Error searching graph memory: {e}")
            return MemorySearchResult(
                entries=[],
                total_count=0,
                search_time=(datetime.now() - start_time).total_seconds(),
                metadata={'error': str(e)}
            )
    
    async def update(self, entry_id: str, updated_entry: MemoryEntry) -> bool:
        """
        Update an existing memory entry and its graph representation.
        
        Args:
            entry_id: ID of the entry to update
            updated_entry: Updated memory entry
            
        Returns:
            True if update successful
        """
        try:
            node_id = self._memory_to_node.get(entry_id)
            if not node_id:
                self._logger.warning(f"Memory entry {entry_id} not found for update")
                return False
            
            node = self._nodes.get(node_id)
            if not node:
                return False
            
            # Update node data
            node.data = {
                'content': updated_entry.content,
                'tags': updated_entry.tags,
                'importance_score': updated_entry.importance_score
            }
            node.properties = updated_entry.metadata.copy()
            node.updated_at = datetime.now()
            
            # Update indexes
            await self._update_node_indexes(node)
            
            # Recreate relationships if content changed significantly
            # (This is a simplified approach - in practice, you'd compare content similarity)
            await self._update_node_relationships(node)
            
            self._logger.debug(f"Successfully updated memory entry: {entry_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Error updating memory entry {entry_id}: {e}")
            return False
    
    async def delete(self, entry_id: str) -> bool:
        """
        Delete a memory entry and its graph representation.
        
        Args:
            entry_id: ID of the entry to delete
            
        Returns:
            True if deletion successful
        """
        try:
            node_id = self._memory_to_node.get(entry_id)
            if not node_id:
                self._logger.warning(f"Memory entry {entry_id} not found for deletion")
                return False
            
            # Remove node and all its edges
            await self._remove_node(node_id)
            
            # Clean up mappings
            del self._memory_to_node[entry_id]
            del self._node_to_memory[node_id]
            
            self._logger.debug(f"Successfully deleted memory entry: {entry_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Error deleting memory entry {entry_id}: {e}")
            return False
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph memory system."""
        # Calculate graph metrics
        node_types_count = {}
        for node_type in NodeType:
            node_types_count[node_type.value] = len(self._node_type_index.get(node_type, set()))
        
        edge_types_count = {}
        for edge_type in EdgeType:
            edge_types_count[edge_type.value] = len(self._edge_type_index.get(edge_type, set()))
        
        # Calculate connectivity metrics
        total_connections = sum(len(edges) for edges in self._adjacency_list.values())
        avg_connections = total_connections / len(self._nodes) if self._nodes else 0
        
        return {
            'total_nodes': len(self._nodes),
            'total_edges': len(self._edges),
            'max_nodes': self._max_nodes,
            'max_edges': self._max_edges,
            'node_types': node_types_count,
            'edge_types': edge_types_count,
            'connectivity': {
                'total_connections': total_connections,
                'average_connections_per_node': avg_connections,
                'connected_components': await self._count_connected_components()
            },
            'indexes': {
                'property_indexes': len(self._property_index),
                'indexing_enabled': self._enable_indexing
            },
            'recent_queries': len(self._query_history)
        }
    
    async def create_embedding(self, content: str) -> List[float]:
        """
        Create embedding for graph-based similarity (simplified).
        
        Args:
            content: Content to embed
            
        Returns:
            Simple feature vector
        """
        # Simple feature extraction for graph memory
        features = []
        
        # Text-based features
        word_count = len(content.split())
        char_count = len(content)
        features.extend([word_count / 100.0, char_count / 1000.0])
        
        # Keyword-based features
        keywords = ['important', 'urgent', 'task', 'project', 'meeting', 'deadline']
        for keyword in keywords:
            features.append(1.0 if keyword in content.lower() else 0.0)
        
        return features
    
    async def _add_node(self, node: GraphNode):
        """Add a node to the graph with indexing."""
        self._nodes[node.id] = node
        
        # Initialize adjacency lists
        self._adjacency_list[node.id] = []
        self._reverse_adjacency_list[node.id] = []
        
        # Update type index
        if self._enable_indexing:
            if node.node_type not in self._node_type_index:
                self._node_type_index[node.node_type] = set()
            self._node_type_index[node.node_type].add(node.id)
            
            # Update property indexes
            await self._update_node_indexes(node)
    
    async def _remove_node(self, node_id: str):
        """Remove a node and all its edges."""
        if node_id not in self._nodes:
            return
        
        node = self._nodes[node_id]
        
        # Remove all edges connected to this node
        edges_to_remove = self._adjacency_list.get(node_id, []).copy()
        edges_to_remove.extend(self._reverse_adjacency_list.get(node_id, []))
        
        for edge_id in set(edges_to_remove):  # Use set to avoid duplicates
            await self._remove_edge(edge_id)
        
        # Remove from type index
        if self._enable_indexing and node.node_type in self._node_type_index:
            self._node_type_index[node.node_type].discard(node_id)
        
        # Remove from property indexes
        await self._remove_node_from_indexes(node)
        
        # Remove node
        del self._nodes[node_id]
        del self._adjacency_list[node_id]
        del self._reverse_adjacency_list[node_id]
    
    async def _add_edge(self, edge: GraphEdge):
        """Add an edge to the graph with indexing."""
        if len(self._edges) >= self._max_edges:
            await self._cleanup_old_edges()
        
        self._edges[edge.id] = edge
        
        # Update adjacency lists
        self._adjacency_list[edge.source_id].append(edge.id)
        self._reverse_adjacency_list[edge.target_id].append(edge.id)
        
        # Update type index
        if self._enable_indexing:
            if edge.edge_type not in self._edge_type_index:
                self._edge_type_index[edge.edge_type] = set()
            self._edge_type_index[edge.edge_type].add(edge.id)
    
    async def _remove_edge(self, edge_id: str):
        """Remove an edge from the graph."""
        if edge_id not in self._edges:
            return
        
        edge = self._edges[edge_id]
        
        # Remove from adjacency lists
        if edge.source_id in self._adjacency_list:
            self._adjacency_list[edge.source_id] = [
                e_id for e_id in self._adjacency_list[edge.source_id] if e_id != edge_id
            ]
        
        if edge.target_id in self._reverse_adjacency_list:
            self._reverse_adjacency_list[edge.target_id] = [
                e_id for e_id in self._reverse_adjacency_list[edge.target_id] if e_id != edge_id
            ]
        
        # Remove from type index
        if self._enable_indexing and edge.edge_type in self._edge_type_index:
            self._edge_type_index[edge.edge_type].discard(edge_id)
        
        # Remove edge
        del self._edges[edge_id]
    
    async def _update_node_indexes(self, node: GraphNode):
        """Update property indexes for a node."""
        if not self._enable_indexing:
            return
        
        for key, value in node.properties.items():
            if key not in self._property_index:
                self._property_index[key] = {}
            if value not in self._property_index[key]:
                self._property_index[key][value] = set()
            self._property_index[key][value].add(node.id)
    
    async def _remove_node_from_indexes(self, node: GraphNode):
        """Remove node from property indexes."""
        if not self._enable_indexing:
            return
        
        for key, value in node.properties.items():
            if key in self._property_index and value in self._property_index[key]:
                self._property_index[key][value].discard(node.id)
                if not self._property_index[key][value]:
                    del self._property_index[key][value]
                if not self._property_index[key]:
                    del self._property_index[key]
    
    async def _create_automatic_relationships(self, node: GraphNode):
        """Create automatic relationships between nodes."""
        # Find similar nodes and create relationships
        similar_nodes = await self._find_similar_nodes(node)
        
        for similar_node_id, similarity in similar_nodes:
            if similarity >= self._relationship_threshold:
                edge_id = f"rel_{node.id}_{similar_node_id}_{uuid.uuid4().hex[:8]}"
                edge = GraphEdge(
                    id=edge_id,
                    source_id=node.id,
                    target_id=similar_node_id,
                    edge_type=EdgeType.SIMILAR_TO,
                    weight=similarity,
                    properties={'auto_created': True, 'similarity_score': similarity},
                    created_at=datetime.now()
                )
                await self._add_edge(edge)
    
    async def _find_similar_nodes(self, node: GraphNode) -> List[Tuple[str, float]]:
        """Find nodes similar to the given node."""
        similar_nodes = []
        node_content = str(node.data.get('content', ''))
        node_tags = set(node.data.get('tags', []))
        
        for other_id, other_node in self._nodes.items():
            if other_id == node.id or other_node.node_type != NodeType.MEMORY:
                continue
            
            # Calculate similarity based on content and tags
            other_content = str(other_node.data.get('content', ''))
            other_tags = set(other_node.data.get('tags', []))
            
            # Simple similarity calculation
            content_similarity = self._calculate_text_similarity(node_content, other_content)
            tag_similarity = len(node_tags & other_tags) / len(node_tags | other_tags) if node_tags | other_tags else 0
            
            # Combined similarity
            similarity = (content_similarity + tag_similarity) / 2
            
            if similarity > 0.1:  # Minimum threshold
                similar_nodes.append((other_id, similarity))
        
        # Sort by similarity and return top results
        similar_nodes.sort(key=lambda x: x[1], reverse=True)
        return similar_nodes[:5]  # Top 5 similar nodes
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    async def _extract_concepts(self, node: GraphNode):
        """Extract concept nodes from memory content."""
        content = str(node.data.get('content', ''))
        tags = node.data.get('tags', [])
        
        # Extract concepts from tags
        for tag in tags:
            concept_id = f"concept_{tag.lower().replace(' ', '_')}"
            
            if concept_id not in self._nodes:
                # Create concept node
                concept_node = GraphNode(
                    id=concept_id,
                    node_type=NodeType.CONCEPT,
                    data={'name': tag, 'type': 'tag_concept'},
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    properties={'concept_type': 'tag', 'name': tag}
                )
                await self._add_node(concept_node)
            
            # Create relationship from memory to concept
            edge_id = f"concept_rel_{node.id}_{concept_id}"
            if edge_id not in self._edges:
                edge = GraphEdge(
                    id=edge_id,
                    source_id=node.id,
                    target_id=concept_id,
                    edge_type=EdgeType.RELATED_TO,
                    weight=1.0,
                    properties={'relationship_type': 'has_concept'},
                    created_at=datetime.now()
                )
                await self._add_edge(edge)
    
    async def _execute_graph_search(self, query: MemoryQuery) -> List[Tuple[str, float]]:
        """Execute graph-based search query."""
        results = []
        
        # Search by content matching
        query_words = set(query.query.lower().split())
        
        for node_id, node in self._nodes.items():
            if node.node_type != NodeType.MEMORY:
                continue
            
            score = 0.0
            
            # Content similarity
            content = str(node.data.get('content', ''))
            content_words = set(content.lower().split())
            if content_words & query_words:
                content_score = len(content_words & query_words) / len(query_words)
                score += content_score * 0.7
            
            # Tag matching
            tags = node.data.get('tags', [])
            tag_words = set(' '.join(tags).lower().split())
            if tag_words & query_words:
                tag_score = len(tag_words & query_words) / len(query_words)
                score += tag_score * 0.3
            
            # Connection bonus (well-connected nodes score higher)
            connections = len(self._adjacency_list.get(node_id, []))
            connection_bonus = min(connections / 10.0, 0.2)  # Max 0.2 bonus
            score += connection_bonus
            
            if score > 0:
                results.append((node_id, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    async def _update_node_relationships(self, node: GraphNode):
        """Update relationships for a node after content change."""
        # This is a simplified approach - remove old auto-created relationships and create new ones
        edges_to_remove = []
        
        for edge_id in self._adjacency_list.get(node.id, []):
            edge = self._edges.get(edge_id)
            if edge and edge.properties.get('auto_created', False):
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            await self._remove_edge(edge_id)
        
        # Create new relationships
        await self._create_automatic_relationships(node)
    
    async def _cleanup_old_nodes(self):
        """Remove old nodes to make space."""
        # Sort nodes by importance and age
        nodes_by_importance = []
        for node_id, node in self._nodes.items():
            if node.node_type == NodeType.MEMORY:
                importance = node.data.get('importance_score', 0.5)
                age = (datetime.now() - node.created_at).total_seconds()
                # Lower score means less important and older
                score = importance - (age / 86400.0) * 0.1  # Reduce score by 0.1 per day
                nodes_by_importance.append((node_id, score))
        
        nodes_by_importance.sort(key=lambda x: x[1])
        
        # Remove lowest scoring nodes
        nodes_to_remove = nodes_by_importance[:len(nodes_by_importance) // 10]  # Remove 10%
        
        for node_id, _ in nodes_to_remove:
            await self._remove_node(node_id)
    
    async def _cleanup_old_edges(self):
        """Remove old edges to make space."""
        # Sort edges by weight and age
        edges_by_score = []
        for edge_id, edge in self._edges.items():
            age = (datetime.now() - edge.created_at).total_seconds()
            score = edge.weight - (age / 86400.0) * 0.05  # Reduce score by 0.05 per day
            edges_by_score.append((edge_id, score))
        
        edges_by_score.sort(key=lambda x: x[1])
        
        # Remove lowest scoring edges
        edges_to_remove = edges_by_score[:len(edges_by_score) // 10]  # Remove 10%
        
        for edge_id, _ in edges_to_remove:
            await self._remove_edge(edge_id)
    
    async def _count_connected_components(self) -> int:
        """Count the number of connected components in the graph."""
        visited = set()
        components = 0
        
        for node_id in self._nodes:
            if node_id not in visited:
                # BFS to find all nodes in this component
                queue = [node_id]
                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        # Add neighbors
                        for edge_id in self._adjacency_list.get(current, []):
                            edge = self._edges.get(edge_id)
                            if edge:
                                queue.append(edge.target_id)
                        for edge_id in self._reverse_adjacency_list.get(current, []):
                            edge = self._edges.get(edge_id)
                            if edge:
                                queue.append(edge.source_id)
                components += 1
        
        return components
    
    async def connect(self) -> bool:
        """Connect to the graph memory system."""
        try:
            self._logger.info(f"Connecting to GraphMemory {self.memory_id}")
            self.is_connected = True
            return True
        except Exception as e:
            self._logger.error(f"Failed to connect to GraphMemory {self.memory_id}: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from the graph memory system."""
        try:
            self._logger.info(f"Disconnecting from GraphMemory {self.memory_id}")
            self.is_connected = False
            return True
        except Exception as e:
            self._logger.error(f"Error disconnecting from GraphMemory {self.memory_id}: {e}")
            return False