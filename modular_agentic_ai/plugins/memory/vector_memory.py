"""
Vector Memory - Memory system using vector embeddings for semantic search and storage.
Provides efficient similarity-based retrieval of stored memories.
"""

import asyncio
import logging
import json
import math
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid

from ...core.interfaces import MemoryInterface, MemoryEntry, MemoryQuery, MemorySearchResult


class VectorMemory(MemoryInterface):
    """
    Vector-based memory system using embeddings for semantic similarity search.
    
    Features:
    - Vector embeddings for content
    - Cosine similarity search
    - Metadata filtering
    - Batch operations
    - Memory consolidation
    """
    
    def __init__(self, memory_id: str, config: Dict[str, Any]):
        super().__init__(memory_id, config)
        self._logger = logging.getLogger(__name__)
        self._embedding_dimension = config.get('embedding_dimension', 768)
        self._similarity_metric = config.get('similarity_metric', 'cosine')
        self._max_entries = config.get('max_entries', 10000)
        self._consolidation_threshold = config.get('consolidation_threshold', 0.95)
        self._enable_metadata_indexing = config.get('enable_metadata_indexing', True)
        
        # Memory storage
        self._entries: Dict[str, MemoryEntry] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._metadata_index: Dict[str, List[str]] = {}  # metadata_key -> [entry_ids]
        self._tag_index: Dict[str, List[str]] = {}  # tag -> [entry_ids]
        
        # Performance tracking
        self._search_history: List[Dict[str, Any]] = []
        self._consolidation_history: List[Dict[str, Any]] = []
    
    async def store(self, entry: MemoryEntry) -> bool:
        """
        Store a memory entry with vector embedding.
        
        Args:
            entry: Memory entry to store
            
        Returns:
            True if storage successful
        """
        try:
            self._logger.debug(f"Storing memory entry: {entry.id}")
            
            # Check capacity
            if len(self._entries) >= self._max_entries:
                # Perform consolidation or remove old entries
                await self._perform_consolidation()
            
            # Generate embedding for content
            embedding = await self.create_embedding(str(entry.content))
            
            # Store entry and embedding
            self._entries[entry.id] = entry
            self._embeddings[entry.id] = embedding
            
            # Update indexes
            await self._update_indexes(entry)
            
            # Check for consolidation opportunities
            if self._should_consolidate(entry):
                await self._consolidate_similar_entries(entry)
            
            self._logger.debug(f"Successfully stored memory entry: {entry.id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Error storing memory entry {entry.id}: {e}")
            return False
    
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a specific memory entry by ID.
        
        Args:
            entry_id: ID of the memory entry
            
        Returns:
            Memory entry if found
        """
        try:
            return self._entries.get(entry_id)
        except Exception as e:
            self._logger.error(f"Error retrieving memory entry {entry_id}: {e}")
            return None
    
    async def search(self, query: MemoryQuery) -> MemorySearchResult:
        """
        Search memory entries using vector similarity and filters.
        
        Args:
            query: Search query with parameters
            
        Returns:
            Search results with ranked entries
        """
        start_time = datetime.now()
        
        try:
            self._logger.debug(f"Searching memories with query: {query.query}")
            
            # Generate query embedding
            query_embedding = await self.create_embedding(query.query)
            
            # Calculate similarities for all entries
            similarities = []
            for entry_id, entry_embedding in self._embeddings.items():
                similarity = self._calculate_similarity(query_embedding, entry_embedding)
                if similarity >= query.similarity_threshold:
                    similarities.append((entry_id, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Apply filters
            filtered_results = await self._apply_filters(similarities, query.filters or {})
            
            # Limit results
            limited_results = filtered_results[:query.limit]
            
            # Convert to memory entries
            result_entries = []
            for entry_id, similarity in limited_results:
                entry = self._entries.get(entry_id)
                if entry:
                    # Add similarity score to metadata
                    enhanced_entry = MemoryEntry(
                        id=entry.id,
                        content=entry.content,
                        timestamp=entry.timestamp,
                        tags=entry.tags,
                        metadata={**entry.metadata, 'similarity_score': similarity},
                        importance_score=entry.importance_score
                    )
                    result_entries.append(enhanced_entry)
            
            end_time = datetime.now()
            search_time = (end_time - start_time).total_seconds()
            
            # Record search in history
            search_record = {
                'timestamp': start_time.isoformat(),
                'query': query.query,
                'results_count': len(result_entries),
                'search_time': search_time,
                'similarity_threshold': query.similarity_threshold
            }
            self._search_history.append(search_record)
            
            # Maintain search history size
            if len(self._search_history) > 100:
                self._search_history.pop(0)
            
            return MemorySearchResult(
                entries=result_entries,
                total_count=len(filtered_results),
                search_time=search_time,
                metadata={
                    'query_embedding_dimension': len(query_embedding),
                    'total_similarities_calculated': len(similarities),
                    'filtered_results': len(filtered_results),
                    'similarity_metric': self._similarity_metric
                }
            )
            
        except Exception as e:
            self._logger.error(f"Error searching memories: {e}")
            return MemorySearchResult(
                entries=[],
                total_count=0,
                search_time=(datetime.now() - start_time).total_seconds(),
                metadata={'error': str(e)}
            )
    
    async def update(self, entry_id: str, updated_entry: MemoryEntry) -> bool:
        """
        Update an existing memory entry.
        
        Args:
            entry_id: ID of the entry to update
            updated_entry: Updated memory entry
            
        Returns:
            True if update successful
        """
        try:
            if entry_id not in self._entries:
                self._logger.warning(f"Memory entry {entry_id} not found for update")
                return False
            
            # Remove old indexes
            old_entry = self._entries[entry_id]
            await self._remove_from_indexes(old_entry)
            
            # Generate new embedding if content changed
            if str(old_entry.content) != str(updated_entry.content):
                new_embedding = await self.create_embedding(str(updated_entry.content))
                self._embeddings[entry_id] = new_embedding
            
            # Update entry
            self._entries[entry_id] = updated_entry
            
            # Update indexes
            await self._update_indexes(updated_entry)
            
            self._logger.debug(f"Successfully updated memory entry: {entry_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Error updating memory entry {entry_id}: {e}")
            return False
    
    async def delete(self, entry_id: str) -> bool:
        """
        Delete a memory entry.
        
        Args:
            entry_id: ID of the entry to delete
            
        Returns:
            True if deletion successful
        """
        try:
            if entry_id not in self._entries:
                self._logger.warning(f"Memory entry {entry_id} not found for deletion")
                return False
            
            # Remove from indexes
            entry = self._entries[entry_id]
            await self._remove_from_indexes(entry)
            
            # Remove entry and embedding
            del self._entries[entry_id]
            del self._embeddings[entry_id]
            
            self._logger.debug(f"Successfully deleted memory entry: {entry_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Error deleting memory entry {entry_id}: {e}")
            return False
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        return {
            'total_entries': len(self._entries),
            'total_embeddings': len(self._embeddings),
            'embedding_dimension': self._embedding_dimension,
            'similarity_metric': self._similarity_metric,
            'max_entries': self._max_entries,
            'metadata_indexes': len(self._metadata_index),
            'tag_indexes': len(self._tag_index),
            'recent_searches': len(self._search_history),
            'consolidations_performed': len(self._consolidation_history),
            'memory_usage': {
                'entries_size': len(self._entries),
                'embeddings_size': len(self._embeddings),
                'indexes_size': len(self._metadata_index) + len(self._tag_index)
            }
        }
    
    async def create_embedding(self, content: str) -> List[float]:
        """
        Create vector embedding for content.
        
        Args:
            content: Content to embed
            
        Returns:
            Vector embedding
        """
        # Simulate embedding generation
        # In a real implementation, this would use a proper embedding model
        
        # Simple hash-based embedding simulation
        content_hash = hash(content.lower())
        random.seed(content_hash)
        
        # Generate random embedding with some consistency
        embedding = []
        for i in range(self._embedding_dimension):
            # Create somewhat meaningful embeddings based on content
            value = random.gauss(0, 1)
            # Add some content-based features
            if 'important' in content.lower():
                value += 0.5
            if 'urgent' in content.lower():
                value += 0.3
            if 'task' in content.lower():
                value += 0.2
            
            embedding.append(value)
        
        # Normalize embedding
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score
        """
        if self._similarity_metric == 'cosine':
            return self._cosine_similarity(embedding1, embedding2)
        elif self._similarity_metric == 'dot_product':
            return sum(a * b for a, b in zip(embedding1, embedding2))
        elif self._similarity_metric == 'euclidean':
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(embedding1, embedding2)))
            return 1 / (1 + distance)  # Convert distance to similarity
        else:
            return self._cosine_similarity(embedding1, embedding2)
    
    def _cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = math.sqrt(sum(a * a for a in embedding1))
        magnitude2 = math.sqrt(sum(b * b for b in embedding2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    async def _apply_filters(self, similarities: List[Tuple[str, float]], 
                           filters: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Apply filters to search results.
        
        Args:
            similarities: List of (entry_id, similarity) tuples
            filters: Filters to apply
            
        Returns:
            Filtered results
        """
        if not filters:
            return similarities
        
        filtered = []
        
        for entry_id, similarity in similarities:
            entry = self._entries.get(entry_id)
            if not entry:
                continue
            
            # Apply filters
            if self._entry_matches_filters(entry, filters):
                filtered.append((entry_id, similarity))
        
        return filtered
    
    def _entry_matches_filters(self, entry: MemoryEntry, filters: Dict[str, Any]) -> bool:
        """Check if entry matches the given filters."""
        for filter_key, filter_value in filters.items():
            if filter_key == 'tags':
                # Tag filter
                required_tags = filter_value if isinstance(filter_value, list) else [filter_value]
                if not any(tag in entry.tags for tag in required_tags):
                    return False
            
            elif filter_key == 'importance_min':
                # Minimum importance filter
                if entry.importance_score < filter_value:
                    return False
            
            elif filter_key == 'importance_max':
                # Maximum importance filter
                if entry.importance_score > filter_value:
                    return False
            
            elif filter_key == 'date_from':
                # Date range filter (from)
                filter_date = datetime.fromisoformat(filter_value) if isinstance(filter_value, str) else filter_value
                if entry.timestamp < filter_date:
                    return False
            
            elif filter_key == 'date_to':
                # Date range filter (to)
                filter_date = datetime.fromisoformat(filter_value) if isinstance(filter_value, str) else filter_value
                if entry.timestamp > filter_date:
                    return False
            
            elif filter_key.startswith('metadata.'):
                # Metadata filter
                metadata_key = filter_key[9:]  # Remove 'metadata.' prefix
                if metadata_key not in entry.metadata or entry.metadata[metadata_key] != filter_value:
                    return False
        
        return True
    
    async def _update_indexes(self, entry: MemoryEntry):
        """Update indexes for the given entry."""
        if not self._enable_metadata_indexing:
            return
        
        # Update tag index
        for tag in entry.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = []
            if entry.id not in self._tag_index[tag]:
                self._tag_index[tag].append(entry.id)
        
        # Update metadata index
        for key, value in entry.metadata.items():
            index_key = f"metadata.{key}.{value}"
            if index_key not in self._metadata_index:
                self._metadata_index[index_key] = []
            if entry.id not in self._metadata_index[index_key]:
                self._metadata_index[index_key].append(entry.id)
    
    async def _remove_from_indexes(self, entry: MemoryEntry):
        """Remove entry from indexes."""
        if not self._enable_metadata_indexing:
            return
        
        # Remove from tag index
        for tag in entry.tags:
            if tag in self._tag_index and entry.id in self._tag_index[tag]:
                self._tag_index[tag].remove(entry.id)
                if not self._tag_index[tag]:
                    del self._tag_index[tag]
        
        # Remove from metadata index
        for key, value in entry.metadata.items():
            index_key = f"metadata.{key}.{value}"
            if index_key in self._metadata_index and entry.id in self._metadata_index[index_key]:
                self._metadata_index[index_key].remove(entry.id)
                if not self._metadata_index[index_key]:
                    del self._metadata_index[index_key]
    
    def _should_consolidate(self, entry: MemoryEntry) -> bool:
        """Determine if consolidation should be performed for this entry."""
        # Simple heuristic: consolidate if we have too many entries
        return len(self._entries) > self._max_entries * 0.8
    
    async def _consolidate_similar_entries(self, entry: MemoryEntry):
        """Consolidate similar entries to save space."""
        if entry.id not in self._embeddings:
            return
        
        entry_embedding = self._embeddings[entry.id]
        similar_entries = []
        
        # Find highly similar entries
        for other_id, other_embedding in self._embeddings.items():
            if other_id != entry.id:
                similarity = self._calculate_similarity(entry_embedding, other_embedding)
                if similarity >= self._consolidation_threshold:
                    similar_entries.append((other_id, similarity))
        
        if similar_entries:
            # Consolidate with the most similar entry
            similar_entries.sort(key=lambda x: x[1], reverse=True)
            most_similar_id, similarity = similar_entries[0]
            
            await self._merge_entries(entry.id, most_similar_id)
            
            # Record consolidation
            consolidation_record = {
                'timestamp': datetime.now().isoformat(),
                'primary_entry': entry.id,
                'merged_entry': most_similar_id,
                'similarity': similarity,
                'entries_before': len(self._entries)
            }
            self._consolidation_history.append(consolidation_record)
    
    async def _merge_entries(self, primary_id: str, secondary_id: str):
        """Merge two memory entries."""
        try:
            primary = self._entries.get(primary_id)
            secondary = self._entries.get(secondary_id)
            
            if not primary or not secondary:
                return
            
            # Merge content and metadata
            merged_content = f"{primary.content}\n[MERGED]: {secondary.content}"
            merged_tags = list(set(primary.tags + secondary.tags))
            merged_metadata = {**secondary.metadata, **primary.metadata}
            merged_metadata['merged_from'] = secondary_id
            merged_metadata['merge_timestamp'] = datetime.now().isoformat()
            
            # Create merged entry
            merged_entry = MemoryEntry(
                id=primary_id,
                content=merged_content,
                timestamp=primary.timestamp,  # Keep original timestamp
                tags=merged_tags,
                metadata=merged_metadata,
                importance_score=max(primary.importance_score, secondary.importance_score)
            )
            
            # Update primary entry
            await self.update(primary_id, merged_entry)
            
            # Delete secondary entry
            await self.delete(secondary_id)
            
            self._logger.debug(f"Merged entries {primary_id} and {secondary_id}")
            
        except Exception as e:
            self._logger.error(f"Error merging entries {primary_id} and {secondary_id}: {e}")
    
    async def _perform_consolidation(self):
        """Perform memory consolidation to free space."""
        if len(self._entries) < self._max_entries:
            return
        
        # Simple consolidation: remove oldest entries with low importance
        entries_by_age_importance = [
            (entry_id, entry.timestamp, entry.importance_score)
            for entry_id, entry in self._entries.items()
        ]
        
        # Sort by importance (ascending) then by age (ascending)
        entries_by_age_importance.sort(key=lambda x: (x[2], x[1]))
        
        # Remove the least important, oldest entries
        entries_to_remove = entries_by_age_importance[:len(self._entries) - self._max_entries + 100]
        
        for entry_id, _, _ in entries_to_remove:
            await self.delete(entry_id)
        
        self._logger.info(f"Consolidated memory: removed {len(entries_to_remove)} entries")
    
    async def connect(self) -> bool:
        """Connect to the vector memory system."""
        try:
            self._logger.info(f"Connecting to VectorMemory {self.memory_id}")
            self.is_connected = True
            return True
        except Exception as e:
            self._logger.error(f"Failed to connect to VectorMemory {self.memory_id}: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from the vector memory system."""
        try:
            self._logger.info(f"Disconnecting from VectorMemory {self.memory_id}")
            self.is_connected = False
            return True
        except Exception as e:
            self._logger.error(f"Error disconnecting from VectorMemory {self.memory_id}: {e}")
            return False