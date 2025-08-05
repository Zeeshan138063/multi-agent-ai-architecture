"""
Service Registry - Central component discovery and management system.
Handles plugin registration, lifecycle, and dependency injection.
"""

import inspect
import importlib
import os
from typing import Dict, Any, List, Optional, Type, Union
from dataclasses import dataclass
from datetime import datetime
import logging

from .interfaces import AgentInterface, ToolInterface, MemoryInterface


@dataclass
class ComponentInfo:
    """Information about a registered component."""
    component_id: str
    component_type: str
    instance: Any
    config: Dict[str, Any]
    registration_time: datetime
    is_active: bool
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class ServiceRegistry:
    """
    Central registry for managing all system components.
    Provides plugin discovery, registration, and dependency injection.
    """
    
    def __init__(self):
        self._components: Dict[str, ComponentInfo] = {}
        self._component_types: Dict[str, List[str]] = {
            'agents': [],
            'tools': [],
            'memory': [],
            'services': [],
            'adapters': []
        }
        self._logger = logging.getLogger(__name__)
        self._initialization_order: List[str] = []
    
    def register_component(self, component_id: str, component_type: str, 
                          instance: Any, config: Dict[str, Any] = None, 
                          dependencies: List[str] = None) -> bool:
        """
        Register a component with the registry.
        
        Args:
            component_id: Unique identifier for the component
            component_type: Type category (agent, tool, memory, service, adapter)
            instance: The component instance
            config: Component configuration
            dependencies: List of component IDs this component depends on
            
        Returns:
            True if registration successful
        """
        if component_id in self._components:
            self._logger.warning(f"Component {component_id} already registered")
            return False
        
        config = config or {}
        dependencies = dependencies or []
        
        component_info = ComponentInfo(
            component_id=component_id,
            component_type=component_type,
            instance=instance,
            config=config,
            registration_time=datetime.now(),
            is_active=True,
            dependencies=dependencies
        )
        
        self._components[component_id] = component_info
        
        # Add to type index
        if component_type in self._component_types:
            self._component_types[component_type].append(component_id)
        else:
            self._component_types[component_type] = [component_id]
        
        self._logger.info(f"Registered {component_type}: {component_id}")
        return True
    
    def unregister_component(self, component_id: str) -> bool:
        """
        Unregister a component from the registry.
        
        Args:
            component_id: ID of component to unregister
            
        Returns:
            True if unregistration successful
        """
        if component_id not in self._components:
            self._logger.warning(f"Component {component_id} not found for unregistration")
            return False
        
        component_info = self._components[component_id]
        
        # Remove from type index
        if component_info.component_type in self._component_types:
            if component_id in self._component_types[component_info.component_type]:
                self._component_types[component_info.component_type].remove(component_id)
        
        # Cleanup component if it has cleanup method
        if hasattr(component_info.instance, 'cleanup'):
            try:
                if inspect.iscoroutinefunction(component_info.instance.cleanup):
                    import asyncio
                    asyncio.create_task(component_info.instance.cleanup())
                else:
                    component_info.instance.cleanup()
            except Exception as e:
                self._logger.error(f"Error during component cleanup: {e}")
        
        del self._components[component_id]
        self._logger.info(f"Unregistered component: {component_id}")
        return True
    
    def get_component(self, component_id: str) -> Optional[Any]:
        """
        Get a component instance by ID.
        
        Args:
            component_id: ID of the component
            
        Returns:
            Component instance or None if not found
        """
        component_info = self._components.get(component_id)
        return component_info.instance if component_info else None
    
    def get_components_by_type(self, component_type: str) -> List[Any]:
        """
        Get all components of a specific type.
        
        Args:
            component_type: Type of components to retrieve
            
        Returns:
            List of component instances
        """
        component_ids = self._component_types.get(component_type, [])
        return [self._components[cid].instance for cid in component_ids 
                if cid in self._components and self._components[cid].is_active]
    
    def get_component_info(self, component_id: str) -> Optional[ComponentInfo]:
        """Get detailed information about a component."""
        return self._components.get(component_id)
    
    def list_components(self, active_only: bool = True) -> Dict[str, ComponentInfo]:
        """
        List all registered components.
        
        Args:
            active_only: Only return active components
            
        Returns:
            Dictionary of component information
        """
        if active_only:
            return {cid: info for cid, info in self._components.items() if info.is_active}
        return self._components.copy()
    
    def activate_component(self, component_id: str) -> bool:
        """Activate a component."""
        if component_id in self._components:
            self._components[component_id].is_active = True
            return True
        return False
    
    def deactivate_component(self, component_id: str) -> bool:
        """Deactivate a component without unregistering."""
        if component_id in self._components:
            self._components[component_id].is_active = False
            return True
        return False
    
    async def initialize_components(self) -> Dict[str, bool]:
        """
        Initialize all registered components in dependency order.
        
        Returns:
            Dictionary mapping component IDs to initialization success
        """
        results = {}
        
        # Sort components by dependencies (basic topological sort)
        sorted_components = self._sort_by_dependencies()
        
        for component_id in sorted_components:
            component_info = self._components[component_id]
            try:
                if hasattr(component_info.instance, 'initialize'):
                    if inspect.iscoroutinefunction(component_info.instance.initialize):
                        success = await component_info.instance.initialize()
                    else:
                        success = component_info.instance.initialize()
                    results[component_id] = success
                    
                    if success:
                        self._logger.info(f"Initialized component: {component_id}")
                    else:
                        self._logger.error(f"Failed to initialize component: {component_id}")
                        component_info.is_active = False
                else:
                    results[component_id] = True
                    
            except Exception as e:
                self._logger.error(f"Error initializing {component_id}: {e}")
                results[component_id] = False
                component_info.is_active = False
        
        return results
    
    def _sort_by_dependencies(self) -> List[str]:
        """Sort components by their dependencies using topological sort."""
        # Simple implementation - in production, use proper topological sort
        sorted_ids = []
        remaining = set(self._components.keys())
        
        while remaining:
            # Find components with no unmet dependencies
            ready = []
            for cid in remaining:
                deps = self._components[cid].dependencies
                if all(dep in sorted_ids or dep not in self._components for dep in deps):
                    ready.append(cid)
            
            if not ready:
                # Circular dependency or missing dependency - add remaining arbitrarily
                ready = list(remaining)
            
            sorted_ids.extend(ready)
            remaining -= set(ready)
        
        return sorted_ids
    
    def discover_plugins(self, plugin_directory: str) -> Dict[str, List[str]]:
        """
        Discover and load plugins from a directory.
        
        Args:
            plugin_directory: Path to the plugins directory
            
        Returns:
            Dictionary mapping plugin types to discovered plugin names
        """
        discovered = {'agents': [], 'tools': [], 'memory': [], 'adapters': []}
        
        if not os.path.exists(plugin_directory):
            self._logger.warning(f"Plugin directory not found: {plugin_directory}")
            return discovered
        
        # Scan for plugin files
        for plugin_type in discovered.keys():
            type_dir = os.path.join(plugin_directory, plugin_type)
            if os.path.exists(type_dir):
                for file in os.listdir(type_dir):
                    if file.endswith('.py') and not file.startswith('__'):
                        plugin_name = file[:-3]  # Remove .py extension
                        discovered[plugin_type].append(plugin_name)
        
        self._logger.info(f"Discovered plugins: {discovered}")
        return discovered
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the registry."""
        stats = {
            'total_components': len(self._components),
            'active_components': len([c for c in self._components.values() if c.is_active]),
            'components_by_type': {t: len(ids) for t, ids in self._component_types.items()},
            'initialization_order': self._initialization_order
        }
        return stats