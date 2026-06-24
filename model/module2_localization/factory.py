# module2_localization/factory.py — Localization method factory with registration
from typing import Dict, Type, List
from .base import LocalizationBase


class LocalizationFactory:
    """Factory for creating and managing localization methods.
    
    Usage:
        factory = LocalizationFactory()
        methods = factory.create_all(config)
        for method in methods:
            pos, clk, details = method.solve(obs, sv_pos, sv_sys, add_info)
    """
    
    _registry: Dict[str, Type[LocalizationBase]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a localization method."""
        def decorator(method_cls: Type[LocalizationBase]):
            cls._registry[name] = method_cls
            return method_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, config: Dict = None) -> LocalizationBase:
        """Create a localization method by name."""
        if name not in cls._registry:
            raise ValueError(f"Unknown method '{name}'. Available: {cls.list_methods()}")
        return cls._registry[name](config=config, name=name)
    
    @classmethod
    def create_all(cls, config: Dict = None) -> List[LocalizationBase]:
        """Create all registered methods."""
        return [cls.create(name, config) for name in cls._registry]
    
    @classmethod
    def list_methods(cls) -> List[str]:
        """List all registered method names."""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._registry
