import logging
from typing import Optional
from rag.providers.kg_provider import get_kg_provider, BaseKGProvider
from api.config import settings

logger = logging.getLogger(__name__)

# Global instance of the active Knowledge Graph Provider
_kg_provider: Optional[BaseKGProvider] = None

def get_graph_db() -> BaseKGProvider:
    """
    Returns the active connected Knowledge Graph Provider instance.
    Must be initialized on application startup via init_graph_db().
    """
    global _kg_provider
    if _kg_provider is None:
        raise RuntimeError("KG Provider has not been initialized. Call init_graph_db() on startup.")
    return _kg_provider

def init_graph_db() -> BaseKGProvider:
    """
    Initializes and connects the KG Provider configured in settings.
    """
    global _kg_provider
    if _kg_provider is not None:
        logger.warning("KG Provider already initialized. Returning existing instance.")
        return _kg_provider

    provider_name = settings.kg_provider
    logger.info(f"Initializing KG Provider: {provider_name}...")
    
    # Instantiate and connect using the existing factory
    _kg_provider = get_kg_provider(provider_name)
    logger.info(f"KG Provider {provider_name} initialized and connected successfully.")
    return _kg_provider

def close_graph_db() -> None:
    """
    Closes the active KG Provider connection.
    """
    global _kg_provider
    if _kg_provider is not None:
        logger.info(f"Closing KG Provider connection ({settings.kg_provider})...")
        try:
            _kg_provider.close()
            logger.info("KG Provider connection closed successfully.")
        except Exception as e:
            logger.error(f"Error while closing KG Provider connection: {e}")
        finally:
            _kg_provider = None
